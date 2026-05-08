#!/usr/bin/env python3
"""
RF-DETR Seg + PyQt окно + перемотка видео: подсчёт людей при пересечении линии DOWN.

Что исправлено:
  • человек живёт TRACK_KEEP_ALIVE_SECONDS после пропажи детекта;
  • уже засчитанный потерянный трек НЕ забирает новых людей далеко от себя;
  • duplicate-фильтр больше не подавляет разных людей, идущих рядом;
  • при перемотке трекер и счётчик сбрасываются только для текущего просмотра;
  • cv2.imshow не используется — показ через PyQt QLabel;
  • сохранение событий: JPG + events.jsonl.

Запуск с окном:
  python rfdetr_live_seg_new.py

Запуск без окна:
  RFDETR_HEADLESS=1 python rfdetr_live_seg_new.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from rfdetr import RFDETRSegMedium

from PyQt5.QtCore import QEvent, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML_PATH = BASE_DIR / "new_dataset" / "data.yaml"
SEG_CHECKPOINT_PATH = BASE_DIR / "new_dataset" / "checkpoint_best_total.pth"

# Можно указать файл, папку с видео, RTSP/URL или индекс камеры.
# Примеры:
# LIVE_SOURCE = 0
# LIVE_SOURCE = "rtsp://user:pass@192.168.1.10:554/stream1"
# LIVE_SOURCE = str(BASE_DIR / "recordings_2" / "cam3" / "2026-05-06_09-22-55.mp4")
LIVE_SOURCE: int | str | list[int | str] = str(BASE_DIR / "recordings_2" / "cam3")

PERSON_CLASS_NAME = "person"
CONF_THRESHOLD = 0.50
DETECT_EVERY_N = 1

# Линия задаётся двумя нормализованными точками [0..1].
LINE_NORM: list[tuple[float, float]] = [
    (0.398038, 0.372452),
    (0.789723, 0.389616)
]

# DOWN = нижняя точка bbox перешла с верхней стороны линии на нижнюю.
LINE_MARGIN_PX = 14.0
SEGMENT_GATE_MARGIN_PX = 30.0

# --- Трекинг / уникальность человека ---
TRACK_KEEP_ALIVE_SECONDS = 15.0
TRACK_HISTORY = 80
MIN_TRACK_HISTORY_FOR_EVENT = 2

# Активный трек: обычная привязка.
TRACK_MAX_CENTER_DISTANCE = 220.0
TRACK_MIN_IOU_FOR_MATCH = 0.02

# Потерянный, но ещё НЕ засчитанный человек: можно искать шире.
REID_MAX_CENTER_DISTANCE = 280.0

# Потерянный и уже засчитанный человек: цепляем обратно только близко.
# Иначе он будет “съедать” новых людей, которые проходят в той же зоне позже.
REID_MAX_CENTER_DISTANCE_COUNTED = 120.0
REID_LOST_PENALTY_PER_FRAME = 3.0

# Если два кандидата похожи, активный/свежий трек должен выигрывать у старого потерянного.
COUNTED_REID_EXTRA_PENALTY = 60.0

# Минимальный размер person.
MIN_PERSON_WIDTH_PX = 24
MIN_PERSON_HEIGHT_PX = 40
MIN_PERSON_AREA_PX2 = 1200

# Duplicate-фильтр: защита от одного и того же события в соседних кадрах.
# Важно: после 10 кадров подавляем только почти тот же bbox, а не просто близкий центр.
DUPLICATE_SUPPRESS_SECONDS = 2.0
DUPLICATE_IOU_THRESHOLD = 0.92
DUPLICATE_CENTER_DISTANCE = 18.0
DUPLICATE_LATE_IOU_THRESHOLD = 0.80
DUPLICATE_MAX_FRAME_GAP = 6

# Если человек появился после перекрытия уже около/ниже линии
# и потом явно ушёл глубже вниз — считаем его.
OCCLUSION_START_MAX_BELOW_D = 90.0
OCCLUSION_MIN_BOTTOM_DELTA = 55.0
OCCLUSION_MIN_CENTER_DELTA = 20.0
OCCLUSION_MAX_HISTORY = 18
OCCLUSION_MIN_CONF = 0.80

# Поздний старт: если человек впервые появился уже около/на линии.
LATE_START_MIN_CENTER_DELTA = 8.0
LATE_START_MIN_BOTTOM_DELTA = 16.0
MIN_RECENT_TRAVEL_PX = 10.0

# Защита от ложных late_start на краях кадра.
LATE_START_MIN_CONF = 0.85
EVENT_REJECT_BORDER_MARGIN_PX = 6

# Отображение и сохранение.
USE_PYQT_VIEWER = os.environ.get("RFDETR_HEADLESS", "0") != "1"
SAVE_EVENTS = True
SAVE_DEBUG_VIDEO = False
OUTPUT_ROOT = BASE_DIR / "rfdetr_person_crossing_logs"
OUTPUT_VIDEO_NAME = "annotated_output.mp4"

WINDOW_TITLE = "RF-DETR person crossing counter | Q / Esc — выход"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

FILE_PLAYBACK_SLOWDOWN = 1.0
FILE_PLAYBACK_CAP_FPS = 25.0

# =========================
# BASIC UTILS
# =========================


def load_class_names_from_yaml(yaml_path: Path) -> list[str]:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if isinstance(names, list):
        return names
    raise ValueError(f"В {yaml_path} поле names должно быть list или dict")


def frame_to_model_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        bgr = frame
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def center_xy(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def bottom_center_xy(box: np.ndarray) -> np.ndarray:
    x1, _y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


def box_wh_area(box: np.ndarray) -> tuple[float, float, float]:
    x1, y1, x2, y2 = map(float, box)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w, h, w * h


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def side_of_line(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]))


def denorm_line(line_norm: list[tuple[float, float]], frame_w: int, frame_h: int) -> tuple[np.ndarray, np.ndarray]:
    if len(line_norm) < 2:
        raise ValueError("LINE_NORM должен содержать минимум 2 точки")

    p1 = np.array([line_norm[0][0] * frame_w, line_norm[0][1] * frame_h], dtype=np.float32)
    p2 = np.array([line_norm[1][0] * frame_w, line_norm[1][1] * frame_h], dtype=np.float32)
    return p1, p2


def infer_bottom_side_sign(frame_w: int, frame_h: int, line_a: np.ndarray, line_b: np.ndarray) -> int:
    top_point = np.array([frame_w * 0.5, frame_h * 0.05], dtype=np.float32)
    bottom_point = np.array([frame_w * 0.5, frame_h * 0.95], dtype=np.float32)

    top_side = np.sign(side_of_line(top_point, line_a, line_b))
    bottom_side = np.sign(side_of_line(bottom_point, line_a, line_b))

    if bottom_side == 0:
        bottom_side = 1
    if top_side == bottom_side:
        bottom_side = -top_side if top_side != 0 else 1

    return int(bottom_side)


def signed_distance_to_line(
    point: np.ndarray,
    line_a: np.ndarray,
    line_b: np.ndarray,
    bottom_side_sign: int,
) -> float:
    line_len = float(np.linalg.norm(line_b - line_a))
    if line_len < 1e-6:
        return 0.0
    return side_of_line(point, line_a, line_b) / line_len * float(bottom_side_sign)

def projection_t_on_segment(point: np.ndarray, line_a: np.ndarray, line_b: np.ndarray) -> float:
    """
    t=0 — начало отрезка, t=1 — конец отрезка.
    t<0 или t>1 значит точка проецируется за пределы короткого отрезка.
    """
    ab = line_b - line_a
    denom = float(np.dot(ab, ab))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(point - line_a, ab) / denom)


def point_near_segment_gate(
    point: np.ndarray,
    line_a: np.ndarray,
    line_b: np.ndarray,
    margin_px: float = SEGMENT_GATE_MARGIN_PX,
) -> bool:
    """
    Разрешает событие только если проекция точки попадает на отрезок
    с небольшим запасом margin_px по краям.
    """
    ab_len = float(np.linalg.norm(line_b - line_a))
    if ab_len < 1e-6:
        return False

    margin_t = margin_px / ab_len
    t = projection_t_on_segment(point, line_a, line_b)
    return -margin_t <= t <= 1.0 + margin_t


def box_near_segment_gate(
    box: np.ndarray,
    line_a: np.ndarray,
    line_b: np.ndarray,
    margin_px: float = SEGMENT_GATE_MARGIN_PX,
) -> bool:
    """
    Считаем объект относящимся к линии, если хотя бы центр низа,
    центр bbox или нижние углы проецируются на короткий отрезок.
    """
    x1, y1, x2, y2 = map(float, box)
    points = [
        bottom_center_xy(box),
        center_xy(box),
        np.array([x1, y2], dtype=np.float32),
        np.array([x2, y2], dtype=np.float32),
    ]
    return any(point_near_segment_gate(p, line_a, line_b, margin_px) for p in points)

def stable_side(distance: float, margin: float) -> str:
    if distance > margin:
        return "below"
    if distance < -margin:
        return "above"
    return "on"


def is_valid_person_size(box: np.ndarray) -> bool:
    w, h, area = box_wh_area(box)
    return w >= MIN_PERSON_WIDTH_PX and h >= MIN_PERSON_HEIGHT_PX and area >= MIN_PERSON_AREA_PX2

def touches_frame_border(box: np.ndarray, frame_w: int, frame_h: int, margin: int = EVENT_REJECT_BORDER_MARGIN_PX) -> bool:
    x1, y1, x2, y2 = map(float, box)
    return (
        x1 <= margin
        or y1 <= margin
        or x2 >= frame_w - 1 - margin
        or y2 >= frame_h - 1 - margin
    )

def source_label(src: int | str) -> str:
    if isinstance(src, int):
        return f"camera_{src}"
    return Path(str(src)).name or str(src)


def source_stem(src: int | str) -> str:
    if isinstance(src, int):
        return f"camera_{src}"
    p = Path(str(src))
    return p.stem if p.suffix else (p.name or "source")


def is_video_file(src: int | str) -> bool:
    return isinstance(src, str) and Path(src).expanduser().is_file() and Path(src).suffix.lower() in VIDEO_EXTENSIONS


def list_video_files(folder: str | Path) -> list[str]:
    base = Path(folder).expanduser()
    return sorted(str(p) for p in base.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def expand_sources(src: int | str | Iterable[int | str]) -> list[int | str]:
    if isinstance(src, (list, tuple)):
        return list(src)
    if isinstance(src, int):
        return [src]

    p = Path(str(src)).expanduser()
    if p.is_dir():
        return list_video_files(p)
    return [src]


def open_capture(src: int | str) -> cv2.VideoCapture | None:
    if isinstance(src, int):
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            return cap
        cap.release()

        if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
            cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
            cap.release()

        return None

    cap = cv2.VideoCapture(str(src))
    if cap.isOpened():
        return cap
    cap.release()
    return None


def playback_wait_ms(cap: cv2.VideoCapture, src: int | str) -> int:
    if not is_video_file(src):
        return 1

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1.0 or fps > 120.0:
        fps = 25.0
    fps = min(fps, FILE_PLAYBACK_CAP_FPS)
    return max(1, round((1000.0 / fps) * FILE_PLAYBACK_SLOWDOWN))


def make_run_dir() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT_ROOT / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

# =========================
# DETECTION
# =========================


@dataclass
class Detection:
    box: np.ndarray
    conf: float
    class_name: str
    mask: np.ndarray | None = None


def load_person_detections(model: RFDETRSegMedium, frame: np.ndarray, class_names: list[str]) -> list[Detection]:
    pred = model.predict(frame_to_model_rgb(frame), threshold=CONF_THRESHOLD)
    if pred is None or len(pred.xyxy) == 0:
        return []

    has_mask = getattr(pred, "mask", None) is not None
    detections: list[Detection] = []

    for i, (xyxy, conf, class_id) in enumerate(zip(pred.xyxy, pred.confidence, pred.class_id)):
        cls_id = int(class_id)
        class_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
        if class_name != PERSON_CLASS_NAME:
            continue

        box = np.array(xyxy, dtype=np.float32)
        if not is_valid_person_size(box):
            continue

        mask = None
        if has_mask:
            mask = np.asarray(pred.mask[i], dtype=bool)

        detections.append(Detection(box=box, conf=float(conf), class_name=class_name, mask=mask))

    return detections

# =========================
# TRACKING
# =========================


@dataclass
class HistoryItem:
    frame_idx: int
    box: np.ndarray
    conf: float
    foot: np.ndarray
    center: np.ndarray
    foot_d: float
    center_d: float
    foot_side: str
    mask: np.ndarray | None = None


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    conf: float
    last_frame_idx: int
    lost: int = 0
    counted_down: bool = False
    updated: bool = True
    ever_above: bool = False
    ever_on: bool = False
    last_stable_side: str | None = None
    history: deque[HistoryItem] = field(default_factory=lambda: deque(maxlen=TRACK_HISTORY))


class PersonTracker:
    def __init__(self, max_lost_frames: int) -> None:
        self.tracks: list[Track] = []
        self.next_id = 0
        self.max_lost_frames = max(1, int(max_lost_frames))

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 0

    def update(
        self,
        detections: list[Detection],
        frame_idx: int,
        line_a: np.ndarray,
        line_b: np.ndarray,
        bottom_side_sign: int,
    ) -> list[Track]:
        for tr in self.tracks:
            tr.updated = False

        pairs: list[tuple[float, int, int]] = []

        for ti, tr in enumerate(self.tracks):
            if tr.lost > self.max_lost_frames:
                continue

            tr_center = center_xy(tr.box)

            for di, det in enumerate(detections):
                det_center = center_xy(det.box)
                dist = float(np.linalg.norm(det_center - tr_center))
                ov = iou_xyxy(tr.box, det.box)

                if tr.lost == 0:
                    max_dist = TRACK_MAX_CENTER_DISTANCE
                    extra_penalty = 0.0
                elif tr.counted_down:
                    max_dist = REID_MAX_CENTER_DISTANCE_COUNTED
                    extra_penalty = COUNTED_REID_EXTRA_PENALTY
                else:
                    max_dist = REID_MAX_CENTER_DISTANCE
                    extra_penalty = 0.0

                if dist <= max_dist or ov >= TRACK_MIN_IOU_FOR_MATCH:
                    score = ov * 1000.0 - dist - tr.lost * REID_LOST_PENALTY_PER_FRAME - extra_penalty
                    pairs.append((score, ti, di))

        pairs.sort(reverse=True, key=lambda x: x[0])
        used_tracks: set[int] = set()
        used_dets: set[int] = set()

        for _score, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue

            tr = self.tracks[ti]
            det = detections[di]
            self._apply_detection(tr, det, frame_idx, line_a, line_b, bottom_side_sign)
            used_tracks.add(ti)
            used_dets.add(di)

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            tr = self._new_track(det, frame_idx, line_a, line_b, bottom_side_sign)
            self.tracks.append(tr)

        alive: list[Track] = []
        for tr in self.tracks:
            if not tr.updated:
                tr.lost += 1
            if tr.lost <= self.max_lost_frames:
                alive.append(tr)

        self.tracks = alive
        return [tr for tr in self.tracks if tr.updated]

    def _new_track(
        self,
        det: Detection,
        frame_idx: int,
        line_a: np.ndarray,
        line_b: np.ndarray,
        bottom_side_sign: int,
    ) -> Track:
        tr = Track(
            track_id=self.next_id,
            box=det.box.copy(),
            conf=det.conf,
            last_frame_idx=frame_idx,
        )
        self.next_id += 1
        self._append_history(tr, det, frame_idx, line_a, line_b, bottom_side_sign)
        return tr

    def _apply_detection(
        self,
        tr: Track,
        det: Detection,
        frame_idx: int,
        line_a: np.ndarray,
        line_b: np.ndarray,
        bottom_side_sign: int,
    ) -> None:
        tr.box = det.box.copy()
        tr.conf = det.conf
        tr.last_frame_idx = frame_idx
        tr.lost = 0
        tr.updated = True
        self._append_history(tr, det, frame_idx, line_a, line_b, bottom_side_sign)

    def _append_history(
        self,
        tr: Track,
        det: Detection,
        frame_idx: int,
        line_a: np.ndarray,
        line_b: np.ndarray,
        bottom_side_sign: int,
    ) -> None:
        foot = bottom_center_xy(det.box)
        cen = center_xy(det.box)
        foot_d = signed_distance_to_line(foot, line_a, line_b, bottom_side_sign)
        center_d = signed_distance_to_line(cen, line_a, line_b, bottom_side_sign)
        side = stable_side(foot_d, LINE_MARGIN_PX)

        item = HistoryItem(
            frame_idx=frame_idx,
            box=det.box.copy(),
            conf=det.conf,
            foot=foot,
            center=cen,
            foot_d=foot_d,
            center_d=center_d,
            foot_side=side,
            mask=None if det.mask is None else det.mask.copy(),
        )
        tr.history.append(item)

        if side == "above":
            tr.ever_above = True
            tr.last_stable_side = "above"
        elif side == "below":
            if tr.last_stable_side is None:
                tr.last_stable_side = "below"
        else:
            tr.ever_on = True

# =========================
# COUNTER
# =========================


@dataclass
class EventRecord:
    frame_idx: int
    track_id: int
    box: np.ndarray
    center: np.ndarray


class CrossingCounter:
    def __init__(self, run_dir: Path, src: int | str, duplicate_suppress_frames: int) -> None:
        self.run_dir = run_dir
        self.src = src
        self.src_label = source_label(src)
        self.src_stem = source_stem(src)
        self.events_dir = run_dir / self.src_stem
        self.events_dir.mkdir(parents=True, exist_ok=True)

        self.count_down = 0
        self.events: list[EventRecord] = []
        self.duplicate_suppress_frames = max(1, int(duplicate_suppress_frames))
        self.jsonl_path = self.events_dir / "events.jsonl"
        self.jsonl = self.jsonl_path.open("w", encoding="utf-8")

    def reset_after_seek(self) -> None:
        self.events.clear()
        self.count_down = 0
        self.jsonl.write(json.dumps({"type": "seek_reset"}, ensure_ascii=False) + "\n")
        self.jsonl.flush()

    def close(self) -> None:
        self.jsonl.close()

    def should_count_down(
            self,
            tr: Track,
            frame_w: int,
            frame_h: int,
            line_a: np.ndarray,
            line_b: np.ndarray,
    ) -> tuple[bool, str]:
        if tr.counted_down:
            return False, "already_counted_track"

        if len(tr.history) < MIN_TRACK_HISTORY_FOR_EVENT:
            return False, "history_too_short"

        prev = tr.history[-2]
        curr = tr.history[-1]
        # ВАЖНО: работаем только с коротким отрезком LINE_NORM.
        # Без этого считается бесконечная прямая, поэтому люди слева/справа от линии тоже засчитываются.
        if not box_near_segment_gate(curr.box, line_a, line_b):
            return False, "outside_line_segment"
        if touches_frame_border(curr.box, frame_w, frame_h):
            return False, "touches_frame_border"
        # Отсекаем краевые артефакты, особенно справа/снизу.
        if touches_frame_border(curr.box, frame_w, frame_h):
            return False, "touches_frame_border"

        foot_cross = prev.foot_d <= -LINE_MARGIN_PX and curr.foot_d >= LINE_MARGIN_PX
        from_above_to_below = prev.foot_side in {"above", "on"} and curr.foot_side == "below"

        if tr.ever_above and (foot_cross or from_above_to_below):
            if self._recent_travel(tr) >= MIN_RECENT_TRAVEL_PX:
                return True, "foot_crossed_from_above"
            return False, "too_static"

        bottom_delta = curr.foot_d - prev.foot_d
        center_delta = curr.center_d - prev.center_d

        # Обычный late_start: человек впервые появился уже рядом с линией и сразу пошёл вниз.
        late_start_allowed = (
                not tr.ever_above
                and len(tr.history) <= 4
                and curr.conf >= LATE_START_MIN_CONF
                and curr.foot_side == "below"
                and bottom_delta >= LATE_START_MIN_BOTTOM_DELTA
                and center_delta >= LATE_START_MIN_CENTER_DELTA
        )
        if late_start_allowed:
            return True, "late_start_moving_down"

        # ВАЖНО: случай перекрытия.
        # Человек мог быть закрыт другим человеком/объектом, потом появился уже ниже линии.
        # Если первые видимые точки были около линии, а потом bbox явно ушёл вниз — считаем.
        hist = list(tr.history)
        if (
                not tr.ever_above
                and 3 <= len(hist) <= OCCLUSION_MAX_HISTORY
                and curr.conf >= OCCLUSION_MIN_CONF
                and curr.foot_side == "below"
        ):
            first = hist[0]

            started_near_line = first.foot_d <= OCCLUSION_START_MAX_BELOW_D
            moved_down_enough = (curr.foot_d - first.foot_d) >= OCCLUSION_MIN_BOTTOM_DELTA
            center_moved_down_enough = (curr.center_d - first.center_d) >= OCCLUSION_MIN_CENTER_DELTA

            if started_near_line and moved_down_enough and center_moved_down_enough:
                return True, "occlusion_reappear_moved_down"

        return False, "no_down_cross"

    def _recent_travel(self, tr: Track, n: int = 8) -> float:
        hist = list(tr.history)[-n:]
        if len(hist) < 2:
            return 0.0
        pts = np.array([h.foot for h in hist], dtype=np.float32)
        span = pts.max(axis=0) - pts.min(axis=0)
        return float(np.linalg.norm(span))

    def is_duplicate(self, frame_idx: int, box: np.ndarray) -> bool:
        cen = center_xy(box)

        for ev in reversed(self.events):
            frame_gap = frame_idx - ev.frame_idx
            if frame_gap > DUPLICATE_MAX_FRAME_GAP:
                break

            ov = iou_xyxy(box, ev.box)
            dist = float(np.linalg.norm(cen - ev.center))

            # Только почти тот же bbox в течение нескольких кадров.
            # Не подавляем людей, которые идут рядом через 10–40 кадров.
            if ov >= DUPLICATE_IOU_THRESHOLD and dist <= DUPLICATE_CENTER_DISTANCE:
                return True

        return False

    def register_down(self, frame: np.ndarray, tr: Track, reason: str) -> np.ndarray:
        curr = tr.history[-1]
        event_box = curr.box

        if self.is_duplicate(curr.frame_idx, event_box):
            tr.counted_down = True
            print(
                f"[DUPLICATE SUPPRESSED] video={self.src_label} "
                f"frame={curr.frame_idx} track_id={tr.track_id} reason={reason}",
                flush=True,
            )
            return frame

        tr.counted_down = True
        self.count_down += 1
        self.events.append(
            EventRecord(
                frame_idx=curr.frame_idx,
                track_id=tr.track_id,
                box=event_box.copy(),
                center=center_xy(event_box),
            )
        )

        w, h, area = box_wh_area(event_box)
        rec = {
            "type": "event",
            "video": self.src_label,
            "frame": curr.frame_idx,
            "direction": "DOWN",
            "track_id": tr.track_id,
            "reason": reason,
            "conf": round(float(curr.conf), 3),
            "box": [round(float(x), 1) for x in event_box.tolist()],
            "object_w": int(w),
            "object_h": int(h),
            "object_area": int(area),
            "count_down_total": self.count_down,
        }

        print("[EVENT DOWN] " + json.dumps(rec, ensure_ascii=False), flush=True)
        self.jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.jsonl.flush()

        out = frame.copy()
        draw_person(out, event_box, f"DOWN #{self.count_down} id={tr.track_id}", color=(0, 0, 255), thickness=3)
        cv2.putText(
            out,
            f"DOWN #{self.count_down} | {reason}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if SAVE_EVENTS:
            filename = f"{self.src_stem}_DOWN_{self.count_down:04d}_frame_{curr.frame_idx:06d}.jpg"
            path = self.events_dir / filename
            cv2.imwrite(str(path), out)
            print(f"  saved: {path}", flush=True)

        return out

# =========================
# DRAW
# =========================


def draw_person(
    frame: np.ndarray,
    box: np.ndarray,
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    line_a: np.ndarray,
    line_b: np.ndarray,
    frame_idx: int,
    count_down: int,
    paused: bool = False,
    source_name: str = "",
) -> np.ndarray:
    out = frame.copy()

    cv2.line(out, tuple(line_a.astype(int)), tuple(line_b.astype(int)), (255, 255, 255), 2)

    active = 0
    for tr in tracks:
        if tr.lost > 0 or not tr.history:
            continue

        active += 1
        curr = tr.history[-1]
        label = f"id={tr.track_id} person {curr.conf:.2f} {curr.foot_side} d={curr.foot_d:.0f}"
        if tr.counted_down:
            label += " COUNTED"
        draw_person(out, curr.box, label, color=(0, 255, 0), thickness=2)

        foot = tuple(curr.foot.astype(int))
        cv2.circle(out, foot, 5, (0, 255, 255), -1)

        hist = list(tr.history)[-20:]
        for a, b in zip(hist, hist[1:]):
            cv2.line(out, tuple(a.foot.astype(int)), tuple(b.foot.astype(int)), (0, 255, 255), 2)

    top_text = f"{source_name} | frame={frame_idx} | DOWN={count_down} | active_tracks={active}"
    if paused:
        top_text += " | PAUSED"

    cv2.putText(
        out,
        top_text,
        (12, out.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out

# =========================
# MODEL
# =========================


def build_model(class_count: int) -> RFDETRSegMedium:
    model = RFDETRSegMedium(pretrain_weights=str(SEG_CHECKPOINT_PATH), num_classes=class_count)
    try:
        model.optimize_for_inference()
    except Exception:
        pass
    return model

# =========================
# WORKER THREAD
# =========================


class ProcessingWorker(QThread):
    frame_ready = pyqtSignal(object)
    log = pyqtSignal(str)
    video_opened = pyqtSignal(int, str)
    frame_progress = pyqtSignal(int)
    source_changed = pyqtSignal(str, int, int)
    finished_ok = pyqtSignal(int, str)

    def __init__(self, model: RFDETRSegMedium, class_names: list[str], live_source, run_dir: Path):
        super().__init__()
        self.model = model
        self.class_names = class_names
        self.live_source = live_source
        self.run_dir = run_dir

        self._stop = False
        self._pause = False
        self._seek_lock = threading.Lock()
        self._seek_frame_1based: int | None = None
        self._step_frames = 0

    def request_stop(self) -> None:
        self._stop = True

    def request_pause_toggle(self) -> None:
        self._pause = not self._pause

    def set_paused(self, value: bool) -> None:
        self._pause = bool(value)

    def request_seek_frame(self, frame_1based: int) -> None:
        with self._seek_lock:
            self._seek_frame_1based = max(1, int(frame_1based))

    def request_step(self, frames: int) -> None:
        with self._seek_lock:
            self._step_frames += int(frames)
            self._pause = True

    def _consume_seek(self) -> int | None:
        with self._seek_lock:
            val = self._seek_frame_1based
            self._seek_frame_1based = None
            return val

    def _consume_step(self) -> int:
        with self._seek_lock:
            val = self._step_frames
            self._step_frames = 0
            return val

    def run(self) -> None:
        sources = expand_sources(self.live_source)
        if not sources:
            self.log.emit(f"Источники не найдены: {self.live_source!r}")
            self.finished_ok.emit(0, str(self.run_dir))
            return

        total_down = 0

        for src_index, src in enumerate(sources, start=1):
            if self._stop:
                break

            down = self._process_one_source(src, src_index, len(sources))
            total_down += down

        self.finished_ok.emit(total_down, str(self.run_dir))

    def _process_one_source(self, src: int | str, src_index: int, src_total: int) -> int:
        cap = open_capture(src)
        if cap is None:
            self.log.emit(f"Не удалось открыть источник: {src!r}")
            return 0

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        seek_enabled = is_video_file(src) and total_frames > 1
        wait_ms = playback_wait_ms(cap, src)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps < 1.0 or fps > 120.0:
            fps = 25.0

        keep_alive_frames = max(1, int(round(fps * TRACK_KEEP_ALIVE_SECONDS)))
        duplicate_suppress_frames = max(1, int(round(fps * DUPLICATE_SUPPRESS_SECONDS)))

        self.video_opened.emit(total_frames if seek_enabled else 0, source_label(src))
        self.source_changed.emit(source_label(src), src_index, src_total)
        self.log.emit(f"\n[{src_index}/{src_total}] {source_label(src)}")

        tracker = PersonTracker(max_lost_frames=keep_alive_frames)
        counter = CrossingCounter(self.run_dir, src, duplicate_suppress_frames=duplicate_suppress_frames)
        writer: cv2.VideoWriter | None = None

        line_a: np.ndarray | None = None
        line_b: np.ndarray | None = None
        bottom_side_sign = 1
        last_dets: list[Detection] = []
        frame_idx = 0
        last_vis: np.ndarray | None = None

        self.log.emit(
            f"Жизнь уникального человека: {TRACK_KEEP_ALIVE_SECONDS:.1f} сек ≈ {keep_alive_frames} кадров | "
            f"duplicate-window ≈ {duplicate_suppress_frames} кадров"
        )
        self.log.emit(f"Источник: {source_label(src)}")
        self.log.emit(f"Артефакты: {counter.events_dir}")
        if seek_enabled:
            self.log.emit(f"Перемотка доступна: 1…{total_frames} кадров")
        else:
            self.log.emit("Перемотка недоступна для камеры/RTSP или неизвестной длины")

        try:
            while not self._stop:
                if self._pause:
                    pending_seek = self._consume_seek()
                    pending_step = self._consume_step()

                    if seek_enabled and pending_step != 0:
                        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or frame_idx)
                        target = max(1, min(total_frames, current + pending_step))
                        pending_seek = target

                    if seek_enabled and pending_seek is not None:
                        frame_idx = self._seek_to_frame(
                            cap=cap,
                            target_frame_1based=pending_seek,
                            total_frames=total_frames,
                            tracker=tracker,
                            counter=counter,
                        )
                        last_dets = []
                        last_vis = None
                    else:
                        self.msleep(25)
                        continue

                else:
                    pending_seek = self._consume_seek()
                    if seek_enabled and pending_seek is not None:
                        frame_idx = self._seek_to_frame(
                            cap=cap,
                            target_frame_1based=pending_seek,
                            total_frames=total_frames,
                            tracker=tracker,
                            counter=counter,
                        )
                        last_dets = []
                        last_vis = None

                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                if seek_enabled:
                    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                    frame_idx = max(1, pos)
                else:
                    frame_idx += 1

                fh, fw = frame.shape[:2]

                if line_a is None or line_b is None:
                    line_a, line_b = denorm_line(LINE_NORM, fw, fh)
                    bottom_side_sign = infer_bottom_side_sign(fw, fh, line_a, line_b)

                if DETECT_EVERY_N <= 1 or frame_idx % DETECT_EVERY_N == 1 or not last_dets:
                    last_dets = load_person_detections(self.model, frame, self.class_names)

                updated_tracks = tracker.update(last_dets, frame_idx, line_a, line_b, bottom_side_sign)
                vis = draw_tracks(
                    frame,
                    tracker.tracks,
                    line_a,
                    line_b,
                    frame_idx,
                    counter.count_down,
                    paused=self._pause,
                    source_name=source_label(src),
                )

                for tr in updated_tracks:
                    should_count, reason = counter.should_count_down(tr, fw, fh, line_a, line_b)
                    if should_count:
                        vis = counter.register_down(vis, tr, reason)

                last_vis = vis
                self.frame_ready.emit(vis)

                if seek_enabled:
                    self.frame_progress.emit(frame_idx)

                if frame_idx % 25 == 0:
                    active_tracks = [t for t in tracker.tracks if t.lost == 0]
                    self.log.emit(
                        f"[frame {frame_idx}] dets={len(last_dets)} "
                        f"active_tracks={len(active_tracks)} DOWN={counter.count_down}"
                    )

                if SAVE_DEBUG_VIDEO:
                    if writer is None:
                        out_path = counter.events_dir / OUTPUT_VIDEO_NAME
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))
                        if not writer.isOpened():
                            self.log.emit(f"Не удалось открыть VideoWriter: {out_path}")
                            writer = None
                    if writer is not None:
                        writer.write(vis)

                if self._pause:
                    continue

                self.msleep(wait_ms)

            if last_vis is not None and self._pause:
                self.frame_ready.emit(last_vis)

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            counter.close()

        self.log.emit(f"Итог {source_label(src)}: DOWN={counter.count_down}")
        return counter.count_down

    def _seek_to_frame(
        self,
        cap: cv2.VideoCapture,
        target_frame_1based: int,
        total_frames: int,
        tracker: PersonTracker,
        counter: CrossingCounter,
    ) -> int:
        target = max(1, int(target_frame_1based))
        if total_frames > 0:
            target = min(target, total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, target - 1)
        tracker.reset()
        counter.reset_after_seek()
        self.frame_progress.emit(target)
        self.log.emit(f"Перемотка → кадр {target}" + (f" / {total_frames}" if total_frames > 0 else ""))
        return target

# =========================
# PYQT MAIN WINDOW
# =========================


class MainWindow(QMainWindow):
    def __init__(self, worker: ProcessingWorker):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.worker = worker
        self.closed = False
        self._pix: QPixmap | None = None
        self._video_total = 0
        self._pause_seek_sync = False
        self._current_frame = 1
        self._paused = False

        central = QWidget()
        main_layout = QVBoxLayout(central)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setStyleSheet("background-color: #1a1a1a;")
        main_layout.addWidget(self.video_label, 1)

        seek_row = QHBoxLayout()
        self.lbl_seek = QLabel("Перемотка: ожидание источника…")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 1)
        self.slider.setEnabled(False)

        self.spin = QSpinBox()
        self.spin.setRange(1, 1)
        self.spin.setEnabled(False)

        self.btn_minus_100 = QPushButton("-100")
        self.btn_minus_10 = QPushButton("-10")
        self.btn_minus_1 = QPushButton("-1")
        self.btn_plus_1 = QPushButton("+1")
        self.btn_plus_10 = QPushButton("+10")
        self.btn_plus_100 = QPushButton("+100")
        self.btn_mid = QPushButton("Середина")
        self.btn_pause = QPushButton("Пауза")

        for btn in [
            self.btn_minus_100,
            self.btn_minus_10,
            self.btn_minus_1,
            self.btn_plus_1,
            self.btn_plus_10,
            self.btn_plus_100,
            self.btn_mid,
        ]:
            btn.setEnabled(False)

        seek_row.addWidget(self.lbl_seek)
        seek_row.addWidget(self.slider, 1)
        seek_row.addWidget(self.spin)
        seek_row.addWidget(self.btn_minus_100)
        seek_row.addWidget(self.btn_minus_10)
        seek_row.addWidget(self.btn_minus_1)
        seek_row.addWidget(self.btn_plus_1)
        seek_row.addWidget(self.btn_plus_10)
        seek_row.addWidget(self.btn_plus_100)
        seek_row.addWidget(self.btn_mid)
        seek_row.addWidget(self.btn_pause)
        main_layout.addLayout(seek_row)

        self.setCentralWidget(central)

        worker.frame_ready.connect(self.on_frame)
        worker.log.connect(self.on_log)
        worker.video_opened.connect(self.on_video_opened)
        worker.frame_progress.connect(self.on_frame_progress)
        worker.source_changed.connect(self.on_source_changed)
        worker.finished_ok.connect(self.on_finished_ok)

        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.spin.editingFinished.connect(self.on_spin_seek)
        self.spin.lineEdit().installEventFilter(self)

        self.btn_mid.clicked.connect(self.on_seek_middle)
        self.btn_pause.clicked.connect(self.on_pause_toggle)
        self.btn_minus_100.clicked.connect(lambda: self.on_step(-100))
        self.btn_minus_10.clicked.connect(lambda: self.on_step(-10))
        self.btn_minus_1.clicked.connect(lambda: self.on_step(-1))
        self.btn_plus_1.clicked.connect(lambda: self.on_step(1))
        self.btn_plus_10.clicked.connect(lambda: self.on_step(10))
        self.btn_plus_100.clicked.connect(lambda: self.on_step(100))

        QShortcut(QKeySequence("Q"), self, activated=self.close)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.close)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.on_pause_toggle)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self.on_step(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self.on_step(1))
        QShortcut(QKeySequence("Ctrl+Left"), self, activated=lambda: self.on_step(-10))
        QShortcut(QKeySequence("Ctrl+Right"), self, activated=lambda: self.on_step(10))
        QShortcut(QKeySequence("M"), self, activated=self.on_seek_middle)

    def on_source_changed(self, name: str, index: int, total: int) -> None:
        self.setWindowTitle(f"{WINDOW_TITLE} | {index}/{total}: {name}")

    def on_video_opened(self, total_frames: int, name: str) -> None:
        self._video_total = int(total_frames)
        self._current_frame = 1

        if self._video_total <= 1:
            self.lbl_seek.setText(f"{name}: перемотка недоступна")
            self.slider.setEnabled(False)
            self.spin.setEnabled(False)
            for btn in [
                self.btn_minus_100,
                self.btn_minus_10,
                self.btn_minus_1,
                self.btn_plus_1,
                self.btn_plus_10,
                self.btn_plus_100,
                self.btn_mid,
            ]:
                btn.setEnabled(False)
            return

        self.lbl_seek.setText(f"{name}: кадр 1…{self._video_total}")
        self.slider.setRange(1, self._video_total)
        self.spin.setRange(1, self._video_total)
        self.slider.setValue(1)
        self.spin.setValue(1)
        self.slider.setEnabled(True)
        self.spin.setEnabled(True)
        for btn in [
            self.btn_minus_100,
            self.btn_minus_10,
            self.btn_minus_1,
            self.btn_plus_1,
            self.btn_plus_10,
            self.btn_plus_100,
            self.btn_mid,
        ]:
            btn.setEnabled(True)

    def on_frame_progress(self, frame_idx: int) -> None:
        self._current_frame = max(1, int(frame_idx))
        if self._video_total <= 1 or self._pause_seek_sync:
            return

        v = max(1, min(self._current_frame, self._video_total))
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(v)
        self.spin.setValue(v)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

    def on_slider_pressed(self) -> None:
        self._pause_seek_sync = True

    def on_slider_released(self) -> None:
        if not self.slider.isEnabled():
            self._pause_seek_sync = False
            return
        v = self.slider.value()
        self.spin.setValue(v)
        self.worker.request_seek_frame(v)
        self._pause_seek_sync = False

    def on_spin_seek(self) -> None:
        if not self.spin.isEnabled() or self._video_total <= 1:
            return
        v = max(1, min(int(self.spin.value()), self._video_total))
        self.spin.setValue(v)
        self.slider.setValue(v)
        self.worker.request_seek_frame(v)
        self.spin.clearFocus()

    def on_seek_middle(self) -> None:
        if self._video_total <= 1:
            return
        mid = max(1, self._video_total // 2)
        self.slider.setValue(mid)
        self.spin.setValue(mid)
        self.worker.request_seek_frame(mid)

    def on_step(self, delta: int) -> None:
        if self._video_total <= 1:
            return
        target = max(1, min(self._video_total, self._current_frame + int(delta)))
        self.slider.setValue(target)
        self.spin.setValue(target)
        self.worker.request_seek_frame(target)

    def on_pause_toggle(self) -> None:
        self._paused = not self._paused
        self.btn_pause.setText("Продолжить" if self._paused else "Пауза")
        self.worker.request_pause_toggle()

    def eventFilter(self, obj, event) -> bool:  # noqa: ANN001
        if obj == self.spin.lineEdit():
            if event.type() == QEvent.FocusIn:
                self._pause_seek_sync = True
            elif event.type() == QEvent.FocusOut:
                self._pause_seek_sync = False
        return super().eventFilter(obj, event)

    def on_frame(self, frame: np.ndarray) -> None:
        self._pix = QPixmap.fromImage(bgr_to_qimage(frame))
        self.apply_scale()

    def apply_scale(self) -> None:
        if self._pix is None or self._pix.isNull():
            return
        self.video_label.setPixmap(
            self._pix.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def on_log(self, msg: str) -> None:
        print(msg, flush=True)

    def on_finished_ok(self, total_down: int, run_dir: str) -> None:
        print(f"\nИТОГО DOWN={total_down} | логи: {run_dir}", flush=True)
        self.btn_pause.setEnabled(False)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self.apply_scale()

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self.closed = True
        self.worker.request_stop()
        self.worker.wait(15000)
        event.accept()

# =========================
# HEADLESS PROCESSING
# =========================


def process_source_headless(
    model: RFDETRSegMedium,
    class_names: list[str],
    src: int | str,
    run_dir: Path,
) -> int:
    cap = open_capture(src)
    if cap is None:
        print(f"Не удалось открыть источник: {src!r}", flush=True)
        return 0

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1.0 or fps > 120.0:
        fps = 25.0

    keep_alive_frames = max(1, int(round(fps * TRACK_KEEP_ALIVE_SECONDS)))
    duplicate_suppress_frames = max(1, int(round(fps * DUPLICATE_SUPPRESS_SECONDS)))

    tracker = PersonTracker(max_lost_frames=keep_alive_frames)
    counter = CrossingCounter(run_dir, src, duplicate_suppress_frames=duplicate_suppress_frames)
    writer: cv2.VideoWriter | None = None

    line_a: np.ndarray | None = None
    line_b: np.ndarray | None = None
    bottom_side_sign = 1
    last_dets: list[Detection] = []
    frame_idx = 0

    print(f"Источник: {source_label(src)}", flush=True)
    print(f"Артефакты: {counter.events_dir}", flush=True)
    print(
        f"Жизнь уникального человека: {TRACK_KEEP_ALIVE_SECONDS:.1f} сек ≈ {keep_alive_frames} кадров | "
        f"duplicate-window ≈ {duplicate_suppress_frames} кадров",
        flush=True,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx += 1
            fh, fw = frame.shape[:2]

            if line_a is None or line_b is None:
                line_a, line_b = denorm_line(LINE_NORM, fw, fh)
                bottom_side_sign = infer_bottom_side_sign(fw, fh, line_a, line_b)

            if DETECT_EVERY_N <= 1 or frame_idx % DETECT_EVERY_N == 1 or not last_dets:
                last_dets = load_person_detections(model, frame, class_names)

            updated_tracks = tracker.update(last_dets, frame_idx, line_a, line_b, bottom_side_sign)
            vis = draw_tracks(
                frame,
                tracker.tracks,
                line_a,
                line_b,
                frame_idx,
                counter.count_down,
                paused=False,
                source_name=source_label(src),
            )

            for tr in updated_tracks:
                should_count, reason = counter.should_count_down(tr, fw, fh, line_a, line_b)
                if should_count:
                    vis = counter.register_down(vis, tr, reason)

            if frame_idx % 25 == 0:
                active_tracks = [t for t in tracker.tracks if t.lost == 0]
                print(
                    f"[frame {frame_idx}] dets={len(last_dets)} "
                    f"active_tracks={len(active_tracks)} DOWN={counter.count_down}",
                    flush=True,
                )

            if SAVE_DEBUG_VIDEO:
                if writer is None:
                    out_path = counter.events_dir / OUTPUT_VIDEO_NAME
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))
                    if not writer.isOpened():
                        print(f"Не удалось открыть VideoWriter: {out_path}", flush=True)
                        writer = None
                if writer is not None:
                    writer.write(vis)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        counter.close()

    print(f"Итог {source_label(src)}: DOWN={counter.count_down}", flush=True)
    return counter.count_down

# =========================
# MAIN
# =========================


def main() -> int:
    if not DATA_YAML_PATH.exists():
        print(f"data.yaml не найден: {DATA_YAML_PATH}", flush=True)
        return 1

    if not SEG_CHECKPOINT_PATH.exists():
        print(f"checkpoint не найден: {SEG_CHECKPOINT_PATH}", flush=True)
        return 1

    class_names = load_class_names_from_yaml(DATA_YAML_PATH)
    if PERSON_CLASS_NAME not in class_names:
        print(f"В data.yaml нет класса {PERSON_CLASS_NAME!r}. Классы: {class_names}", flush=True)
        return 1

    print("Классы:", ", ".join(f"{i}:{name}" for i, name in enumerate(class_names)), flush=True)
    print("Загрузка RF-DETR Seg...", flush=True)
    model = build_model(len(class_names))

    run_dir = make_run_dir()
    print(f"Папка прогона: {run_dir}", flush=True)

    sources = expand_sources(LIVE_SOURCE)
    if not sources:
        print(f"Источники не найдены: {LIVE_SOURCE!r}", flush=True)
        return 1

    if USE_PYQT_VIEWER:
        app = QApplication(sys.argv)
        worker = ProcessingWorker(model=model, class_names=class_names, live_source=LIVE_SOURCE, run_dir=run_dir)
        win = MainWindow(worker)
        win.show()
        worker.start()
        return int(app.exec_())

    total = 0
    for i, src in enumerate(sources, start=1):
        print(f"\n[{i}/{len(sources)}] {source_label(src)}", flush=True)
        total += process_source_headless(model, class_names, src, run_dir)

    print(f"\nИТОГО DOWN={total} | логи: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
