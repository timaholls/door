#!/usr/bin/env python3
"""
Эфир с RF-DETR Seg (instance segmentation): маски + контуры на кадре, логика событий как в rfdetr_live.

Источник (LIVE_SOURCE): int — камера; str — один RTSP/файл; list — несколько URL (как в camera.py).
Статические зоны игнора — только если источник привязан к папке cam0 (или URL с сегментом /cam0/).

Зависимость: pip install PyQt5 supervision

Без окна: RFDETR_LIVE_HEADLESS=1 python rfdetr_live_seg.py

Чекпоинт: SEG_CHECKPOINT_PATH (веса сегментации, не bbox-only).

DOWN-логика (занос):
  Дверь должна пересечь ВЕРХНЮЮ линию (STATIC_TRIPWIRE_LINE_NORM) сверху вниз,
  а затем пересечь НИЖНЮЮ линию (STATIC_TRIPWIRE_DOWN_LINE_NORM) сверху вниз.
  Только после пересечения обеих линий засчитывается событие DOWN.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRSegMedium

# неполный cv2 (пустой namespace в site-packages/cv2) ломает albumentations → rfdetr
if not hasattr(cv2, "CV_8U"):
    print(
        "Ошибка: OpenCV установлен неполно (нет cv2.CV_8U). Переустанови:\n"
        "  pip uninstall opencv-python opencv-python-headless -y\n"
        "  pip install opencv-python"
    )
    sys.exit(1)

import rfdetr_video_events as rv

# Верхняя линия — для UP (вынос) и первый порог DOWN (занос)
STATIC_TRIPWIRE_LINE_NORM: list[tuple[float, float]] = [
    (0.172653, 0.561082),
    (0.475232, 0.424178),
]

# Нижняя линия — второй порог DOWN (занос). Дверь должна пересечь её ПОСЛЕ верхней.
STATIC_TRIPWIRE_DOWN_LINE_NORM: list[tuple[float, float]] = [
    (0.170274, 0.838710),
    (0.653457, 0.548387),
]


try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QImage, QKeySequence, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QShortcut,
        QWidget,
    )

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

# ---------- источник (выбери один) ----------
LIVE_SOURCE_URLS: list[str] = [
    # "rtsp://viewer:ViewerPass_9347X@94.41.120.115:8554/cam44",
    # "rtsp://viewer:ViewerPass_9347X@94.41.120.115:8554/cam47",
    # "rtsp://viewer:ViewerPass_9347X@94.41.120.115:8554/cam49",
]
RECORDINGS_CAM0_DIR = "/home/art/PycharmProjects/PythonProject12/recordings_2/cam0"
_rtsp_one = os.environ.get("RFDETR_LIVE_RTSP", "").strip()
if _rtsp_one:
    LIVE_SOURCE: int | str | list = _rtsp_one
elif LIVE_SOURCE_URLS:
    LIVE_SOURCE = LIVE_SOURCE_URLS
else:
    LIVE_SOURCE = RECORDINGS_CAM0_DIR
# LIVE_SOURCE = 0

WINDOW_TITLE = "RF-DETR Seg — эфир (PyQt) | Q / Esc — выход"

# Веса сегментации (train_seg.py → output_seg/…)
SEG_CHECKPOINT_PATH = str(
    Path(__file__).resolve().parent / "new_dataset" / "checkpoint_best_total.pth"
)
# Прозрачность заливки маски на превью (0–1)
LIVE_MASK_OPACITY = 0.45
LIVE_POLYGON_THICKNESS = 2

LOOP_VIDEO_FILE = False
DETECT_EVERY_N = 1
SHOW_TRIPWIRE = True

FILE_PLAYBACK_SLOWDOWN = 2.0
FILE_PLAYBACK_CAP_FPS = 18.0

ALSO_SAVE_MP4 = False
MP4_OUTPUT = Path(__file__).resolve().parent / "recordings_2/cam0"

# ---------- логи и сохранение кадров ----------
LIVE_RUN_ROOT = Path(__file__).resolve().parent / "rfdetr_live_logs_0"
LOG_EVERY_N_FRAMES = 10
SAVE_ANNOTATED_JPEG_EVERY_N = 0

# ---------- debug / relaxed live event logic ----------
LIVE_DEBUG_EVENT_DECISIONS = False
LIVE_MIN_TRACK_HISTORY = 2
LIVE_LINE_MARGIN_PX = 18.0
LIVE_RELAXED_ROI_EXPAND_X = 0.45
LIVE_RELAXED_ROI_EXPAND_Y = 0.28
LIVE_RELAXED_MAX_PERSON_CENTER_DIST_FACTOR = 1.55

LIVE_MIN_UP_TRAVEL_PX = 24.0
LIVE_MIN_DOWN_TRAVEL_PX = 14.0   # для DOWN немного мягче — дверь при заносе двигается компактнее
LIVE_MOVE_WINDOW_FRAMES = 8

LIVE_REQUIRED_BELOW_FRAMES = 2
LIVE_REQUIRED_ABOVE_FRAMES = 1   # для DOWN мягче: на улице трек нестабилен

# ---------- стабилизация класса события ----------
EVENT_CLASS_WINDOW = 7

# ---------- фильтр минимального размера ----------
MIN_DOOR_WIDTH_PX = 60
MIN_DOOR_HEIGHT_PX = 80
MIN_DOOR_AREA_PX2 = 6500

MIN_TRIM_WIDTH_PX = 20
MIN_TRIM_HEIGHT_PX = 80
MIN_TRIM_AREA_PX2 = 2500

FILTER_SMALL_OBJECTS_BEFORE_TRACKING = False

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def _is_video_file(src) -> bool:
    return isinstance(src, str) and Path(src).expanduser().is_file()


def _is_video_dir(src) -> bool:
    return isinstance(src, str) and Path(src).expanduser().is_dir()


def _source_label(src) -> str:
    if isinstance(src, int):
        return f"camera_{src}"
    p = Path(str(src)).expanduser()
    return p.name or str(src)


def _list_video_files(src_dir: str) -> list[str]:
    base = Path(src_dir).expanduser()
    return sorted(
        str(p)
        for p in base.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _expand_sources(src) -> list:
    if isinstance(src, (list, tuple)):
        return [x for x in src]
    if isinstance(src, int):
        return [src]
    if _is_video_dir(src):
        return _list_video_files(src)
    return [src]


def _try_open_video_capture(src) -> cv2.VideoCapture | None:
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
    cap = cv2.VideoCapture(src)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def _format_capture_open_error(src) -> str:
    lines = [
        f"Не удалось открыть источник {src!r}.",
    ]
    if isinstance(src, int):
        lines.extend([
            "  Локальная камера по индексу:",
            "    • Есть ли устройство: ls -l /dev/video*",
            "    • Доступ: пользователь в группе «video» (groups; иначе sudo usermod -aG video $USER и перелогин).",
            "    • Другой индекс: в коде LIVE_SOURCE = 1 или переменная RFDETR_LIVE_RTSP для IP-камеры.",
            "    • Виртуалка/WSL: USB-камера часто недоступна — используйте RTSP или видеофайл.",
        ])
    else:
        lines.extend([
            "  URL или путь:",
            "    • RTSP: проверьте сеть, учётные данные, поток (vlc/ffplay на этом URL).",
            "    • Файл: существует ли путь и права на чтение.",
        ])
    lines.append(
        "  Сообщение OpenCV про libavdevice при индексе камеры часто можно игнорировать; "
        "для потока с камеры по сети задайте RFDETR_LIVE_RTSP."
    )
    return "\n".join(lines)


def _playback_wait_ms(cap, src) -> int:
    if isinstance(src, int):
        return 1
    if _is_video_file(src):
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps < 1.0 or fps > 120.0:
            fps = 25.0
        fps = min(fps, FILE_PLAYBACK_CAP_FPS)
        ms = max(1.0, (1000.0 / fps) * FILE_PLAYBACK_SLOWDOWN)
        return max(1, round(ms))
    return 1


def _probe_fps(src) -> float:
    if isinstance(src, (list, tuple)) and len(src) > 0:
        return _probe_fps(src[0])
    if _is_video_dir(src):
        videos = _list_video_files(src)
        if not videos:
            return 25.0
        src = videos[0]
    cap = _try_open_video_capture(src)
    if cap is None:
        return 25.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps < 1.0 or fps > 120.0:
        return 25.0
    return fps


def _build_seg_model(checkpoint_path: str, num_classes: int) -> RFDETRSegMedium:
    model = RFDETRSegMedium(pretrain_weights=checkpoint_path, num_classes=num_classes)
    try:
        model.optimize_for_inference()
    except Exception:
        pass
    return model


def _load_frame_detections_seg(model, frame, class_names: list[str]) -> list:
    detections = model.predict(rv.frame_to_model_rgb(frame), threshold=rv.CONF_THRESHOLD)
    result = []
    if detections is None or len(detections.xyxy) == 0:
        return result
    has_mask = detections.mask is not None
    for i, (xyxy, confidence, class_id) in enumerate(
        zip(detections.xyxy, detections.confidence, detections.class_id)
    ):
        class_id = int(class_id)
        class_name = (
            class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        )
        item = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": float(confidence),
            "box": np.array(xyxy, dtype=np.float32),
        }
        if has_mask:
            item["mask"] = np.asarray(detections.mask[i], dtype=bool)
        result.append(item)
    return result


def _color_bgr(class_name: str):
    if class_name == rv.DOOR_CLASS_NAME:
        return (255, 255, 0)
    if class_name == rv.TRIM_CLASS_NAME:
        return (255, 0, 255)
    if class_name == rv.PERSON_CLASS_NAME:
        return (0, 255, 0)
    return (180, 180, 180)


def _annotate_frame(
    frame: np.ndarray,
    last_dets: list,
    frame_idx: int,
    line_px=None,
    line_down_px=None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    vis = frame.copy()
    if len(last_dets) > 0 and all("mask" in d for d in last_dets):
        xyxy = np.array([d["box"] for d in last_dets], dtype=np.float32)
        conf = np.array([d["confidence"] for d in last_dets], dtype=np.float32)
        cid = np.array([d["class_id"] for d in last_dets], dtype=int)
        masks = np.stack([d["mask"] for d in last_dets])
        dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cid, mask=masks)
        vis = sv.MaskAnnotator(
            opacity=LIVE_MASK_OPACITY,
            color_lookup=sv.ColorLookup.CLASS,
        ).annotate(scene=vis, detections=dets)
        vis = sv.PolygonAnnotator(
            thickness=LIVE_POLYGON_THICKNESS,
            color_lookup=sv.ColorLookup.CLASS,
        ).annotate(scene=vis, detections=dets)
        labels = [f"{d['class_name']} {d['confidence']:.2f}" for d in last_dets]
        vis = sv.LabelAnnotator(text_scale=0.5, text_padding=4).annotate(
            scene=vis,
            detections=dets,
            labels=labels,
        )
    else:
        for d in last_dets:
            label = f"{d['class_name']} {d['confidence']:.2f}"
            rv.draw_box_with_label(vis, d["box"], label, _color_bgr(d["class_name"]), thickness=2)

    if SHOW_TRIPWIRE:
        # Верхняя линия (UP / первый порог DOWN) — белая
        if line_px is not None:
            cv2.line(
                vis,
                tuple(line_px[0].astype(int)),
                tuple(line_px[1].astype(int)),
                (255, 255, 255),
                2,
            )
        # Нижняя линия (второй порог DOWN) — жёлтая
        if line_down_px is not None:
            cv2.line(
                vis,
                tuple(line_down_px[0].astype(int)),
                tuple(line_down_px[1].astype(int)),
                (0, 255, 255),
                2,
            )

    cv2.putText(
        vis,
        f"frame {frame_idx} | dets {len(last_dets)}",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def _new_run_dir() -> Path:
    LIVE_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = LIVE_RUN_ROOT / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _det_stats(last_dets: list) -> dict:
    counts = dict(Counter(d["class_name"] for d in last_dets))
    objects = []
    for d in last_dets:
        b = d["box"]
        objects.append({
            "class": d["class_name"],
            "conf": round(float(d["confidence"]), 4),
            "xyxy": [round(float(x), 1) for x in b],
        })
    return {"counts": counts, "objects": objects}


def _write_run_info(path: Path, src, extra: dict) -> None:
    lines = [
        f"started_utc={datetime.utcnow().isoformat()}Z",
        f"source={src!r}",
        f"LOG_EVERY_N_FRAMES={LOG_EVERY_N_FRAMES}",
        f"SAVE_ANNOTATED_JPEG_EVERY_N={SAVE_ANNOTATED_JPEG_EVERY_N}",
        f"DETECT_EVERY_N={DETECT_EVERY_N}",
        f"EVENT_CLASS_WINDOW={EVENT_CLASS_WINDOW}",
    ]
    for k, v in extra.items():
        lines.append(f"{k}={v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _source_stem(src) -> str:
    if isinstance(src, int):
        return f"camera_{src}"
    p = Path(str(src)).expanduser()
    if p.suffix:
        return p.stem
    return p.name or "live"


def _recording_mp4_path(
        current_src: int | str,
        sources: list,
        events_dir: Path,
        single_video_file_out: Path,
) -> Path:
    if len(sources) == 1 and _is_video_file(current_src):
        return single_video_file_out
    return events_dir / f"{_source_stem(current_src)}.mp4"


def _tracking_group_name(class_name: str) -> str:
    if class_name in {rv.DOOR_CLASS_NAME, rv.TRIM_CLASS_NAME}:
        return "door_trim_group"
    return class_name


def _is_valid_object_size(class_name: str, box) -> tuple[bool, str]:
    x1, y1, x2, y2 = map(float, box)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h

    if class_name == rv.DOOR_CLASS_NAME:
        if w < MIN_DOOR_WIDTH_PX:
            return False, f"door_width_lt_{MIN_DOOR_WIDTH_PX}"
        if h < MIN_DOOR_HEIGHT_PX:
            return False, f"door_height_lt_{MIN_DOOR_HEIGHT_PX}"
        if area < MIN_DOOR_AREA_PX2:
            return False, f"door_area_lt_{MIN_DOOR_AREA_PX2}"
        return True, "ok"

    if class_name == rv.TRIM_CLASS_NAME:
        if w < MIN_TRIM_WIDTH_PX:
            return False, f"trim_width_lt_{MIN_TRIM_WIDTH_PX}"
        if h < MIN_TRIM_HEIGHT_PX:
            return False, f"trim_height_lt_{MIN_TRIM_HEIGHT_PX}"
        if area < MIN_TRIM_AREA_PX2:
            return False, f"trim_area_lt_{MIN_TRIM_AREA_PX2}"
        return True, "ok"

    return True, "ok"


def _mask_centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return 0, 0
    return int(xs.mean()), int(ys.mean())


def _blend_mask_contour_bgr(
    vis: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    opacity: float,
    contour_thickness: int,
) -> None:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return
    layer = vis.copy()
    layer[m] = color_bgr
    cv2.addWeighted(layer, opacity, vis, 1.0 - opacity, 0, dst=vis)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color_bgr, contour_thickness, lineType=cv2.LINE_AA)


def _copy_person_info_with_mask(person_info: dict | None, person_dets: list) -> dict | None:
    if person_info is None:
        return None
    out = dict(person_info)
    best_iou = 0.3
    best_mask = None
    for det in person_dets:
        if det.get("class_name") != rv.PERSON_CLASS_NAME or "mask" not in det:
            continue
        iou = rv.iou_xyxy(person_info["box"], det["box"])
        if iou > best_iou:
            best_iou = iou
            best_mask = det["mask"]
    if best_mask is not None:
        out["mask"] = np.asarray(best_mask, dtype=bool).copy()
    return out


class LiveEventProcessor:
    def __init__(self, src, run_dir: Path, log_fn):
        self.src = src
        self.run_dir = run_dir
        self.log = log_fn
        self.source_name = Path(str(src)).name if isinstance(src, str) else f"camera_{src}"
        self.source_stem = _source_stem(src)
        self.events_dir = run_dir / self.source_stem
        self.events_dir.mkdir(parents=True, exist_ok=True)

        # Верхняя линия (UP + первый порог DOWN)
        self.line = None
        self.top_side_sign_value = None
        self.bottom_side_sign_value = None

        # Нижняя линия (второй порог DOWN)
        self.line_down = None
        self.top_side_sign_value_down = None
        self.bottom_side_sign_value_down = None

        self.tracks = []
        self.next_track_id = 0
        self.events_count = 0
        self.events_count_down = 0
        self.rejected_count = 0
        self.rejected_count_down = 0

    def describe_output(self) -> str:
        return f"События и reject-кадры: {self.events_dir}"

    def _ensure_line(self, frame: np.ndarray) -> None:
        if self.line is not None:
            return
        h, w = frame.shape[:2]
        self.line = rv.denorm_line(STATIC_TRIPWIRE_LINE_NORM, w, h)
        (
            self.top_side_sign_value,
            self.bottom_side_sign_value,
        ) = rv.infer_top_bottom_sides(w, h, self.line)

        self.line_down = rv.denorm_line(STATIC_TRIPWIRE_DOWN_LINE_NORM, w, h)
        (
            self.top_side_sign_value_down,
            self.bottom_side_sign_value_down,
        ) = rv.infer_top_bottom_sides(w, h, self.line_down)

    def annotate_frame(self, frame: np.ndarray, all_dets: list, frame_idx: int) -> np.ndarray:
        self._ensure_line(frame)
        return _annotate_frame(frame, all_dets, frame_idx, self.line, self.line_down)

    # ------------------------------------------------------------------ #
    #  Вспомогательные методы для ВЕРХНЕЙ линии
    # ------------------------------------------------------------------ #

    def _is_clearly_below(self, box) -> bool:
        metrics = self._box_line_metrics(box)
        return metrics["state"] == "below" or metrics["center_d"] > LIVE_LINE_MARGIN_PX

    def _is_clearly_above(self, box) -> bool:
        metrics = self._box_line_metrics(box)
        return metrics["state"] == "above" or metrics["center_d"] < -LIVE_LINE_MARGIN_PX

    def _signed_distance_to_line(self, point: np.ndarray, line) -> float:
        a, b = line
        line_len = float(np.linalg.norm(b - a))
        if line_len < 1e-6:
            return 0.0
        return float(rv.side_of_line(point, a, b) / line_len)

    def _segment_intersects_box(self, box, line) -> bool:
        if line is None:
            return False
        x1, y1, x2, y2 = map(float, box)
        if x2 <= x1 or y2 <= y1:
            return False
        rect = (
            int(round(x1)),
            int(round(y1)),
            max(1, int(round(x2 - x1))),
            max(1, int(round(y2 - y1))),
        )
        a = (int(round(float(line[0][0]))), int(round(float(line[0][1]))))
        b = (int(round(float(line[1][0]))), int(round(float(line[1][1]))))
        retval, _, _ = cv2.clipLine(rect, a, b)
        return bool(retval)

    def _bottom_distance(self, box) -> float:
        point = rv.box_bottom_center(box)
        return self._signed_distance_to_line(point, self.line) * float(self.bottom_side_sign_value)

    def _center_distance(self, box) -> float:
        point = rv.center(box)
        return self._signed_distance_to_line(point, self.line) * float(self.bottom_side_sign_value)

    def _box_line_metrics(self, box) -> dict:
        return self._box_line_metrics_for(box, self.line, self.bottom_side_sign_value)

    def _box_line_metrics_for(self, box, line, bottom_sign) -> dict:
        """Универсальный расчёт метрик для произвольной линии."""
        x1, y1, x2, y2 = map(float, box)
        points = [
            np.array([x1, y1], dtype=np.float32),
            np.array([x2, y1], dtype=np.float32),
            np.array([x1, y2], dtype=np.float32),
            np.array([x2, y2], dtype=np.float32),
        ]
        distances = [
            self._signed_distance_to_line(p, line) * float(bottom_sign)
            for p in points
        ]
        center_pt = rv.center(box)
        center_d = self._signed_distance_to_line(center_pt, line) * float(bottom_sign)
        bottom_pt = rv.box_bottom_center(box)
        bottom_d = self._signed_distance_to_line(bottom_pt, line) * float(bottom_sign)

        min_d = min(distances)
        max_d = max(distances)
        if min_d > LIVE_LINE_MARGIN_PX:
            state = "below"
        elif max_d < -LIVE_LINE_MARGIN_PX:
            state = "above"
        else:
            state = "intersects"

        if state == "intersects" and not self._segment_intersects_box(box, line):
            state = "below" if center_d > 0.0 else "above"

        return {
            "state": state,
            "min_d": float(min_d),
            "max_d": float(max_d),
            "center_d": float(center_d),
            "bottom_d": float(bottom_d),
        }

    # ------------------------------------------------------------------ #
    #  Вспомогательные методы для НИЖНЕЙ линии
    # ------------------------------------------------------------------ #

    def _box_line_metrics_down(self, box) -> dict:
        return self._box_line_metrics_for(box, self.line_down, self.bottom_side_sign_value_down)

    def _is_clearly_above_down_line(self, box) -> bool:
        """Объект явно выше нижней линии (ещё не пересёк её)."""
        metrics = self._box_line_metrics_down(box)
        return metrics["state"] == "above" or metrics["center_d"] < -LIVE_LINE_MARGIN_PX

    def _is_clearly_below_down_line(self, box) -> bool:
        """Объект явно ниже нижней линии (уже прошёл через неё)."""
        metrics = self._box_line_metrics_down(box)
        return metrics["state"] == "below" or metrics["center_d"] > LIVE_LINE_MARGIN_PX

    # ------------------------------------------------------------------ #
    #  Перемещение трека
    # ------------------------------------------------------------------ #

    def _recent_y_travel(self, track) -> float:
        hist = track.get("history", [])
        if len(hist) < 2:
            return 0.0
        window = hist[-LIVE_MOVE_WINDOW_FRAMES:]
        if len(window) < 2:
            return 0.0
        y_first = float(rv.center(window[0]["box"])[1])
        y_last = float(rv.center(window[-1]["box"])[1])
        return y_first - y_last

    # ------------------------------------------------------------------ #
    #  UP: пересечение верхней линии снизу вверх
    # ------------------------------------------------------------------ #

    def _track_started_below_relaxed(self, track) -> bool:
        if "ever_below" in track:
            return bool(track["ever_below"])
        return any(self._is_clearly_below(item["box"]) for item in track["history"])

    def _crossed_line_from_below_relaxed(self, track) -> bool:
        if len(track["history"]) < 2:
            return False

        prev_metrics = self._box_line_metrics(track["history"][-2]["box"])
        curr_metrics = self._box_line_metrics(track["history"][-1]["box"])
        prev_center_d = prev_metrics["center_d"]
        curr_center_d = curr_metrics["center_d"]

        if not (
                track.get("ever_below", False)
                and prev_metrics["state"] in {"below", "intersects"}
                and curr_metrics["state"] == "intersects"
                and prev_center_d > 0.0
                and curr_center_d <= 0.0
                and curr_center_d < prev_center_d
        ):
            return False

        if track.get("below_frames", 0) < LIVE_REQUIRED_BELOW_FRAMES:
            return False

        if self._recent_y_travel(track) < LIVE_MIN_UP_TRAVEL_PX:
            return False

        return True

    # ------------------------------------------------------------------ #
    #  DOWN: двухлинейная логика (занос)
    #
    #  Условие:
    #    1. Трек когда-либо был выше верхней линии  (ever_above_upper)
    #    2. Трек пересёк верхнюю линию сверху вниз  (crossed_upper_down)
    #    3. Трек пересёк нижнюю линию сверху вниз   (crossed_lower_down)
    #
    #  Порядок: сначала верхняя, потом нижняя.
    #  Флаги хранятся в треке, чтобы не считать повторно.
    # ------------------------------------------------------------------ #

    def _is_clearly_above_upper(self, box) -> bool:
        metrics = self._box_line_metrics(box)
        return metrics["state"] == "above" or metrics["center_d"] < -LIVE_LINE_MARGIN_PX

    def _update_down_stage_flags(self, track: dict, frame_idx: int) -> None:
        """
        Обновляет флаги прохождения этапов DOWN в треке.
        Вызывается каждый кадр после обновления истории.
        """
        box = track["box"]

        # --- Этап 0: трек хоть раз был явно выше верхней линии ---
        if not track.get("down_ever_above_upper", False):
            if self._is_clearly_above_upper(box):
                track["down_ever_above_upper"] = True

        # --- Этап 1: пересечение верхней линии сверху вниз ---
        if track.get("down_ever_above_upper", False) and not track.get("down_crossed_upper", False):
            hist = track.get("history", [])
            if len(hist) >= 2:
                pm = self._box_line_metrics(hist[-2]["box"])
                cm = self._box_line_metrics(hist[-1]["box"])
                # центр перешёл с "above" на "intersects/below"
                if (
                    pm["center_d"] < 0.0
                    and cm["center_d"] >= 0.0
                    and pm["state"] in {"above", "intersects"}
                    and cm["state"] in {"intersects", "below"}
                ):
                    if track.get("above_frames_upper", 0) >= LIVE_REQUIRED_ABOVE_FRAMES:
                        travel = -self._recent_y_travel(track)  # движение вниз → travel > 0
                        if travel >= LIVE_MIN_DOWN_TRAVEL_PX:
                            track["down_crossed_upper"] = True
                            track["down_crossed_upper_frame"] = frame_idx
                            if LIVE_DEBUG_EVENT_DECISIONS:
                                self.log(
                                    f"[DOWN stage1] {self.source_name} track={track['id']} "
                                    f"frame={frame_idx} пересёк ВЕРХНЮЮ линию ↓"
                                )

        # --- Этап 2: пересечение нижней линии сверху вниз (только после этапа 1) ---
        if track.get("down_crossed_upper", False) and not track.get("down_crossed_lower", False):
            hist = track.get("history", [])
            if len(hist) >= 2:
                pm_d = self._box_line_metrics_down(hist[-2]["box"])
                cm_d = self._box_line_metrics_down(hist[-1]["box"])
                if (
                    pm_d["center_d"] < 0.0
                    and cm_d["center_d"] >= 0.0
                    and pm_d["state"] in {"above", "intersects"}
                    and cm_d["state"] in {"intersects", "below"}
                ):
                    track["down_crossed_lower"] = True
                    track["down_crossed_lower_frame"] = frame_idx
                    if LIVE_DEBUG_EVENT_DECISIONS:
                        self.log(
                            f"[DOWN stage2] {self.source_name} track={track['id']} "
                            f"frame={frame_idx} пересёк НИЖНЮЮ линию ↓ → READY"
                        )

    def _down_event_ready(self, track: dict) -> bool:
        """True, если трек прошёл оба этапа и событие ещё не засчитано."""
        return (
            not track.get("counted_down", False)
            and track.get("down_crossed_upper", False)
            and track.get("down_crossed_lower", False)
        )

    # ------------------------------------------------------------------ #
    #  Поиск person для объекта
    # ------------------------------------------------------------------ #

    def _find_best_person_for_object_live(self, obj_det, person_dets, frame_shape):
        best = rv.find_best_person_for_object(obj_det, person_dets, frame_shape)
        if best is not None:
            return _copy_person_info_with_mask(best, person_dets)

        obj_box = obj_det["box"]
        obj_c = rv.center(obj_box)
        ow, oh, _ = rv.box_wh_area(obj_box)
        obj_diag = float(np.hypot(ow, oh))

        fh, fw = frame_shape[:2]
        roi = rv.expand_box(
            obj_box,
            fw,
            fh,
            LIVE_RELAXED_ROI_EXPAND_X,
            LIVE_RELAXED_ROI_EXPAND_Y,
        )
        rx1, ry1, rx2, ry2 = roi

        best_score = -1e18
        for p in person_dets:
            p_box = p["box"]
            p_c = rv.center(p_box)
            if not (rx1 <= p_c[0] <= rx2 and ry1 <= p_c[1] <= ry2):
                continue
            dist = float(np.linalg.norm(p_c - obj_c))
            if dist > obj_diag * LIVE_RELAXED_MAX_PERSON_CENTER_DIST_FACTOR:
                continue
            overlap = rv.iou_xyxy(roi, p_box)
            pw, ph, parea = rv.box_wh_area(p_box)
            score = p["confidence"] * 1000.0 + parea * 0.01 - dist * 12.0 + overlap * 300.0
            if score > best_score:
                best_score = score
                best = {
                    "box": p_box.copy(),
                    "confidence": p["confidence"],
                    "w": int(pw),
                    "h": int(ph),
                    "area": int(parea),
                    "center_dist": dist,
                    "iou_with_roi": overlap,
                    "relaxed": True,
                }

        return _copy_person_info_with_mask(best, person_dets)

    # ------------------------------------------------------------------ #
    #  Evaluate UP (старая логика, без изменений)
    # ------------------------------------------------------------------ #

    def _evaluate_track(self, track: dict) -> dict:
        prev_d = None
        prev_center_d = None
        prev_box_state = None
        if len(track["history"]) >= 2:
            prev_metrics = self._box_line_metrics(track["history"][-2]["box"])
            prev_d = prev_metrics["bottom_d"]
            prev_center_d = prev_metrics["center_d"]
            prev_box_state = prev_metrics["state"]
        curr_metrics = self._box_line_metrics(track["history"][-1]["box"])
        curr_d = curr_metrics["bottom_d"]

        relaxed_started_below = self._track_started_below_relaxed(track)
        relaxed_crossed = self._crossed_line_from_below_relaxed(track)

        reason = None
        if track["counted_up"]:
            reason = "ALREADY_COUNTED_UP"
        elif len(track["history"]) < LIVE_MIN_TRACK_HISTORY:
            reason = f"HISTORY_LT_{LIVE_MIN_TRACK_HISTORY}"
        elif not relaxed_started_below:
            reason = "NOT_STARTED_BELOW_LINE"
        elif not relaxed_crossed:
            reason = "NO_CROSS_YET"
        else:
            reason = "READY_EVENT"

        return {
            "reason": reason,
            "curr_person": None,
            "person_info": None,
            "has_person": True,
            "hist_has_person": False,
            "prev_bottom_dist": prev_d,
            "curr_bottom_dist": curr_d,
            "prev_center_dist": prev_center_d,
            "curr_center_dist": curr_metrics["center_d"],
            "prev_box_state": prev_box_state,
            "curr_box_state": curr_metrics["state"],
            "curr_box_min_d": curr_metrics["min_d"],
            "curr_box_max_d": curr_metrics["max_d"],
            "strict_started_below": relaxed_started_below,
            "relaxed_started_below": relaxed_started_below,
            "strict_crossed": relaxed_crossed,
            "relaxed_crossed": relaxed_crossed,
        }

    def _get_event_class_recent(self, track: dict, n: int = EVENT_CLASS_WINDOW) -> str:
        hist = track.get("history", [])
        if not hist:
            return track.get("class_name_current", track.get("tracking_group", "unknown"))
        recent = hist[-n:]
        cnt = Counter(item["class_name"] for item in recent if "class_name" in item)
        if not cnt:
            return track.get("class_name_current", track.get("tracking_group", "unknown"))
        best_count = max(cnt.values())
        candidates = {cls for cls, v in cnt.items() if v == best_count}
        for item in reversed(recent):
            cls = item.get("class_name")
            if cls in candidates:
                return cls
        return recent[-1].get("class_name", track.get("class_name_current", "unknown"))

    def _log_track_decision(self, track: dict, frame_idx: int, eval_data: dict) -> None:
        if not LIVE_DEBUG_EVENT_DECISIONS:
            return
        ow, oh, oarea = rv.box_wh_area(track["box"])
        recent_event_class = self._get_event_class_recent(track)
        parts = [
            f"[TRACK ?] {self.source_name}",
            f"frame={frame_idx}",
            f"track_id={track['id']}",
            f"tracking_group={track['tracking_group']}",
            f"class_current={track['class_name_current']}",
            f"class_recent={recent_event_class}",
            f"conf={track['confidence']:.2f}",
            f"hist={len(track['history'])}",
            f"lost={track['lost']}",
            f"obj_w={int(ow)}",
            f"obj_h={int(oh)}",
            f"obj_area={int(oarea)}",
            f"started_below={int(eval_data['strict_started_below'])}/{int(eval_data['relaxed_started_below'])}",
            f"crossed={int(eval_data['strict_crossed'])}/{int(eval_data['relaxed_crossed'])}",
            f"has_person={int(eval_data['has_person'])}",
            f"hist_person={int(eval_data['hist_has_person'])}",
            f"reason={eval_data['reason']}",
            f"down_stage=upper:{int(track.get('down_crossed_upper', False))}"
            f"/lower:{int(track.get('down_crossed_lower', False))}",
        ]
        if eval_data["prev_box_state"] is not None:
            parts.append(f"prev_box={eval_data['prev_box_state']}")
        parts.append(f"curr_box={eval_data['curr_box_state']}")
        if eval_data["prev_bottom_dist"] is not None:
            parts.append(f"prev_line_d={eval_data['prev_bottom_dist']:.1f}")
        parts.append(f"curr_line_d={eval_data['curr_bottom_dist']:.1f}")
        self.log(" | ".join(parts))

    def _save_debug_frame(self, filename: str, image: np.ndarray) -> None:
        out_path = self.events_dir / filename
        if cv2.imwrite(str(out_path), image):
            self.log(f"  сохранён кадр: {out_path}")

    def _draw_reject_frame(
        self,
        frame: np.ndarray,
        obj_box,
        obj_class: str,
        ow: float,
        oh: float,
        obj_mask: np.ndarray | None = None,
        banner: str = "REJECT | no person near object",
    ) -> np.ndarray:
        rej = frame.copy()
        if not rv.DRAW:
            return rej
        obj_color = (255, 0, 0)
        if obj_mask is not None and np.any(obj_mask):
            _blend_mask_contour_bgr(rej, obj_mask, obj_color, LIVE_MASK_OPACITY, LIVE_POLYGON_THICKNESS)
            cx, cy = _mask_centroid(obj_mask)
            cv2.putText(
                rej,
                f"{obj_class.upper()} REJECT {int(ow)}x{int(oh)}",
                (max(5, cx - 100), max(22, cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, obj_color, 2, cv2.LINE_AA,
            )
        else:
            rv.draw_box_with_label(
                rej, obj_box, f"{obj_class.upper()} REJECT {int(ow)}x{int(oh)}", obj_color, thickness=3,
            )
        cv2.putText(rej, banner, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(rej, tuple(self.line[0].astype(int)), tuple(self.line[1].astype(int)), (255, 255, 255), 2)
        cv2.line(rej, tuple(self.line_down[0].astype(int)), tuple(self.line_down[1].astype(int)), (0, 255, 255), 2)
        return rej

    def _draw_event_frame(
        self,
        frame: np.ndarray,
        track: dict,
        obj_box,
        obj_class: str,
        ow: float,
        oh: float,
        person_info,
        assoc_str: str,
        obj_mask: np.ndarray | None = None,
        direction: str = "UP",
    ) -> np.ndarray:
        out = frame.copy()
        if not rv.DRAW:
            return out
        obj_color = (255, 255, 0) if obj_class == rv.DOOR_CLASS_NAME else (255, 0, 255)
        if obj_mask is not None and np.any(obj_mask):
            _blend_mask_contour_bgr(out, obj_mask, obj_color, LIVE_MASK_OPACITY, LIVE_POLYGON_THICKNESS)
            cx, cy = _mask_centroid(obj_mask)
            cv2.putText(
                out,
                f"{obj_class.upper()} {int(ow)}x{int(oh)} conf={track['confidence']:.2f}",
                (max(5, cx - 120), max(22, cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, obj_color, 2, cv2.LINE_AA,
            )
        else:
            rv.draw_box_with_label(
                out, obj_box,
                f"{obj_class.upper()} {int(ow)}x{int(oh)} conf={track['confidence']:.2f}",
                obj_color, thickness=3,
            )
        if person_info is not None:
            pm = person_info.get("mask")
            if pm is not None and np.any(pm):
                _blend_mask_contour_bgr(out, pm, (0, 255, 0), LIVE_MASK_OPACITY, LIVE_POLYGON_THICKNESS)
                pcx, pcy = _mask_centroid(pm)
                cv2.putText(
                    out,
                    f"PERSON {person_info['w']}x{person_info['h']} conf={person_info['confidence']:.2f}",
                    (max(5, pcx - 120), max(22, pcy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
                )
            else:
                rv.draw_box_with_label(
                    out, person_info["box"],
                    f"PERSON {person_info['w']}x{person_info['h']} conf={person_info['confidence']:.2f}",
                    (0, 255, 0), thickness=3,
                )
        cv2.putText(out, f"{direction} | {assoc_str}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(out, tuple(self.line[0].astype(int)), tuple(self.line[1].astype(int)), (255, 255, 255), 2)
        cv2.line(out, tuple(self.line_down[0].astype(int)), tuple(self.line_down[1].astype(int)), (0, 255, 255), 2)
        return out

    # ------------------------------------------------------------------ #
    #  Главный метод обработки кадра
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray, all_dets: list, frame_idx: int) -> np.ndarray:
        self._ensure_line(frame)
        debug_frame = frame.copy()
        vis = _annotate_frame(debug_frame, all_dets, frame_idx, self.line, self.line_down)

        primary_dets, person_dets = rv.split_detections(all_dets)

        enriched_primary = []
        for det in primary_dets:
            is_valid_size, invalid_reason = _is_valid_object_size(det["class_name"], det["box"])

            if FILTER_SMALL_OBJECTS_BEFORE_TRACKING and not is_valid_size:
                ow, oh, oarea = rv.box_wh_area(det["box"])
                self.log(
                    f"[SKIP SMALL] {self.source_name} | frame={frame_idx} | "
                    f"class={det['class_name']} | conf={det['confidence']:.2f} | "
                    f"w={int(ow)} h={int(oh)} area={int(oarea)} | reason={invalid_reason}"
                )
                continue

            person_info = self._find_best_person_for_object_live(det, person_dets, frame.shape)
            det2 = det.copy()
            det2["person_info"] = person_info
            det2["tracking_group"] = _tracking_group_name(det["class_name"])
            det2["size_valid"] = is_valid_size
            det2["size_invalid_reason"] = invalid_reason
            enriched_primary.append(det2)

        updated_ids = set()
        for det in enriched_primary:
            obj_box = det["box"]
            obj_class = det["class_name"]
            tracking_group = det["tracking_group"]
            obj_c = rv.center(obj_box)

            best_track = None
            best_score = -1e18

            for tr in self.tracks:
                if tr["tracking_group"] != tracking_group:
                    continue
                prev_c = tr["centers"][-1]
                dist = float(np.linalg.norm(obj_c - prev_c))
                if dist > rv.TRACK_DISTANCE:
                    continue
                overlap = rv.iou_xyxy(obj_box, tr["box"])
                score = overlap * 1000.0 - dist
                if score > best_score:
                    best_score = score
                    best_track = tr

            if best_track is not None:
                is_below_now = self._is_clearly_below(obj_box)
                is_above_now = self._is_clearly_above(obj_box)
                is_above_upper_now = self._is_clearly_above_upper(obj_box)

                best_track["box"] = obj_box.copy()
                best_track["confidence"] = det["confidence"]
                best_track["class_name_current"] = obj_class
                best_track["person_info_current"] = det["person_info"]
                best_track["lost"] = 0
                best_track["ever_below"] = best_track["ever_below"] or is_below_now
                best_track["ever_above"] = best_track["ever_above"] or is_above_now
                best_track["below_frames"] = best_track.get("below_frames", 0) + (1 if is_below_now else 0)
                best_track["above_frames"] = best_track.get("above_frames", 0) + (1 if is_above_now else 0)
                # счётчик кадров "явно выше верхней линии" — для DOWN
                best_track["above_frames_upper"] = best_track.get("above_frames_upper", 0) + (
                    1 if is_above_upper_now else 0
                )
                best_track["centers"].append(obj_c)
                best_track["updated_this_frame"] = True

                if len(best_track["centers"]) > rv.TRACK_HISTORY:
                    best_track["centers"].pop(0)

                hist_item = {
                    "frame_id": frame_idx,
                    "box": obj_box.copy(),
                    "confidence": det["confidence"],
                    "class_name": obj_class,
                    "person_info": det["person_info"],
                }
                if "mask" in det:
                    hist_item["mask"] = det["mask"].copy()
                best_track["history"].append(hist_item)
                if len(best_track["history"]) > rv.TRACK_HISTORY:
                    best_track["history"].pop(0)

                # Обновляем флаги DOWN-этапов
                self._update_down_stage_flags(best_track, frame_idx)

                updated_ids.add(best_track["id"])
            else:
                is_below_now = self._is_clearly_below(obj_box)
                is_above_now = self._is_clearly_above(obj_box)
                is_above_upper_now = self._is_clearly_above_upper(obj_box)
                hist_item = {
                    "frame_id": frame_idx,
                    "box": obj_box.copy(),
                    "confidence": det["confidence"],
                    "class_name": obj_class,
                    "person_info": det["person_info"],
                }
                if "mask" in det:
                    hist_item["mask"] = det["mask"].copy()
                new_track = {
                    "id": self.next_track_id,
                    "tracking_group": tracking_group,
                    "class_name_current": obj_class,
                    "box": obj_box.copy(),
                    "confidence": det["confidence"],
                    "rejected_small": False,
                    "person_info_current": det["person_info"],
                    "lost": 0,
                    "counted_up": False,
                    "counted_down": False,
                    "rejected": False,
                    "rejected_down": False,
                    "rejected_small_down": False,
                    "updated_this_frame": True,
                    "ever_below": is_below_now,
                    "ever_above": is_above_now,
                    "below_frames": 1 if is_below_now else 0,
                    "above_frames": 1 if is_above_now else 0,
                    "above_frames_upper": 1 if is_above_upper_now else 0,
                    "centers": [obj_c],
                    "history": [hist_item],
                    # DOWN-этапы
                    "down_ever_above_upper": is_above_upper_now,
                    "down_crossed_upper": False,
                    "down_crossed_lower": False,
                    "down_crossed_upper_frame": None,
                    "down_crossed_lower_frame": None,
                }
                self._update_down_stage_flags(new_track, frame_idx)
                self.tracks.append(new_track)
                updated_ids.add(self.next_track_id)
                self.next_track_id += 1

        alive_tracks = []
        for tr in self.tracks:
            if tr["id"] not in updated_ids:
                tr["lost"] += 1
                tr["updated_this_frame"] = False
            if tr["lost"] <= rv.MAX_LOST:
                alive_tracks.append(tr)
        self.tracks = alive_tracks

        # ---------------------------------------------------------------- #
        #  UP events
        # ---------------------------------------------------------------- #
        for tr in self.tracks:
            if tr["counted_up"]:
                continue
            eval_data = self._evaluate_track(tr)
            if tr.get("updated_this_frame"):
                self._log_track_decision(tr, frame_idx, eval_data)

            if len(tr["history"]) < LIVE_MIN_TRACK_HISTORY:
                continue
            if not eval_data["relaxed_started_below"]:
                continue
            if not eval_data["relaxed_crossed"]:
                continue

            obj_box = tr["box"]
            obj_class = self._get_event_class_recent(tr, EVENT_CLASS_WINDOW)
            ow, oh, oarea = rv.box_wh_area(obj_box)

            is_valid_size, invalid_reason = _is_valid_object_size(obj_class, obj_box)
            if not is_valid_size:
                if not tr.get("rejected_small", False):
                    tr["rejected_small"] = True
                    self.log(
                        f"[REJECT SMALL ✖] {self.source_name} | frame={frame_idx} | "
                        f"class={obj_class} | obj_w={int(ow)} obj_h={int(oh)} obj_area={int(oarea)} | "
                        f"reason={invalid_reason}"
                    )
                continue

            tr["counted_up"] = True
            self.events_count += 1

            assoc_str = obj_class
            recent_classes = [item.get("class_name") for item in tr["history"][-EVENT_CLASS_WINDOW:]]
            log_data = {
                "video": self.source_name,
                "frame": frame_idx,
                "direction": "UP",
                "assoc": assoc_str,
                "object_class": obj_class,
                "object_class_window": EVENT_CLASS_WINDOW,
                "recent_classes": recent_classes,
                "object_w": int(ow),
                "object_h": int(oh),
                "object_area": int(oarea),
            }
            self.log("[EVENT ✔] " + json.dumps(log_data, ensure_ascii=False))

            if rv.SAVE:
                out = self._draw_event_frame(
                    debug_frame, tr, obj_box, obj_class, ow, oh, None, assoc_str,
                    tr["history"][-1].get("mask"),
                )
                filename = f"{self.source_stem}_UP_{assoc_str}_{frame_idx:06d}.jpg"
                self._save_debug_frame(filename, out)
                vis = out

        # ---------------------------------------------------------------- #
        #  DOWN events — двухлинейная логика
        # ---------------------------------------------------------------- #
        for tr in self.tracks:
            if tr["counted_down"]:
                continue
            if not self._down_event_ready(tr):
                continue

            obj_box = tr["box"]
            obj_class = self._get_event_class_recent(tr, EVENT_CLASS_WINDOW)
            if obj_class != rv.DOOR_CLASS_NAME:
                continue

            ow, oh, oarea = rv.box_wh_area(obj_box)
            is_valid_size, invalid_reason = _is_valid_object_size(obj_class, obj_box)
            if not is_valid_size:
                if not tr.get("rejected_small_down", False):
                    tr["rejected_small_down"] = True
                    self.log(
                        f"[REJECT SMALL ✖ DOWN] {self.source_name} | frame={frame_idx} | "
                        f"class={obj_class} | obj_w={int(ow)} obj_h={int(oh)} obj_area={int(oarea)} | "
                        f"reason={invalid_reason}"
                    )
                continue

            tr["counted_down"] = True
            self.events_count_down += 1

            assoc_str = obj_class
            recent_classes = [item.get("class_name") for item in tr["history"][-EVENT_CLASS_WINDOW:]]
            log_data = {
                "video": self.source_name,
                "frame": frame_idx,
                "direction": "DOWN",
                "assoc": assoc_str,
                "object_class": obj_class,
                "object_class_window": EVENT_CLASS_WINDOW,
                "recent_classes": recent_classes,
                "object_w": int(ow),
                "object_h": int(oh),
                "object_area": int(oarea),
                "down_upper_frame": tr.get("down_crossed_upper_frame"),
                "down_lower_frame": tr.get("down_crossed_lower_frame"),
            }
            self.log("[EVENT ✔ DOWN] " + json.dumps(log_data, ensure_ascii=False))

            if rv.SAVE:
                out = self._draw_event_frame(
                    debug_frame, tr, obj_box, obj_class, ow, oh, None, assoc_str,
                    tr["history"][-1].get("mask"),
                    direction="DOWN",
                )
                filename = f"{self.source_stem}_DOWN_{assoc_str}_{frame_idx:06d}.jpg"
                self._save_debug_frame(filename, out)
                vis = out

        return vis


if HAS_PYQT:

    def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


    class DetWorker(QThread):
        frame_ready = pyqtSignal(object)
        log = pyqtSignal(str)
        finished_ok = pyqtSignal()

        def __init__(
                self,
                src,
                model,
                class_names,
                save_mp4: bool,
                mp4_path: Path,
                out_fps: float,
                run_dir: Path,
        ):
            super().__init__()
            self.src = src
            self.model = model
            self.class_names = class_names
            self._save_mp4 = save_mp4
            self._mp4_path = mp4_path
            self._out_fps = out_fps
            self._run_dir = run_dir
            self._stop = False
            self._wait_ms = 1
            self.video_writer: cv2.VideoWriter | None = None
            self._jsonl_fp: object | None = None

        def request_stop(self) -> None:
            self._stop = True

        def run(self) -> None:
            _write_run_info(
                self._run_dir / "run_info.txt",
                self.src,
                {
                    "out_fps": self._out_fps,
                    "sources": len(_expand_sources(self.src)),
                    "model": "RFDETRSegMedium",
                    "seg_checkpoint": SEG_CHECKPOINT_PATH,
                },
            )
            if LOG_EVERY_N_FRAMES > 0:
                self._jsonl_fp = open(
                    self._run_dir / "detections.jsonl", "w", encoding="utf-8"
                )
            else:
                self._jsonl_fp = None
            _run_msg = (
                f"Папка прогона: {self._run_dir} | консоль + detections.jsonl каждые {LOG_EVERY_N_FRAMES} кадр(ов)"
                if LOG_EVERY_N_FRAMES > 0
                else f"Папка прогона: {self._run_dir} | jsonl выкл. (LOG_EVERY_N_FRAMES=0), есть run_info.txt"
            )
            self.log.emit(_run_msg)
            if SAVE_ANNOTATED_JPEG_EVERY_N > 0:
                self.log.emit(f"JPEG размеченных кадров: каждые {SAVE_ANNOTATED_JPEG_EVERY_N} кадр(ов)")
            sources = _expand_sources(self.src)
            if not sources:
                self.log.emit(f"Не найдены видео в папке: {self.src!r}")
                self.finished_ok.emit()
                return
            if len(sources) > 1:
                self.log.emit(f"Автопереключение по видео: найдено {len(sources)} файлов.")

            total_events = 0
            total_rejected = 0

            for source_idx, current_src in enumerate(sources, start=1):
                if self._stop:
                    break

                cap = _try_open_video_capture(current_src)
                if cap is None:
                    self.log.emit(_format_capture_open_error(current_src))
                    continue

                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                wait_ms = _playback_wait_ms(cap, current_src)
                out_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if out_fps < 1.0 or out_fps > 120.0:
                    out_fps = self._out_fps

                if _is_video_file(current_src):
                    self.log.emit(
                        f"[{source_idx}/{len(sources)}] {_source_label(current_src)} | "
                        f"пауза ~{wait_ms} ms между кадрами (по FPS)."
                    )
                else:
                    self.log.emit(f"[{source_idx}/{len(sources)}] {_source_label(current_src)} | поток без паузы.")

                event_processor = LiveEventProcessor(current_src, self._run_dir, self.log.emit)
                self.log.emit(event_processor.describe_output())
                _write_run_info(
                    event_processor.events_dir / "source_info.txt",
                    current_src,
                    {
                        "wait_ms": wait_ms,
                        "out_fps": out_fps,
                        "source_index": source_idx,
                        "seg_checkpoint": SEG_CHECKPOINT_PATH,
                    },
                )

                last_dets: list = []
                frame_idx = 0

                while not self._stop:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if LOOP_VIDEO_FILE and len(sources) == 1 and _is_video_file(current_src):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_idx = 0
                            self.log.emit("Повтор с начала файла.")
                            continue
                        self.log.emit(f"Конец источника: {_source_label(current_src)}")
                        break

                    frame_idx += 1
                    did_detect = DETECT_EVERY_N <= 1 or (frame_idx % DETECT_EVERY_N == 1)
                    if did_detect:
                        last_dets = _load_frame_detections_seg(self.model, frame, self.class_names)

                    if frame_idx == 1:
                        st0 = _det_stats(last_dets)
                        _tail = (
                            f"дальше сводка каждые {LOG_EVERY_N_FRAMES} кадр."
                            if LOG_EVERY_N_FRAMES > 0
                            else "периодическая сводка в консоль выключена (LOG_EVERY_N_FRAMES=0)"
                        )
                        self.log.emit(
                            f"Первый кадр {_source_label(current_src)}: объектов {len(last_dets)}, "
                            f"классы: {st0['counts']} ({_tail}; артефакты → {event_processor.events_dir})"
                        )

                    if did_detect:
                        vis = event_processor.process_frame(frame, last_dets, frame_idx)
                    else:
                        vis = event_processor.annotate_frame(frame, last_dets, frame_idx)
                    h, w = vis.shape[:2]

                    if self._save_mp4 and self.video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        target_mp4 = _recording_mp4_path(
                            current_src, sources, event_processor.events_dir, self._mp4_path
                        )
                        target_mp4.parent.mkdir(parents=True, exist_ok=True)
                        self.video_writer = cv2.VideoWriter(str(target_mp4), fourcc, out_fps, (w, h))
                        if not self.video_writer.isOpened():
                            self.log.emit(f"Не удалось открыть VideoWriter: {target_mp4}")
                            self._save_mp4 = False

                    if self.video_writer is not None and self.video_writer.isOpened():
                        self.video_writer.write(vis)

                    if LOG_EVERY_N_FRAMES > 0 and frame_idx % LOG_EVERY_N_FRAMES == 0:
                        stats = _det_stats(last_dets)
                        self.log.emit(
                            f"[кадр {frame_idx}] всего {len(last_dets)} | по классам: {stats['counts']}"
                        )
                        rec = {"video": _source_label(current_src), "frame": frame_idx, **stats}
                        if self._jsonl_fp is not None:
                            self._jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            self._jsonl_fp.flush()

                    if (
                            SAVE_ANNOTATED_JPEG_EVERY_N > 0
                            and frame_idx % SAVE_ANNOTATED_JPEG_EVERY_N == 0
                    ):
                        jpeg_path = event_processor.events_dir / f"annotated_{frame_idx:06d}.jpg"
                        if cv2.imwrite(str(jpeg_path), vis):
                            self.log.emit(f"  сохранён кадр: {jpeg_path}")

                    self.frame_ready.emit(vis)
                    if wait_ms > 0:
                        self.msleep(wait_ms)

                cap.release()
                total_events += event_processor.events_count
                total_events += event_processor.events_count_down
                total_rejected += event_processor.rejected_count
                total_rejected += event_processor.rejected_count_down
                self.log.emit(
                    f"Итог по {_source_label(current_src)}: события UP {event_processor.events_count}, "
                    f"DOWN {event_processor.events_count_down}; отклонено UP {event_processor.rejected_count}, "
                    f"DOWN {event_processor.rejected_count_down}"
                )
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None

            if self._jsonl_fp is not None:
                self._jsonl_fp.close()
                self._jsonl_fp = None
            self.log.emit(
                f"Итог по событиям (UP+DOWN): найдено {total_events}, "
                f"отклонено {total_rejected}"
            )
            self.finished_ok.emit()


    class MainWindow(QMainWindow):
        def __init__(self, worker: DetWorker):
            super().__init__()
            self.setWindowTitle(WINDOW_TITLE)
            self._worker = worker
            self._full_pix: QPixmap | None = None

            central = QWidget()
            layout = QHBoxLayout(central)
            self._label = QLabel()
            self._label.setAlignment(Qt.AlignCenter)
            self._label.setMinimumSize(960, 540)
            self._label.setStyleSheet("background-color: #1a1a1a;")
            layout.addWidget(self._label)
            self.setCentralWidget(central)

            worker.frame_ready.connect(self._on_frame)
            worker.log.connect(self._on_log)

            QShortcut(QKeySequence("Q"), self, activated=self.close)
            QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.close)

        def _apply_scale(self) -> None:
            if self._full_pix is None or self._full_pix.isNull():
                return
            self._label.setPixmap(
                self._full_pix.scaled(
                    self._label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

        def _on_frame(self, vis: np.ndarray) -> None:
            self._full_pix = QPixmap.fromImage(_bgr_to_qimage(vis))
            self._apply_scale()

        def _on_log(self, msg: str) -> None:
            print(msg, flush=True)

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            self._apply_scale()

        def closeEvent(self, event) -> None:
            self._worker.request_stop()
            self._worker.wait(15000)
            if self._worker.video_writer is not None:
                self._worker.video_writer.release()
                self._worker.video_writer = None
            event.accept()


def run_headless_mp4(model, class_names, src) -> int:
    run_dir = _new_run_dir()
    _write_run_info(
        run_dir / "run_info.txt",
        src,
        {
            "mode": "headless_seg",
            "sources": len(_expand_sources(src)),
            "model": "RFDETRSegMedium",
            "seg_checkpoint": SEG_CHECKPOINT_PATH,
        },
    )
    jsonl_fp = None
    if LOG_EVERY_N_FRAMES > 0:
        jsonl_fp = open(run_dir / "detections.jsonl", "w", encoding="utf-8")
    print(f"Логи прогона: {run_dir}", flush=True)
    rc = 0
    total_events = 0
    total_rejected = 0
    sources = _expand_sources(src)
    if not sources:
        print(f"Не найдены видео в папке: {src!r}", flush=True)
        if jsonl_fp is not None:
            jsonl_fp.close()
        return 1
    if len(sources) > 1:
        print(f"Автопереключение по видео: найдено {len(sources)} файлов.", flush=True)

    try:
        for source_idx, current_src in enumerate(sources, start=1):
            cap = _try_open_video_capture(current_src)
            if cap is None:
                print(_format_capture_open_error(current_src), flush=True)
                rc = 1
                continue

            out_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if out_fps < 1.0 or out_fps > 120.0:
                out_fps = 25.0
            wait_ms = _playback_wait_ms(cap, current_src)
            print(
                f"[{source_idx}/{len(sources)}] {_source_label(current_src)} | пауза ~{wait_ms} ms",
                flush=True,
            )

            writer = None
            last_dets: list = []
            frame_idx = 0
            event_processor = LiveEventProcessor(current_src, run_dir, lambda msg: print(msg, flush=True))
            print(event_processor.describe_output(), flush=True)
            _write_run_info(
                event_processor.events_dir / "source_info.txt",
                current_src,
                {
                    "wait_ms": wait_ms,
                    "out_fps": out_fps,
                    "mode": "headless_seg",
                    "source_index": source_idx,
                    "seg_checkpoint": SEG_CHECKPOINT_PATH,
                },
            )

            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if LOOP_VIDEO_FILE and len(sources) == 1 and _is_video_file(current_src):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        continue
                    break
                frame_idx += 1
                did_detect = DETECT_EVERY_N <= 1 or (frame_idx % DETECT_EVERY_N == 1)
                if did_detect:
                    last_dets = _load_frame_detections_seg(model, frame, class_names)
                if frame_idx == 1:
                    st0 = _det_stats(last_dets)
                    print(
                        f"Первый кадр {_source_label(current_src)}: объектов {len(last_dets)}, классы: {st0['counts']}",
                        flush=True,
                    )
                if did_detect:
                    vis = event_processor.process_frame(frame, last_dets, frame_idx)
                else:
                    vis = event_processor.annotate_frame(frame, last_dets, frame_idx)
                h, w = vis.shape[:2]
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    target_mp4 = _recording_mp4_path(
                        current_src, sources, event_processor.events_dir, MP4_OUTPUT
                    )
                    writer = cv2.VideoWriter(str(target_mp4), fourcc, out_fps, (w, h))
                    if not writer.isOpened():
                        print("Не удалось открыть VideoWriter", flush=True)
                        rc = 1
                        break
                writer.write(vis)

                if LOG_EVERY_N_FRAMES > 0 and frame_idx % LOG_EVERY_N_FRAMES == 0:
                    stats = _det_stats(last_dets)
                    print(
                        f"[кадр {frame_idx}] всего {len(last_dets)} | по классам: {stats['counts']}",
                        flush=True,
                    )
                    if jsonl_fp is not None:
                        jsonl_fp.write(
                            json.dumps({"video": _source_label(current_src), "frame": frame_idx, **stats},
                                       ensure_ascii=False)
                            + "\n"
                        )
                        jsonl_fp.flush()
                if (
                        SAVE_ANNOTATED_JPEG_EVERY_N > 0
                        and frame_idx % SAVE_ANNOTATED_JPEG_EVERY_N == 0
                ):
                    jpeg_path = event_processor.events_dir / f"annotated_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(jpeg_path), vis)
                    print(f"  сохранён кадр: {jpeg_path}", flush=True)

                if wait_ms > 0:
                    cv2.waitKey(wait_ms)

            total_events += event_processor.events_count
            total_events += event_processor.events_count_down
            total_rejected += event_processor.rejected_count
            total_rejected += event_processor.rejected_count_down
            print(
                f"Итог по {_source_label(current_src)}: события UP {event_processor.events_count}, "
                f"DOWN {event_processor.events_count_down}; отклонено UP {event_processor.rejected_count}, "
                f"DOWN {event_processor.rejected_count_down}",
                flush=True,
            )
            cap.release()
            if writer is not None:
                writer.release()
    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()

    print(
        f"Итог по событиям (UP+DOWN): найдено {total_events}, "
        f"отклонено {total_rejected}",
        flush=True,
    )
    if rc == 0:
        print(f"Готово: {MP4_OUTPUT} | логи: {run_dir}", flush=True)
    return rc


def main() -> int:
    if not Path(SEG_CHECKPOINT_PATH).exists():
        print(f"Чекпоинт сегментации не найден: {SEG_CHECKPOINT_PATH}")
        return 1
    if not Path(rv.DATA_YAML_PATH).exists():
        print(f"data.yaml не найден: {rv.DATA_YAML_PATH}")
        return 1

    if os.environ.get("RFDETR_LIVE_HEADLESS") == "1":
        class_names = rv.load_class_names_from_yaml(rv.DATA_YAML_PATH)
        print("RF-DETR Seg | классы:", ", ".join(f"{i}:{n}" for i, n in enumerate(class_names)))
        model = _build_seg_model(SEG_CHECKPOINT_PATH, num_classes=len(class_names))
        return run_headless_mp4(model, class_names, LIVE_SOURCE)

    if not HAS_PYQT:
        print("Нужен PyQt5: pip install PyQt5")
        return 1

    class_names = rv.load_class_names_from_yaml(rv.DATA_YAML_PATH)
    print("RF-DETR Seg | классы:", ", ".join(f"{i}:{n}" for i, n in enumerate(class_names)))
    model = _build_seg_model(SEG_CHECKPOINT_PATH, num_classes=len(class_names))

    out_fps = _probe_fps(LIVE_SOURCE)
    run_dir = _new_run_dir()
    print(f"Логи и артефакты прогона: {run_dir}", flush=True)

    app = QApplication(sys.argv)
    worker = DetWorker(
        LIVE_SOURCE, model, class_names, ALSO_SAVE_MP4, MP4_OUTPUT, out_fps, run_dir
    )
    win = MainWindow(worker)
    win.show()
    worker.start()
    return int(app.exec_())


if __name__ == "__main__":
    sys.exit(main())