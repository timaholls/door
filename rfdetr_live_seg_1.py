#!/usr/bin/env python3
"""
Эфир с RF-DETR Seg (instance segmentation): маски + контуры на кадре, логика событий.

Источник (LIVE_SOURCE): int — камера; str — один RTSP/файл; list — несколько URL.
Без окна: RFDETR_LIVE_HEADLESS=1 python rfdetr_live_seg.py
Чекпоинт: SEG_CHECKPOINT_PATH (веса сегментации).
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRSegMedium

if not hasattr(cv2, "CV_8U"):
    print(
        "Ошибка: OpenCV установлен неполно (нет cv2.CV_8U). Переустанови:\n"
        "  pip uninstall opencv-python opencv-python-headless -y\n"
        "  pip install opencv-python"
    )
    sys.exit(1)

import rfdetr_video_events as rv

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

STATIC_TRIPWIRE_LINE_NORM: list[tuple[float, float]] = [
    (0.135965, 0.613486),
    (0.684906, 0.483107),
    (0.135012, 0.606017),
]

LIVE_SOURCE_URLS: list[str] = [
    # "rtsp://viewer:ViewerPass_9347X@94.41.120.115:8554/cam44",
    # "rtsp://viewer:ViewerPass_9347X@94.41.120.115:8554/cam47",
    # "rtsp://viewer:ViewerPass_9347X@94.41.120.115:8554/cam49",
]
RECORDINGS_CAM0_DIR = str(Path(__file__).resolve().parent / "recordings_2/cam1")
_rtsp_one = os.environ.get("RFDETR_LIVE_RTSP", "").strip()
if _rtsp_one:
    LIVE_SOURCE: int | str | list = _rtsp_one
elif LIVE_SOURCE_URLS:
    LIVE_SOURCE = LIVE_SOURCE_URLS
else:
    LIVE_SOURCE = RECORDINGS_CAM0_DIR

WINDOW_TITLE = "RF-DETR Seg — эфир (PyQt) | Q / Esc — выход"

SEG_CHECKPOINT_PATH = str(
    Path(__file__).resolve().parent / "output_seg" / "checkpoint_best_total.pth"
)

LIVE_MASK_OPACITY = 0.45
LIVE_POLYGON_THICKNESS = 2
LOOP_VIDEO_FILE = False
DETECT_EVERY_N = 1
SHOW_TRIPWIRE = True
ALSO_SAVE_MP4 = False
MP4_OUTPUT = Path(__file__).resolve().parent / "rfdetr_live_out.mp4"

FILE_PLAYBACK_SLOWDOWN = 2.0
FILE_PLAYBACK_CAP_FPS = 18.0

LIVE_RUN_ROOT = Path(__file__).resolve().parent / "rfdetr_live_logs_1"
LOG_EVERY_N_FRAMES = 10  # каждые N кадров: «[кадр N] всего … | по классам: …»; 0 — выключить.
SAVE_ANNOTATED_JPEG_EVERY_N = 0

# --- события ---
LINE_TRIPWIRE_REARM_FRAMES = 15
LINE_EVENT_SIMPLE_FIRST_HIT_PER_TRACK = True
LINE_EVENT_SUPPRESS_FRAGMENT_OF_LATCHED_TRACK = True
LINE_EVENT_FRAGMENT_RECENT_FRAMES = 25
LINE_EVENT_FRAGMENT_IOU_THR = 0.55
LINE_EVENT_FRAGMENT_CONTAINMENT_THR = 0.85
LINE_EVENT_USE_PERSON_ASSOCIATION = False
LINE_EVENT_REJECT_DOOR_OVERLAPPING_PERSON = False
LINE_EVENT_PERSON_OVERLAP_HISTORY_FRAMES = 5

TRIPWIRE_MASK_LINE_THICKNESS_PX = 4
TRIPWIRE_MASK_MIN_OVERLAP_PX = 12

# Дедупликация LINE_CROSS: если новый трек той же двери пересекает линию,
# пока она ещё физически там — не выдавать повторное событие.
DOOR_LINE_EVENT_DEDUP_ENABLED = True
# Сколько кадров «замораживаем» зону после последнего события.
# При FPS~25 и двери 2-3 сек в кадре: 25*5=125. Ставим с запасом.
DOOR_LINE_EVENT_DEDUP_FRAMES = 80
# Если центр нового bbox дальше этого расстояния — это другая дверь, не дубль.
DOOR_LINE_EVENT_DEDUP_CENTER_DIST_PX = 500.0
# Минимальный IoU нового bbox с bbox последнего события (другой track_id).
# Низкий порог: дверь могла немного сместиться.
DOOR_LINE_EVENT_DEDUP_MIN_IOU_DIFFERENT_TRACK = 0.10

# SKIP/REJECT на каждом кадре и пр. — только при True (шумит в эфире).
LIVE_DEBUG_EVENT_DECISIONS = False
LIVE_DEBUG_DOOR_LINE_PER_FRAME_FORCE = False
LIVE_DEBUG_DOOR_LINE_PER_FRAME = False

LIVE_MIN_TRACK_HISTORY = 1
LIVE_LINE_MARGIN_PX = 18.0
LIVE_RELAXED_ROI_EXPAND_X = 0.45
LIVE_RELAXED_ROI_EXPAND_Y = 0.28
LIVE_RELAXED_MAX_PERSON_CENTER_DIST_FACTOR = 1.55

EVENT_CLASS_WINDOW = 7
EVENT_PRIMARY_CLASSES = (rv.DOOR_CLASS_NAME,)
EVENT_MIN_TRACK_HISTORY = 4
EVENT_MIN_MEAN_CONFIDENCE = 0.35
EVENT_MIN_MASK_OVERLAP_PX = 40
EVENT_MIN_MASK_OVERLAP_PER_BBOX_W = 0.15
EVENT_MIN_MASK_FILL_RATIO = 0.20

LINE_EVENT_REQUIRE_MOTION = True
EVENT_MOTION_LOOKBACK_FRAMES = 8
EVENT_STATIC_CENTER_TRAVEL_PX = 8.0
EVENT_STATIC_SIZE_DELTA_PX = 12.0
EVENT_STATIC_CENTER_TRAVEL_REL = 0.04
EVENT_STATIC_SIZE_DELTA_REL = 0.05

MIN_DOOR_WIDTH_PX = 200
MIN_DOOR_HEIGHT_PX = 280
MIN_DOOR_AREA_PX2 = 80000
MAX_DOOR_AREA_PX2 = 0
MAX_DOOR_ASPECT_HW = 3.0
MAX_DOOR_ASPECT_WH = 1.6     # было 3.0 — ширина не может быть больше высоты в 1.2 раза

MIN_TRIM_WIDTH_PX = 20
MIN_TRIM_HEIGHT_PX = 80
MIN_TRIM_AREA_PX2 = 2500

MIN_DOOR_CONFIDENCE = 0.30
MIN_TRIM_CONFIDENCE = 0.30

DOOR_PERSON_OVERLAP_IOU_THR = 0.45
DOOR_PERSON_OVERLAP_CONTAINMENT_THR = 0.60
DOOR_PERSON_OVERLAP_MASK_RATIO_THR = 0.55
DOOR_PERSON_AREA_RATIO_MAX = 1.5

FILTER_SMALL_OBJECTS_BEFORE_TRACKING = False

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


# ---------------------------------------------------------------------------
# Вспомогательные функции (источники / захват)
# ---------------------------------------------------------------------------

def _is_video_file(src) -> bool:
    return isinstance(src, str) and Path(src).expanduser().is_file()


def _is_video_dir(src) -> bool:
    return isinstance(src, str) and Path(src).expanduser().is_dir()


def _source_label(src) -> str:
    if isinstance(src, int):
        return f"camera_{src}"
    p = Path(str(src)).expanduser()
    return p.name or str(src)


def _source_stem(src) -> str:
    if isinstance(src, int):
        return f"camera_{src}"
    p = Path(str(src)).expanduser()
    return p.stem if p.suffix else (p.name or "live")


def _recording_mp4_path(
        current_src: int | str,
        sources: list,
        events_dir: Path,
        single_video_file_out: Path,
) -> Path:
    """Прямой эфир / поток: mp4 в папке прогона; один локальный видеофайл — путь как раньше."""
    if len(sources) == 1 and _is_video_file(current_src):
        return single_video_file_out
    return events_dir / f"{_source_stem(current_src)}.mp4"


def _list_video_files(src_dir: str) -> list[str]:
    base = Path(src_dir).expanduser()
    return sorted(
        str(p) for p in base.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _expand_sources(src) -> list:
    if isinstance(src, (list, tuple)):
        return list(src)
    if isinstance(src, int):
        return [src]
    if _is_video_dir(src):
        return _list_video_files(src)
    return [src]


def _try_open_video_capture(src) -> Optional[cv2.VideoCapture]:
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
    return cap if cap.isOpened() else None


def _format_capture_open_error(src) -> str:
    lines = [f"Не удалось открыть источник {src!r}."]
    if isinstance(src, int):
        lines += [
            "  Локальная камера:",
            "    • ls -l /dev/video* ; группа «video» (sudo usermod -aG video $USER)",
            "    • Другой индекс или RFDETR_LIVE_RTSP для IP-камеры.",
            "    • WSL/виртуалка: используйте RTSP или видеофайл.",
        ]
    else:
        lines += [
            "  URL/путь: проверьте сеть, учётные данные или доступность файла.",
        ]
    return "\n".join(lines)


def _playback_wait_ms(cap: cv2.VideoCapture, src) -> int:
    if isinstance(src, int):
        return 1
    if _is_video_file(src):
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps < 1.0 or fps > 120.0:
            fps = 25.0
        fps = min(fps, FILE_PLAYBACK_CAP_FPS)
        return max(1, round((1000.0 / fps) * FILE_PLAYBACK_SLOWDOWN))
    return 1


def _probe_fps(src) -> float:
    if isinstance(src, (list, tuple)):
        return _probe_fps(src[0]) if src else 25.0
    if _is_video_dir(src):
        videos = _list_video_files(src)
        src = videos[0] if videos else None
        if src is None:
            return 25.0
    cap = _try_open_video_capture(src)
    if cap is None:
        return 25.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if 1.0 <= fps <= 120.0 else 25.0


# ---------------------------------------------------------------------------
# Размер / валидация объекта
# ---------------------------------------------------------------------------

def _is_valid_object_size(class_name: str, box) -> tuple[bool, str]:
    x1, y1, x2, y2 = map(float, box)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    area = w * h
    if class_name == rv.DOOR_CLASS_NAME:
        if w < MIN_DOOR_WIDTH_PX:  return False, f"door_width_lt_{MIN_DOOR_WIDTH_PX}"
        if h < MIN_DOOR_HEIGHT_PX: return False, f"door_height_lt_{MIN_DOOR_HEIGHT_PX}"
        if area < MIN_DOOR_AREA_PX2:  return False, f"door_area_lt_{MIN_DOOR_AREA_PX2}"
        if MAX_DOOR_AREA_PX2 > 0 and area > MAX_DOOR_AREA_PX2:
            return False, f"door_area_gt_{MAX_DOOR_AREA_PX2}"
        if MAX_DOOR_ASPECT_HW > 0 and w > 0 and (h / w) > MAX_DOOR_ASPECT_HW:
            return False, f"door_aspect_h/w_gt_{MAX_DOOR_ASPECT_HW}"
        if MAX_DOOR_ASPECT_WH > 0 and h > 0 and (w / h) > MAX_DOOR_ASPECT_WH:
            return False, f"door_aspect_w/h_gt_{MAX_DOOR_ASPECT_WH}"
    elif class_name == rv.TRIM_CLASS_NAME:
        if w < MIN_TRIM_WIDTH_PX:  return False, f"trim_width_lt_{MIN_TRIM_WIDTH_PX}"
        if h < MIN_TRIM_HEIGHT_PX: return False, f"trim_height_lt_{MIN_TRIM_HEIGHT_PX}"
        if area < MIN_TRIM_AREA_PX2:  return False, f"trim_area_lt_{MIN_TRIM_AREA_PX2}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Отрисовка
# ---------------------------------------------------------------------------

def _color_bgr(class_name: str):
    return {
        rv.DOOR_CLASS_NAME: (255, 255, 0),
        rv.TRIM_CLASS_NAME: (255, 0, 255),
        rv.PERSON_CLASS_NAME: (0, 255, 0),
    }.get(class_name, (180, 180, 180))


def _blend_mask_contour_bgr(vis, mask, color_bgr, opacity, thickness):
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return
    layer = vis.copy()
    layer[m] = color_bgr
    cv2.addWeighted(layer, opacity, vis, 1.0 - opacity, 0, dst=vis)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, color_bgr, thickness, lineType=cv2.LINE_AA)


def _mask_centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    return (int(xs.mean()), int(ys.mean())) if len(xs) else (0, 0)


def _draw_tripwire(vis: np.ndarray, line: np.ndarray) -> None:
    cv2.line(vis, tuple(line[0].astype(int)), tuple(line[1].astype(int)), (255, 255, 255), 2)


def _annotate_frame(frame: np.ndarray, last_dets: list, frame_idx: int,
                    line: Optional[np.ndarray] = None) -> np.ndarray:
    h, w = frame.shape[:2]
    vis = frame.copy()
    if last_dets and all("mask" in d for d in last_dets):
        xyxy = np.array([d["box"] for d in last_dets], dtype=np.float32)
        conf = np.array([d["confidence"] for d in last_dets], dtype=np.float32)
        cid = np.array([d["class_id"] for d in last_dets], dtype=int)
        masks = np.stack([d["mask"] for d in last_dets])
        dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cid, mask=masks)
        vis = sv.MaskAnnotator(opacity=LIVE_MASK_OPACITY,
                               color_lookup=sv.ColorLookup.CLASS).annotate(vis, dets)
        vis = sv.PolygonAnnotator(thickness=LIVE_POLYGON_THICKNESS,
                                  color_lookup=sv.ColorLookup.CLASS).annotate(vis, dets)
        labels = [f"{d['class_name']} {d['confidence']:.2f}" for d in last_dets]
        vis = sv.LabelAnnotator(text_scale=0.5, text_padding=4).annotate(vis, dets, labels)
    else:
        for d in last_dets:
            rv.draw_box_with_label(vis, d["box"],
                                   f"{d['class_name']} {d['confidence']:.2f}",
                                   _color_bgr(d["class_name"]), thickness=2)
    if SHOW_TRIPWIRE and line is not None:
        _draw_tripwire(vis, line)
    cv2.putText(vis, f"frame {frame_idx} | dets {len(last_dets)}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

def _build_seg_model(checkpoint_path: str, num_classes: int) -> RFDETRSegMedium:
    model = RFDETRSegMedium(pretrain_weights=checkpoint_path, num_classes=num_classes)
    try:
        model.optimize_for_inference()
    except Exception:
        pass
    return model


def _load_frame_detections_seg(model, frame, class_names: list[str]) -> list:
    detections = model.predict(rv.frame_to_model_rgb(frame), threshold=rv.CONF_THRESHOLD)
    if detections is None or len(detections.xyxy) == 0:
        return []
    has_mask = detections.mask is not None
    result = []
    for i, (xyxy, confidence, class_id) in enumerate(
            zip(detections.xyxy, detections.confidence, detections.class_id)
    ):
        class_id = int(class_id)
        item = {
            "class_id": class_id,
            "class_name": class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}",
            "confidence": float(confidence),
            "box": np.array(xyxy, dtype=np.float32),
        }
        if has_mask:
            item["mask"] = np.asarray(detections.mask[i], dtype=bool)
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Логи / утилиты прогона
# ---------------------------------------------------------------------------

def _new_run_dir() -> Path:
    LIVE_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = LIVE_RUN_ROOT / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _det_stats(last_dets: list) -> dict:
    counts = dict(Counter(d["class_name"] for d in last_dets))
    objects = [{"class": d["class_name"], "conf": round(float(d["confidence"]), 4),
                "xyxy": [round(float(x), 1) for x in d["box"]]} for d in last_dets]
    return {"counts": counts, "objects": objects}


def _write_run_info(path: Path, src, extra: dict) -> None:
    lines = [
                f"started_utc={datetime.utcnow().isoformat()}Z",
                f"source={src!r}",
                f"LOG_EVERY_N_FRAMES={LOG_EVERY_N_FRAMES}",
                f"SAVE_ANNOTATED_JPEG_EVERY_N={SAVE_ANNOTATED_JPEG_EVERY_N}",
                f"DETECT_EVERY_N={DETECT_EVERY_N}",
                f"EVENT_CLASS_WINDOW={EVENT_CLASS_WINDOW}",
            ] + [f"{k}={v}" for k, v in extra.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# SimpleTracker
# ---------------------------------------------------------------------------

class SimpleTracker:
    """Легковесный трекер: IoU + евклидово расстояние, без внешних зависимостей."""

    def __init__(self):
        self.tracks: list[dict] = []
        self._next_id = 0

    def update(self, enriched_dets: list, frame_idx: int,
               on_rearm: callable,
               is_new_track_latched: callable = None) -> tuple[list[dict], set[int]]:
        """
        Обновляет треки по списку обогащённых детекций.
        Возвращает (alive_tracks, updated_ids).
        on_rearm(track, hist_item)         — колбэк после обновления/создания трека.
        is_new_track_latched(det) -> bool  — если True, новый трек сразу counted_up=True
                                             (подавление дублей при «моргании» детектора).
        """
        updated_ids: set[int] = set()

        for det in enriched_dets:
            obj_box = det["box"]
            obj_c = rv.center(obj_box)
            tracking_grp = det["tracking_group"]

            best_tr, best_score = None, -1e18
            for tr in self.tracks:
                if tr["tracking_group"] != tracking_grp:
                    continue
                dist = float(np.linalg.norm(obj_c - tr["centers"][-1]))
                if dist > rv.TRACK_DISTANCE:
                    continue
                score = rv.iou_xyxy(obj_box, tr["box"]) * 1000.0 - dist
                if score > best_score:
                    best_score, best_tr = score, tr

            hist_item = self._make_hist_item(det, frame_idx)

            if best_tr is not None:
                self._update_existing(best_tr, det, obj_c, hist_item)
                on_rearm(best_tr, hist_item)
                updated_ids.add(best_tr["id"])
            else:
                latched = bool(is_new_track_latched and is_new_track_latched(det))
                new_tr = self._create_track(det, obj_c, hist_item, counted_up=latched)
                new_tr["latched_on_create"] = latched
                on_rearm(new_tr, hist_item)
                self.tracks.append(new_tr)
                updated_ids.add(new_tr["id"])

        # пометить потерянные
        for tr in self.tracks:
            if tr["id"] not in updated_ids:
                tr["lost"] += 1
                tr["updated_this_frame"] = False

        self.tracks = [t for t in self.tracks if t["lost"] <= rv.MAX_LOST]
        return self.tracks, updated_ids

    # ------------------------------------------------------------------
    def _make_hist_item(self, det: dict, frame_idx: int) -> dict:
        item = {
            "frame_id": frame_idx,
            "box": det["box"].copy(),
            "confidence": det["confidence"],
            "class_name": det["class_name"],
            "person_info": det.get("person_info"),
        }
        if "mask" in det:
            item["mask"] = det["mask"].copy()
        return item

    def _update_existing(self, tr: dict, det: dict, obj_c: np.ndarray, hist_item: dict):
        tr["box"] = det["box"].copy()
        tr["confidence"] = det["confidence"]
        tr["class_name_current"] = det["class_name"]
        tr["person_info_current"] = det.get("person_info")
        tr["lost"] = 0
        tr["updated_this_frame"] = True
        tr["centers"].append(obj_c)
        if len(tr["centers"]) > rv.TRACK_HISTORY:
            tr["centers"].pop(0)
        tr["history"].append(hist_item)
        if len(tr["history"]) > rv.TRACK_HISTORY:
            tr["history"].pop(0)

    def _create_track(self, det: dict, obj_c: np.ndarray, hist_item: dict,
                      counted_up: bool = False) -> dict:
        tr = {
            "id": self._next_id,
            "tracking_group": det["tracking_group"],
            "class_name_current": det["class_name"],
            "box": det["box"].copy(),
            "confidence": det["confidence"],
            "person_info_current": det.get("person_info"),
            "lost": 0,
            "counted_up": counted_up,
            "updated_this_frame": True,
            "tripwire_leave_streak": 0,
            "centers": [obj_c],
            "history": [hist_item],
            "logged_rejections": set(),
        }
        self._next_id += 1
        return tr


# ---------------------------------------------------------------------------
# LiveEventProcessor
# ---------------------------------------------------------------------------

class LiveEventProcessor:
    def __init__(self, src, run_dir: Path, log_fn):
        self.src = src
        self.run_dir = run_dir
        self.log = log_fn
        self.source_name = Path(str(src)).name if isinstance(src, str) else f"camera_{src}"
        self.source_stem = _source_stem(src)
        self.events_dir = run_dir / self.source_stem
        self.events_dir.mkdir(parents=True, exist_ok=True)

        self.line: Optional[np.ndarray] = None
        self.bottom_side_sign: float = 1.0

        self.tracker = SimpleTracker()
        self.events_count = 0
        self.rejected_count = 0

        # состояние последнего события (door/trim)
        self._last_event_frame: Optional[int] = None
        self._last_event_center: Optional[np.ndarray] = None
        self._last_event_box: Optional[np.ndarray] = None
        self._last_event_track_id: Optional[int] = None

        # история person для overlap-проверки
        self._person_history: list[list] = []
        # текущий кадр (обновляется в process_frame, нужен для dedup-колбэка)
        self._current_frame_idx: int = 0

    # ------------------------------------------------------------------
    # Инициализация линии
    # ------------------------------------------------------------------

    def _ensure_line(self, frame: np.ndarray) -> None:
        if self.line is not None:
            return
        h, w = frame.shape[:2]
        self.line = rv.denorm_line(STATIC_TRIPWIRE_LINE_NORM, w, h)
        _, self.bottom_side_sign = rv.infer_top_bottom_sides(w, h, self.line)

    # ------------------------------------------------------------------
    # Геометрия / tripwire
    # ------------------------------------------------------------------

    def _signed_dist(self, point: np.ndarray) -> float:
        a, b = self.line
        line_len = float(np.linalg.norm(b - a))
        if line_len < 1e-6:
            return 0.0
        return float(rv.side_of_line(point, a, b) / line_len) * self.bottom_side_sign

    def _track_direction(self, tr: dict) -> str:
        hist = tr.get("history", [])
        boxes = [it["box"] for it in hist if it.get("box") is not None]
        if len(boxes) < 4:
            return "unknown"

        dists = [self._signed_dist(rv.center(b)) for b in boxes]
        mid = len(dists) // 2
        first_mean = sum(dists[:mid]) / mid
        last_mean = sum(dists[mid:]) / (len(dists) - mid)

        delta = last_mean - first_mean
        if abs(delta) < 3.0:
            return "unknown"

        return "down" if delta > 0 else "up"

    def _box_line_state(self, box) -> str:
        x1, y1, x2, y2 = map(float, box)
        corners = [
            np.array([x1, y1], dtype=np.float32),
            np.array([x2, y1], dtype=np.float32),
            np.array([x1, y2], dtype=np.float32),
            np.array([x2, y2], dtype=np.float32),
        ]
        dists = [self._signed_dist(p) for p in corners]
        if min(dists) > LIVE_LINE_MARGIN_PX:
            return "below"
        if max(dists) < -LIVE_LINE_MARGIN_PX:
            return "above"
        return "intersects"

    def _bbox_straddles(self, box) -> bool:
        x1, y1, x2, y2 = map(float, box)
        corners = [
            np.array([x1, y1], dtype=np.float32),
            np.array([x2, y1], dtype=np.float32),
            np.array([x1, y2], dtype=np.float32),
            np.array([x2, y2], dtype=np.float32),
        ]
        vals = [self._signed_dist(p) for p in corners]
        eps = 0.25
        return any(v > eps for v in vals) and any(v < -eps for v in vals)

    def _mask_line_overlap_px(self, mask: np.ndarray) -> int:
        if self.line is None or not np.any(mask):
            return 0
        h, w = mask.shape[:2]
        stripe = np.zeros((h, w), dtype=np.uint8)
        p0 = tuple(np.clip(self.line[0].astype(int), [0, 0], [w - 1, h - 1]))
        p1 = tuple(np.clip(self.line[1].astype(int), [0, 0], [w - 1, h - 1]))
        cv2.line(stripe, p0, p1, 255, TRIPWIRE_MASK_LINE_THICKNESS_PX, cv2.LINE_8)
        return int(np.count_nonzero(mask.astype(bool) & (stripe > 0)))

    def _object_hits_tripwire(self, hist_item: dict) -> bool:
        mask = hist_item.get("mask")
        if mask is not None:
            mk = np.asarray(mask, dtype=bool)
            return mk.any() and self._mask_line_overlap_px(mk) >= TRIPWIRE_MASK_MIN_OVERLAP_PX
        box = hist_item["box"]
        state = self._box_line_state(box)
        return state == "intersects" or self._bbox_straddles(box)

    # ------------------------------------------------------------------
    # Проверка door ≈ person
    # ------------------------------------------------------------------

    def _push_person_history(self, person_dets: list) -> None:
        self._person_history.append(person_dets or [])
        if len(self._person_history) > LINE_EVENT_PERSON_OVERLAP_HISTORY_FRAMES:
            self._person_history.pop(0)

    def _door_overlaps_person_list(self, door_box, door_mask, person_dets: list) -> tuple[bool, str]:
        if not LINE_EVENT_REJECT_DOOR_OVERLAPPING_PERSON or not person_dets:
            return False, ""
        dw, dh, darea = rv.box_wh_area(door_box)
        if darea <= 0:
            return False, ""
        door_mask_arr = np.asarray(door_mask, dtype=bool) if door_mask is not None else None
        door_mask_area = float(np.count_nonzero(door_mask_arr)) if door_mask_arr is not None else 0.0

        for p in person_dets:
            if p.get("class_name") != rv.PERSON_CLASS_NAME:
                continue
            pbox = p.get("box")
            if pbox is None:
                continue
            pw, ph, parea = rv.box_wh_area(pbox)
            if parea <= 0:
                continue
            inter = (max(0.0, min(float(door_box[2]), float(pbox[2])) - max(float(door_box[0]), float(pbox[0]))) *
                     max(0.0, min(float(door_box[3]), float(pbox[3])) - max(float(door_box[1]), float(pbox[1]))))
            if inter <= 0:
                continue
            iou = inter / (darea + parea - inter + 1e-6)
            cont_door = inter / (darea + 1e-6)
            area_ratio = darea / (parea + 1e-6)
            big_door = area_ratio > DOOR_PERSON_AREA_RATIO_MAX

            if not big_door and iou >= DOOR_PERSON_OVERLAP_IOU_THR:
                return True, f"iou={iou:.2f} area_ratio={area_ratio:.2f}"
            if cont_door >= DOOR_PERSON_OVERLAP_CONTAINMENT_THR:
                return True, f"door⊂person={cont_door:.2f}"
            if door_mask_arr is not None and "mask" in p and door_mask_area > 0:
                pmask = np.asarray(p["mask"], dtype=bool)
                if pmask.shape == door_mask_arr.shape:
                    ratio = float(np.count_nonzero(door_mask_arr & pmask)) / door_mask_area
                    if ratio >= DOOR_PERSON_OVERLAP_MASK_RATIO_THR:
                        return True, f"mask_ratio={ratio:.2f}"
        return False, ""

    def _door_overlaps_any_recent_person(self, door_box, door_mask) -> tuple[bool, str]:
        for i, pdets in enumerate(self._person_history):
            ov, reason = self._door_overlaps_person_list(door_box, door_mask, pdets)
            if ov:
                return True, f"{reason} | person_age={len(self._person_history) - 1 - i}f"
        return False, ""

    # ------------------------------------------------------------------
    # Статичность объекта
    # ------------------------------------------------------------------

    def _track_is_static(self, tr: dict) -> tuple[bool, str]:
        if not LINE_EVENT_REQUIRE_MOTION:
            return False, ""
        hist = tr.get("history", [])
        if len(hist) < EVENT_MOTION_LOOKBACK_FRAMES:
            return False, ""
        boxes = [it["box"] for it in hist[-EVENT_MOTION_LOOKBACK_FRAMES:] if it.get("box") is not None]
        if len(boxes) < 2:
            return False, ""
        c0, w0, h0 = rv.center(boxes[0]), *rv.box_wh_area(boxes[0])[:2]
        scale = max(1.0, float(max(w0, h0)))
        max_ct = max_sd = 0.0
        for b in boxes[1:]:
            c = rv.center(b)
            w, h, _ = rv.box_wh_area(b)
            max_ct = max(max_ct, float(np.linalg.norm(c - c0)))
            max_sd = max(max_sd, abs(float(w) - float(w0)), abs(float(h) - float(h0)))
        ct_static = max_ct < EVENT_STATIC_CENTER_TRAVEL_PX or max_ct / scale < EVENT_STATIC_CENTER_TRAVEL_REL
        sd_static = max_sd < EVENT_STATIC_SIZE_DELTA_PX or max_sd / scale < EVENT_STATIC_SIZE_DELTA_REL
        if not (ct_static and sd_static):
            return False, ""
        return True, f"travel={max_ct:.1f}px size_delta={max_sd:.1f}px"

    # ------------------------------------------------------------------
    # Дедупликация / фрагменты
    # ------------------------------------------------------------------

    def _is_duplicate_event(self, frame_idx: int, obj_class: str, obj_box, track_id: int) -> bool:
        if not DOOR_LINE_EVENT_DEDUP_ENABLED:
            return False
        if obj_class not in (rv.DOOR_CLASS_NAME, rv.TRIM_CLASS_NAME):
            return False
        if self._last_event_frame is None:
            return False
        if frame_idx - self._last_event_frame > DOOR_LINE_EVENT_DEDUP_FRAMES:
            return False
        if float(np.linalg.norm(rv.center(obj_box) - self._last_event_center)) >= DOOR_LINE_EVENT_DEDUP_CENTER_DIST_PX:
            return False
        if self._last_event_track_id is not None and int(track_id) == int(self._last_event_track_id):
            return True
        if self._last_event_box is None:
            return False
        return float(rv.iou_xyxy(self._last_event_box, obj_box)) >= DOOR_LINE_EVENT_DEDUP_MIN_IOU_DIFFERENT_TRACK

    def _new_track_in_dedup_zone(self, det: dict) -> bool:
        if not DOOR_LINE_EVENT_DEDUP_ENABLED:
            return False

        if det.get("class_name") not in (rv.DOOR_CLASS_NAME, rv.TRIM_CLASS_NAME):
            return False

        if self._last_event_frame is None or self._last_event_center is None:
            return False

        elapsed = self._current_frame_idx - self._last_event_frame
        if elapsed < 0 or elapsed > DOOR_LINE_EVENT_DEDUP_FRAMES:
            return False

        hist_item = {
            "box": det["box"],
            "mask": det.get("mask"),
        }

        if not self._object_hits_tripwire(hist_item):
            return False

        obj_box = det["box"]
        dist = float(np.linalg.norm(rv.center(obj_box) - self._last_event_center))

        if dist >= DOOR_LINE_EVENT_DEDUP_CENTER_DIST_PX:
            return False

        if self._last_event_box is None:
            return True

        iou = float(rv.iou_xyxy(self._last_event_box, obj_box))
        return iou >= DOOR_LINE_EVENT_DEDUP_MIN_IOU_DIFFERENT_TRACK

    def _is_fragment_of_latched(self, tr: dict, frame_idx: int) -> Optional[dict]:
        if not LINE_EVENT_SUPPRESS_FRAGMENT_OF_LATCHED_TRACK:
            return None
        if tr.get("tracking_group") != "door_trim_group":
            return None
        if self._last_event_frame is None or self._last_event_track_id is None:
            return None
        if frame_idx - self._last_event_frame > LINE_EVENT_FRAGMENT_RECENT_FRAMES:
            return None
        if int(self._last_event_track_id) == int(tr["id"]):
            return None
        latched = next((t for t in self.tracker.tracks
                        if int(t["id"]) == int(self._last_event_track_id)), None)
        if latched is None or not latched.get("updated_this_frame") or not latched.get("counted_up"):
            return None
        x1a, y1a, x2a, y2a = map(float, tr["box"])
        x1b, y1b, x2b, y2b = map(float, latched["box"])
        aA = max(0.0, x2a - x1a) * max(0.0, y2a - y1a)
        aB = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
        if aA <= 0 or aB <= 0:
            return None
        inter = max(0.0, min(x2a, x2b) - max(x1a, x1b)) * max(0.0, min(y2a, y2b) - max(y1a, y1b))
        if inter <= 0:
            return None
        iou = inter / (aA + aB - inter)
        containment = inter / min(aA, aB)
        return latched if (
                    iou >= LINE_EVENT_FRAGMENT_IOU_THR or containment >= LINE_EVENT_FRAGMENT_CONTAINMENT_THR) else None

    def _record_event_location(self, frame_idx: int, obj_class: str, obj_box, track_id: int) -> None:
        if obj_class in (rv.DOOR_CLASS_NAME, rv.TRIM_CLASS_NAME):
            self._last_event_frame = frame_idx
            self._last_event_center = np.asarray(rv.center(obj_box), dtype=np.float32).copy()
            self._last_event_box = np.asarray(obj_box, dtype=np.float32).copy()
            self._last_event_track_id = int(track_id)

    # ------------------------------------------------------------------
    # Класс события по окну истории
    # ------------------------------------------------------------------

    def _event_class(self, track: dict) -> str:
        hist = track.get("history", [])
        if not hist:
            return track.get("class_name_current", "unknown")
        recent = hist[-EVENT_CLASS_WINDOW:]
        cnt = Counter(it["class_name"] for it in recent if "class_name" in it)
        if not cnt:
            return track.get("class_name_current", "unknown")
        best_cnt = max(cnt.values())
        cands = {c for c, v in cnt.items() if v == best_cnt}
        for it in reversed(recent):
            if it.get("class_name") in cands:
                return it["class_name"]
        return recent[-1].get("class_name", "unknown")

    # ------------------------------------------------------------------
    # Rearm tripwire
    # ------------------------------------------------------------------

    def _rearm_callback(self, tr: dict, hist_item: dict) -> None:
        hit = self._object_hits_tripwire(hist_item)
        tr["tripwire_leave_streak"] = 0 if hit else tr.get("tripwire_leave_streak", 0) + 1
        if LINE_EVENT_SIMPLE_FIRST_HIT_PER_TRACK:
            return
        if tr["tripwire_leave_streak"] >= LINE_TRIPWIRE_REARM_FRAMES:
            tr["counted_up"] = False

    # ------------------------------------------------------------------
    # Проверка готовности к событию (одна функция вместо цепочки if-continue)
    # ------------------------------------------------------------------

    def _reject_reason(self, tr: dict, frame_idx: int) -> Optional[str]:
        """Возвращает причину отказа или None — событие выдаём."""
        hist = tr["history"]
        last_hit = self._object_hits_tripwire(hist[-1])
        prev_hit = len(hist) >= 2 and self._object_hits_tripwire(hist[-2])
        latched = tr.get("counted_up", False)
        fresh = len(hist) == 1 or not prev_hit

        if latched:
            if tr.get("latched_on_create"):
                return "latched_on_create_dedup"
            return "latched"
        if len(hist) < LIVE_MIN_TRACK_HISTORY:
            return f"hist_lt_{LIVE_MIN_TRACK_HISTORY}"
        if not last_hit:
            return "no_line_hit"
        if not LINE_EVENT_SIMPLE_FIRST_HIT_PER_TRACK and not fresh:
            return "still_on_line"

        obj_class = self._event_class(tr)

        if obj_class not in EVENT_PRIMARY_CLASSES:
            return f"class_not_primary:{obj_class}"

        ok, size_reason = _is_valid_object_size(obj_class, tr["box"])
        if not ok:
            return f"small:{size_reason}"

        if len(hist) < EVENT_MIN_TRACK_HISTORY:
            return f"hist_lt_event:{EVENT_MIN_TRACK_HISTORY}"

        if EVENT_MIN_MEAN_CONFIDENCE > 0.0:
            confs = [float(it.get("confidence", 0.0)) for it in hist[-EVENT_CLASS_WINDOW:]
                     if it.get("class_name") == obj_class]
            if confs and sum(confs) / len(confs) < EVENT_MIN_MEAN_CONFIDENCE:
                return "low_mean_conf"

        last_hist = hist[-1]
        mk = last_hist.get("mask")
        if mk is not None:
            mk_arr = np.asarray(mk, dtype=bool)
            if mk_arr.any():
                ow, oh, _ = rv.box_wh_area(tr["box"])
                overlap_px = self._mask_line_overlap_px(mk_arr)
                mask_area = int(np.count_nonzero(mk_arr))
                fill_ratio = mask_area / max(1.0, float(ow) * float(oh))
                overlap_per_w = overlap_px / max(1.0, float(ow))
                if (overlap_px < EVENT_MIN_MASK_OVERLAP_PX or
                        overlap_per_w < EVENT_MIN_MASK_OVERLAP_PER_BBOX_W or
                        fill_ratio < EVENT_MIN_MASK_FILL_RATIO):
                    return f"weak_mask_overlap:{overlap_px}px"

        if self._is_duplicate_event(frame_idx, obj_class, tr["box"], tr["id"]):
            return "dedup_line"

        if self._is_fragment_of_latched(tr, frame_idx) is not None:
            return "fragment_of_latched"

        if obj_class == rv.DOOR_CLASS_NAME:
            ov, reason = self._door_overlaps_any_recent_person(tr["box"], last_hist.get("mask"))
            if ov:
                return f"door≈person:{reason}"

        is_static, static_reason = self._track_is_static(tr)
        if is_static:
            return f"static:{static_reason}"

        return None  # всё ок

    # ------------------------------------------------------------------
    # Обогащение детекций перед трекингом
    # ------------------------------------------------------------------

    def _enrich_dets(self, primary_dets: list, person_dets: list,
                     frame: np.ndarray, frame_idx: int) -> list:
        result = []
        for det in primary_dets:
            cls_name = det["class_name"]
            conf = float(det.get("confidence", 0.0))
            min_conf = MIN_DOOR_CONFIDENCE if cls_name == rv.DOOR_CLASS_NAME else \
                MIN_TRIM_CONFIDENCE if cls_name == rv.TRIM_CLASS_NAME else 0.0

            if min_conf > 0 and conf < min_conf:
                if LIVE_DEBUG_EVENT_DECISIONS:
                    self.log(f"[SKIP LOW CONF] {self.source_name} | frame={frame_idx} | "
                             f"class={cls_name} conf={conf:.2f}<{min_conf:.2f}")
                continue

            if cls_name == rv.DOOR_CLASS_NAME:
                ov, reason = self._door_overlaps_any_recent_person(det["box"], det.get("mask"))
                if ov:
                    if LIVE_DEBUG_EVENT_DECISIONS:
                        self.log(f"[SKIP DOOR≈PERSON] {self.source_name} | frame={frame_idx} | {reason}")
                    continue

            ok, inv_reason = _is_valid_object_size(cls_name, det["box"])
            if FILTER_SMALL_OBJECTS_BEFORE_TRACKING and not ok:
                if LIVE_DEBUG_EVENT_DECISIONS:
                    self.log(f"[SKIP SMALL] {self.source_name} | frame={frame_idx} | {inv_reason}")
                continue

            det2 = det.copy()
            det2["tracking_group"] = (
                "door_trim_group" if cls_name in {rv.DOOR_CLASS_NAME, rv.TRIM_CLASS_NAME} else cls_name
            )
            det2["person_info"] = (
                self._find_person_for_det(det, person_dets, frame.shape)
                if LINE_EVENT_USE_PERSON_ASSOCIATION else None
            )
            result.append(det2)
        return result

    def _find_person_for_det(self, det: dict, person_dets: list, frame_shape) -> Optional[dict]:
        best = rv.find_best_person_for_object(det, person_dets, frame_shape)
        if best is None:
            obj_box = det["box"]
            obj_c = rv.center(obj_box)
            ow, oh, _ = rv.box_wh_area(obj_box)
            obj_diag = float(np.hypot(ow, oh))
            fh, fw = frame_shape[:2]
            roi = rv.expand_box(obj_box, fw, fh, LIVE_RELAXED_ROI_EXPAND_X, LIVE_RELAXED_ROI_EXPAND_Y)
            rx1, ry1, rx2, ry2 = roi
            best_score = -1e18
            for p in person_dets:
                pc = rv.center(p["box"])
                if not (rx1 <= pc[0] <= rx2 and ry1 <= pc[1] <= ry2):
                    continue
                dist = float(np.linalg.norm(pc - obj_c))
                if dist > obj_diag * LIVE_RELAXED_MAX_PERSON_CENTER_DIST_FACTOR:
                    continue
                pw, ph, parea = rv.box_wh_area(p["box"])
                score = p["confidence"] * 1000 + parea * 0.01 - dist * 12 + rv.iou_xyxy(roi, p["box"]) * 300
                if score > best_score:
                    best_score = score
                    best = {"box": p["box"].copy(), "confidence": p["confidence"],
                            "w": int(pw), "h": int(ph), "area": int(parea),
                            "center_dist": dist, "relaxed": True}
        if best is not None and person_dets:
            out = dict(best)
            best_iou, best_mask = 0.3, None
            for pd in person_dets:
                if pd.get("class_name") != rv.PERSON_CLASS_NAME or "mask" not in pd:
                    continue
                iou = rv.iou_xyxy(best["box"], pd["box"])
                if iou > best_iou:
                    best_iou, best_mask = iou, pd["mask"]
            if best_mask is not None:
                out["mask"] = np.asarray(best_mask, dtype=bool).copy()
            return out
        return best

    # ------------------------------------------------------------------
    # Отрисовка событийных кадров
    # ------------------------------------------------------------------

    def _draw_event_frame(self, frame, tr, obj_box, obj_class, ow, oh,
                          person_info, assoc_str, obj_mask=None) -> np.ndarray:
        out = frame.copy()
        color = (255, 255, 0) if obj_class == rv.DOOR_CLASS_NAME else (255, 0, 255)
        if obj_mask is not None and np.any(obj_mask):
            _blend_mask_contour_bgr(out, obj_mask, color, LIVE_MASK_OPACITY, LIVE_POLYGON_THICKNESS)
            cx, cy = _mask_centroid(obj_mask)
            cv2.putText(out, f"{obj_class.upper()} {int(ow)}x{int(oh)} conf={tr['confidence']:.2f}",
                        (max(5, cx - 120), max(22, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        else:
            rv.draw_box_with_label(out, obj_box,
                                   f"{obj_class.upper()} {int(ow)}x{int(oh)} conf={tr['confidence']:.2f}",
                                   color, thickness=3)
        if person_info is not None:
            pm = person_info.get("mask")
            pc = (0, 255, 0)
            if pm is not None and np.any(pm):
                _blend_mask_contour_bgr(out, pm, pc, LIVE_MASK_OPACITY, LIVE_POLYGON_THICKNESS)
            else:
                rv.draw_box_with_label(out, person_info["box"],
                                       f"PERSON {person_info['w']}x{person_info['h']}", pc, thickness=3)
        cv2.putText(out, f"LINE | {assoc_str}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        _draw_tripwire(out, self.line)
        return out

    # ------------------------------------------------------------------
    # Главный метод обработки кадра
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, all_dets: list, frame_idx: int) -> np.ndarray:
        self._ensure_line(frame)
        self._current_frame_idx = frame_idx
        vis = _annotate_frame(frame, all_dets, frame_idx, self.line)

        primary_dets, person_dets = rv.split_detections(all_dets)
        self._push_person_history(person_dets)

        enriched = self._enrich_dets(primary_dets, person_dets, frame, frame_idx)
        self.tracker.update(enriched, frame_idx, self._rearm_callback,
                            is_new_track_latched=self._new_track_in_dedup_zone)

        for tr in self.tracker.tracks:
            if not tr.get("updated_this_frame"):
                continue

            reject = self._reject_reason(tr, frame_idx)
            if reject is not None:
                # логируем только первый раз для каждой категории
                key = reject.split(":")[0]
                if key not in tr.get("logged_rejections", set()):
                    tr.setdefault("logged_rejections", set()).add(key)
                    if LIVE_DEBUG_EVENT_DECISIONS:
                        self.log(f"[REJECT {key}] {self.source_name} | frame={frame_idx} | "
                                 f"track={tr['id']} | {reject}")
                # dedup_line тоже защёлкивает
                if reject == "dedup_line":
                    tr["counted_up"] = True
                continue

            # --- выдаём событие ---
            obj_class = self._event_class(tr)
            obj_box = tr["box"]
            ow, oh, _ = rv.box_wh_area(obj_box)
            obj_mask = tr["history"][-1].get("mask")
            person_info = tr["history"][-1].get("person_info") if LINE_EVENT_USE_PERSON_ASSOCIATION else None

            direction = self._track_direction(tr)
            assoc_str = f"{obj_class}+line+{direction}"

            self._record_event_location(frame_idx, obj_class, obj_box, tr["id"])
            tr["counted_up"] = True
            self.events_count += 1

            log_data = {
                "video": self.source_name,
                "frame": frame_idx,
                "direction": direction,
                "event_type": "LINE_CROSS",
                "assoc": assoc_str,
                "object_class": obj_class,
                "recent_classes": [it.get("class_name") for it in tr["history"][-EVENT_CLASS_WINDOW:]],
                "object_w": int(ow),
                "object_h": int(oh),
                "object_area": int(ow * oh),
                "line_hit": True,
            }
            if person_info is not None:
                log_data.update({"person_w": person_info["w"], "person_h": person_info["h"],
                                 "person_dist": round(person_info["center_dist"], 1)})
            self.log("[EVENT ✔] " + json.dumps(log_data, ensure_ascii=False))

            if rv.SAVE:
                out = self._draw_event_frame(frame, tr, obj_box, obj_class, ow, oh,
                                             person_info, assoc_str, obj_mask)

                fname = f"{self.source_stem}_LINE_{assoc_str}_{frame_idx:06d}.jpg"
                path = self.events_dir / fname

                if cv2.imwrite(str(path), out):
                    self.log(f"  сохранён кадр: {path}")
                vis = out

        return vis

    def annotate_frame(self, frame: np.ndarray, all_dets: list, frame_idx: int) -> np.ndarray:
        self._ensure_line(frame)
        return _annotate_frame(frame, all_dets, frame_idx, self.line)

    def describe_output(self) -> str:
        return f"События и reject-кадры: {self.events_dir}"


# ---------------------------------------------------------------------------
# PyQt UI
# ---------------------------------------------------------------------------

try:
    from PyQt5.QtCore import QEvent, Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QImage, QKeySequence, QPixmap
    from PyQt5.QtWidgets import (
        QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton,
        QShortcut, QSlider, QSpinBox, QVBoxLayout, QWidget,
    )

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

if HAS_PYQT:

    def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
        arr = np.ascontiguousarray(bgr)
        h, w = arr.shape[:2]
        # Format_BGR888 (Qt ≥ 5.14) — без cvtColor
        fmt = getattr(QImage, "Format_BGR888", None)
        if fmt is not None:
            return QImage(arr.data, w, h, arr.strides[0], fmt).copy()
        # fallback
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()


    class DetWorker(QThread):
        frame_ready = pyqtSignal(object)  # QImage
        log = pyqtSignal(str)
        finished_ok = pyqtSignal()
        video_opened = pyqtSignal(int)  # total_frames (0 = нет перемотки)
        frame_progress = pyqtSignal(int)

        def __init__(self, src, model, class_names, save_mp4, mp4_path, out_fps, run_dir):
            super().__init__()
            self.src = src
            self.model = model
            self.class_names = class_names
            self._save_mp4 = save_mp4
            self._mp4_path = mp4_path
            self._out_fps = out_fps
            self._run_dir = run_dir
            self._stop = False
            self._paused = False
            self.video_writer: Optional[cv2.VideoWriter] = None
            self._seek_lock = threading.Lock()
            self._seek_target: Optional[int] = None
            self._total_frames = 0

        def request_stop(self) -> None:
            self._stop = True

        def request_pause(self) -> None:
            self._paused = True

        def request_resume(self) -> None:
            self._paused = False

        def toggle_pause(self) -> bool:
            self._paused = not self._paused
            return self._paused

        def request_seek_frame(self, f: int) -> None:
            with self._seek_lock:
                self._seek_target = int(f)

        def _pop_seek(self) -> Optional[int]:
            with self._seek_lock:
                t, self._seek_target = self._seek_target, None
            return t

        def _apply_seek(self, cap, current_src, run_dir) -> tuple[Optional[int], LiveEventProcessor]:
            t = self._pop_seek()
            if t is None or not _is_video_file(current_src):
                return None, None
            t = max(1, min(t, self._total_frames or t))
            cap.set(cv2.CAP_PROP_POS_FRAMES, t - 1)
            self.log.emit(f"Перемотка → кадр {t}" +
                          (f" / {self._total_frames}" if self._total_frames else ""))
            return t, LiveEventProcessor(current_src, run_dir, self.log.emit)

        def run(self) -> None:
            _write_run_info(self._run_dir / "run_info.txt", self.src,
                            {"out_fps": self._out_fps,
                             "model": "RFDETRSegMedium",
                             "seg_checkpoint": SEG_CHECKPOINT_PATH})
            jsonl_fp = (open(self._run_dir / "detections.jsonl", "w", encoding="utf-8")
                        if LOG_EVERY_N_FRAMES > 0 else None)
            self.log.emit(f"Папка прогона: {self._run_dir}")
            sources = _expand_sources(self.src)
            if not sources:
                self.log.emit(f"Не найдены видео: {self.src!r}")
                self.finished_ok.emit()
                return
            if len(sources) > 1:
                self.log.emit(f"Автопереключение: {len(sources)} файлов.")

            total_events = 0
            for s_idx, current_src in enumerate(sources, 1):
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
                if out_fps < 1 or out_fps > 120:
                    out_fps = self._out_fps
                self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                self.video_opened.emit(
                    self._total_frames if (_is_video_file(current_src) and self._total_frames > 1) else 0
                )

                self.log.emit(f"[{s_idx}/{len(sources)}] {_source_label(current_src)}")
                proc = LiveEventProcessor(current_src, self._run_dir, self.log.emit)
                last_dets: list = []
                frame_idx = 0

                while not self._stop:
                    # --- пауза ---
                    while self._paused and not self._stop:
                        # во время паузы разрешаем перемотку
                        seek_t, new_proc = self._apply_seek(cap, current_src, self._run_dir)
                        if new_proc is not None:
                            proc = new_proc
                            last_dets = []
                            frame_idx = seek_t
                        self.msleep(50)
                        continue

                    seek_t, new_proc = self._apply_seek(cap, current_src, self._run_dir)
                    if new_proc is not None:
                        proc = new_proc
                        last_dets = []
                        frame_idx = seek_t

                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if LOOP_VIDEO_FILE and len(sources) == 1 and _is_video_file(current_src):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_idx = 0
                            continue
                        self.log.emit(f"Конец: {_source_label(current_src)}")
                        break

                    frame_idx += 1
                    detect_now = DETECT_EVERY_N <= 1 or (frame_idx % DETECT_EVERY_N == 1)
                    if detect_now:
                        last_dets = _load_frame_detections_seg(self.model, frame, self.class_names)

                    vis = (proc.process_frame(frame, last_dets, frame_idx) if detect_now
                           else proc.annotate_frame(frame, last_dets, frame_idx))
                    h, w = vis.shape[:2]

                    if self._save_mp4 and self.video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out_path = _recording_mp4_path(
                            current_src, sources, proc.events_dir, self._mp4_path
                        )
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        self.video_writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (w, h))

                    if self.video_writer and self.video_writer.isOpened():
                        self.video_writer.write(vis)

                    if LOG_EVERY_N_FRAMES > 0 and frame_idx % LOG_EVERY_N_FRAMES == 0:
                        stats = _det_stats(last_dets)
                        self.log.emit(
                            f"[кадр {frame_idx}] всего {len(last_dets)} | по классам: {stats['counts']}"
                        )
                        if jsonl_fp:
                            jsonl_fp.write(json.dumps(
                                {"video": _source_label(current_src), "frame": frame_idx, **stats},
                                ensure_ascii=False) + "\n")
                            jsonl_fp.flush()

                    if SAVE_ANNOTATED_JPEG_EVERY_N > 0 and frame_idx % SAVE_ANNOTATED_JPEG_EVERY_N == 0:
                        p = proc.events_dir / f"annotated_{frame_idx:06d}.jpg"
                        if cv2.imwrite(str(p), vis):
                            self.log.emit(f"  сохранён: {p}")

                    # передаём QImage — меньше работы в UI-потоке
                    self.frame_ready.emit(_bgr_to_qimage(vis))
                    if wait_ms > 0:
                        self.msleep(wait_ms)
                    if self._total_frames > 1 and _is_video_file(current_src):
                        self.frame_progress.emit(frame_idx)

                cap.release()
                total_events += proc.events_count
                self.log.emit(f"Итог {_source_label(current_src)}: событий {proc.events_count}")
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None

            if jsonl_fp:
                jsonl_fp.close()
            self.log.emit(f"Итого событий: {total_events}")
            self.finished_ok.emit()


    class MainWindow(QMainWindow):
        def __init__(self, worker: DetWorker):
            super().__init__()
            self.setWindowTitle(WINDOW_TITLE)
            self._worker = worker
            self._video_total = 0
            self._pause_sync = False
            self._last_pixmap_size = (0, 0)
            self._is_paused = False

            central = QWidget()
            vbox = QVBoxLayout(central)
            vbox.setSpacing(4)
            vbox.setContentsMargins(4, 4, 4, 4)

            # --- видео ---
            self._label = QLabel()
            self._label.setAlignment(Qt.AlignCenter)
            self._label.setMinimumSize(960, 540)
            self._label.setStyleSheet("background-color:#1a1a1a;")
            vbox.addWidget(self._label, 1)

            # --- строка 1: метка + ползунок + спинбокс ---
            row1 = QHBoxLayout()
            row1.setSpacing(4)
            self._lbl_seek = QLabel("Перемотка: ожидание файла…")
            self._lbl_seek.setFixedWidth(220)
            self._slider = QSlider(Qt.Horizontal)
            self._slider.setRange(1, 1)
            self._slider.setEnabled(False)
            self._spin = QSpinBox()
            self._spin.setRange(1, 1)
            self._spin.setEnabled(False)
            self._spin.setFixedWidth(80)
            row1.addWidget(self._lbl_seek)
            row1.addWidget(self._slider, 1)
            row1.addWidget(self._spin)
            vbox.addLayout(row1)

            # --- строка 2: кнопки шага + середина + пауза ---
            row2 = QHBoxLayout()
            row2.setSpacing(3)

            def _step_btn(label: str, delta: int) -> QPushButton:
                b = QPushButton(label)
                b.setFixedWidth(54)
                b.setEnabled(False)
                b.clicked.connect(lambda: self._seek_relative(delta))
                return b

            self._step_btns: list[QPushButton] = []
            for label, delta in [("-100", -100), ("-10", -10), ("-1", -1),
                                 ("+1", 1), ("+10", 10), ("+100", 100)]:
                b = _step_btn(label, delta)
                self._step_btns.append(b)
                row2.addWidget(b)

            row2.addSpacing(8)

            self._btn_mid = QPushButton("Середина")
            self._btn_mid.setFixedWidth(80)
            self._btn_mid.setEnabled(False)
            self._btn_mid.clicked.connect(self._on_mid)
            row2.addWidget(self._btn_mid)

            row2.addSpacing(8)

            self._btn_pause = QPushButton("⏸  Пауза")
            self._btn_pause.setFixedWidth(100)
            self._btn_pause.setCheckable(True)
            self._btn_pause.setEnabled(True)
            self._btn_pause.setStyleSheet(
                "QPushButton { background:#3a3a3a; color:white; border-radius:4px; padding:3px 6px; }"
                "QPushButton:checked { background:#b85000; }"
                "QPushButton:hover   { background:#555; }"
            )
            self._btn_pause.clicked.connect(self._on_pause_toggle)
            row2.addWidget(self._btn_pause)

            row2.addStretch(1)
            vbox.addLayout(row2)

            self.setCentralWidget(central)

            # --- сигналы воркера ---
            worker.frame_ready.connect(self._on_frame)
            worker.log.connect(lambda m: print(m, flush=True))
            worker.video_opened.connect(self._on_video_opened)
            worker.frame_progress.connect(self._on_progress)

            # --- сигналы слайдера / спина ---
            self._slider.sliderPressed.connect(lambda: setattr(self, "_pause_sync", True))
            self._slider.sliderReleased.connect(self._on_slider_released)
            self._spin.editingFinished.connect(self._on_spin_edited)
            self._spin.lineEdit().installEventFilter(self)

            # --- горячие клавиши ---
            for key, fn in [
                ("Q", self.close),
                ("Escape", self.close),
                ("M", self._on_mid),
                ("Space", self._on_pause_toggle),
                ("Left", lambda: self._seek_relative(-1)),
                ("Right", lambda: self._seek_relative(1)),
            ]:
                QShortcut(QKeySequence(key), self, activated=fn)

        def eventFilter(self, obj, event):
            if obj is self._spin.lineEdit():
                if event.type() == QEvent.FocusIn:
                    self._pause_sync = True
                elif event.type() == QEvent.FocusOut:
                    self._pause_sync = False
            return super().eventFilter(obj, event)

        def _on_video_opened(self, total: int) -> None:
            self._video_total = total
            enabled = total > 1
            self._slider.setEnabled(enabled)
            self._spin.setEnabled(enabled)
            self._btn_mid.setEnabled(enabled)
            for b in self._step_btns:
                b.setEnabled(enabled)
            if enabled:
                self._slider.setRange(1, total)
                self._spin.setRange(1, total)
                self._lbl_seek.setText(f"Кадр 1…{total}")
            else:
                self._lbl_seek.setText("Перемотка недоступна (поток/камера)")

        def _on_progress(self, f: int) -> None:
            if self._video_total <= 1 or self._pause_sync:
                return
            v = max(1, min(f, self._video_total))
            self._slider.blockSignals(True);
            self._spin.blockSignals(True)
            self._slider.setValue(v);
            self._spin.setValue(v)
            self._slider.blockSignals(False);
            self._spin.blockSignals(False)

        def _on_slider_released(self) -> None:
            v = self._slider.value()
            self._spin.setValue(v)
            self._worker.request_seek_frame(v)
            self._pause_sync = False

        def _on_spin_edited(self) -> None:
            if not self._spin.isEnabled():
                return
            v = max(1, min(self._spin.value(), self._video_total))
            self._slider.setValue(v)
            self._worker.request_seek_frame(v)
            self._spin.clearFocus()

        def _on_mid(self) -> None:
            if self._video_total <= 1:
                return
            mid = max(1, self._video_total // 2)
            self._slider.setValue(mid);
            self._spin.setValue(mid)
            self._worker.request_seek_frame(mid)

        def _seek_relative(self, delta: int) -> None:
            """Перемотка на ±N кадров от текущей позиции."""
            if self._video_total <= 1:
                return
            cur = self._spin.value()
            tgt = max(1, min(cur + delta, self._video_total))
            self._slider.setValue(tgt)
            self._spin.setValue(tgt)
            self._worker.request_seek_frame(tgt)

        def _on_pause_toggle(self) -> None:
            self._is_paused = self._worker.toggle_pause()
            if self._is_paused:
                self._btn_pause.setText("▶  Продолжить")
                self._btn_pause.setChecked(True)
            else:
                self._btn_pause.setText("⏸  Пауза")
                self._btn_pause.setChecked(False)

        def _on_frame(self, qimg: QImage) -> None:
            pix = QPixmap.fromImage(qimg)
            label_sz = self._label.size()
            if (label_sz.width(), label_sz.height()) != self._last_pixmap_size:
                self._last_pixmap_size = (label_sz.width(), label_sz.height())
            self._label.setPixmap(
                pix.scaled(label_sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            self._last_pixmap_size = (0, 0)

        def closeEvent(self, event) -> None:
            self._worker.request_stop()
            self._worker.wait(15000)
            if self._worker.video_writer:
                self._worker.video_writer.release()
            event.accept()


# ---------------------------------------------------------------------------
# Headless режим
# ---------------------------------------------------------------------------

def run_headless(model, class_names, src) -> int:
    run_dir = _new_run_dir()
    _write_run_info(run_dir / "run_info.txt", src,
                    {"mode": "headless_seg", "model": "RFDETRSegMedium",
                     "seg_checkpoint": SEG_CHECKPOINT_PATH})
    jsonl_fp = (open(run_dir / "detections.jsonl", "w", encoding="utf-8")
                if LOG_EVERY_N_FRAMES > 0 else None)
    print(f"Логи: {run_dir}", flush=True)

    sources = _expand_sources(src)
    if not sources:
        print(f"Не найдены видео: {src!r}", flush=True)
        return 1

    rc = 0
    total_events = 0
    try:
        for s_idx, current_src in enumerate(sources, 1):
            cap = _try_open_video_capture(current_src)
            if cap is None:
                print(_format_capture_open_error(current_src), flush=True);
                rc = 1;
                continue

            out_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            if out_fps < 1 or out_fps > 120: out_fps = 25.0
            wait_ms = _playback_wait_ms(cap, current_src)
            proc = LiveEventProcessor(current_src, run_dir, lambda m: print(m, flush=True))
            writer = None
            last_dets: list = []
            frame_idx = 0

            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if LOOP_VIDEO_FILE and len(sources) == 1 and _is_video_file(current_src):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0);
                        frame_idx = 0;
                        continue
                    break
                frame_idx += 1
                detect_now = DETECT_EVERY_N <= 1 or (frame_idx % DETECT_EVERY_N == 1)
                if detect_now:
                    last_dets = _load_frame_detections_seg(model, frame, class_names)
                vis = (proc.process_frame(frame, last_dets, frame_idx) if detect_now
                       else proc.annotate_frame(frame, last_dets, frame_idx))
                h, w = vis.shape[:2]
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_path = _recording_mp4_path(
                        current_src, sources, proc.events_dir, MP4_OUTPUT
                    )
                    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (w, h))
                    if not writer.isOpened():
                        print("Не удалось открыть VideoWriter", flush=True);
                        rc = 1;
                        break
                writer.write(vis)
                if LOG_EVERY_N_FRAMES > 0 and frame_idx % LOG_EVERY_N_FRAMES == 0:
                    stats = _det_stats(last_dets)
                    print(
                        f"[кадр {frame_idx}] всего {len(last_dets)} | по классам: {stats['counts']}",
                        flush=True,
                    )
                    if jsonl_fp:
                        jsonl_fp.write(json.dumps(
                            {"video": _source_label(current_src), "frame": frame_idx, **stats},
                            ensure_ascii=False) + "\n")
                        jsonl_fp.flush()
                if SAVE_ANNOTATED_JPEG_EVERY_N > 0 and frame_idx % SAVE_ANNOTATED_JPEG_EVERY_N == 0:
                    cv2.imwrite(str(proc.events_dir / f"annotated_{frame_idx:06d}.jpg"), vis)
                if wait_ms > 0:
                    cv2.waitKey(wait_ms)

            total_events += proc.events_count
            print(f"[{s_idx}] {_source_label(current_src)}: событий {proc.events_count}", flush=True)
            cap.release()
            if writer: writer.release()
    finally:
        if jsonl_fp: jsonl_fp.close()

    print(f"Итого событий: {total_events}", flush=True)
    return rc


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> int:
    if not Path(SEG_CHECKPOINT_PATH).exists():
        print(f"Чекпоинт не найден: {SEG_CHECKPOINT_PATH}");
        return 1
    if not Path(rv.DATA_YAML_PATH).exists():
        print(f"data.yaml не найден: {rv.DATA_YAML_PATH}");
        return 1

    class_names = rv.load_class_names_from_yaml(rv.DATA_YAML_PATH)
    print("RF-DETR Seg | классы:", ", ".join(f"{i}:{n}" for i, n in enumerate(class_names)))
    model = _build_seg_model(SEG_CHECKPOINT_PATH, num_classes=len(class_names))

    if os.environ.get("RFDETR_LIVE_HEADLESS") == "1":
        return run_headless(model, class_names, LIVE_SOURCE)

    if not HAS_PYQT:
        print("Нужен PyQt5: pip install PyQt5");
        return 1

    run_dir = _new_run_dir()
    print(f"Прогон: {run_dir}", flush=True)
    app = QApplication(sys.argv)
    worker = DetWorker(LIVE_SOURCE, model, class_names, ALSO_SAVE_MP4, MP4_OUTPUT,
                       _probe_fps(LIVE_SOURCE), run_dir)
    win = MainWindow(worker)
    win.show()
    worker.start()
    return int(app.exec_())


if __name__ == "__main__":
    sys.exit(main())
