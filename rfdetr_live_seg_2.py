#!/usr/bin/env python3
"""
Эфир с RF-DETR Seg (instance segmentation): маски + контуры на кадре, логика событий как в rfdetr_live.

Источник (LIVE_SOURCE): int — камера; str — один RTSP/файл; list — несколько URL (как в camera.py).

Зависимость: pip install PyQt5 supervision

Без окна: RFDETR_LIVE_HEADLESS=1 python rfdetr_live_seg.py

Чекпоинт: SEG_CHECKPOINT_PATH (веса сегментации, не bbox-only).
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

STATIC_TRIPWIRE_LINE_NORM: list[tuple[float, float]] = [
    (0.216014, 0.494380),
    (0.768757, 0.383457),
]

try:
    import threading

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

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

# ---------- источник (выбери один) ----------
LIVE_SOURCE_URLS: list[str] = [
    # "rtsp://viewer:ViewerPass_9347X@94.41.120.115:8554/cam44",
]
RECORDINGS_CAM0_DIR = str(Path(__file__).resolve().parent / "recordings_2/cam2")
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

FILE_PLAYBACK_SLOWDOWN = 2.0
FILE_PLAYBACK_CAP_FPS = 18.0

ALSO_SAVE_MP4 = False
MP4_OUTPUT = Path(__file__).resolve().parent / "rfdetr_live_out.mp4"

LIVE_RUN_ROOT = Path(__file__).resolve().parent / "rfdetr_live_logs_2"
LOG_EVERY_N_FRAMES = 10
SAVE_ANNOTATED_JPEG_EVERY_N = 0

LIVE_DEBUG_EVENT_DECISIONS = False
LINE_TRIPWIRE_REARM_FRAMES = 15
LINE_EVENT_SIMPLE_FIRST_HIT_PER_TRACK = True
LINE_EVENT_SUPPRESS_FRAGMENT_OF_LATCHED_TRACK = True
LINE_EVENT_FRAGMENT_RECENT_FRAMES = 25
LINE_EVENT_FRAGMENT_IOU_THR = 0.55
LINE_EVENT_FRAGMENT_CONTAINMENT_THR = 0.85
LINE_EVENT_USE_PERSON_ASSOCIATION = False

TRIPWIRE_MASK_LINE_THICKNESS_PX = 4
TRIPWIRE_MASK_MIN_OVERLAP_PX = 12

DOOR_LINE_EVENT_DEDUP_FRAMES = 55   # потеря трека + повторное касание линии той же дверью
DOOR_LINE_EVENT_DEDUP_CENTER_DIST_PX = 200.0  # центры дальше — считаем другой объект
DOOR_LINE_EVENT_DEDUP_MIN_IOU_DIFFERENT_TRACK = 0.40  # ниже порог: маска bbox сильно меняется на линии
# Если track_id другой (фрагментация трека), но центр почти тот же — это та же дверь (IoU мог просесть).
DOOR_LINE_EVENT_DEDUP_STRICT_CENTER_PX = 150.0
DOOR_LINE_EVENT_DEDUP_ENABLED = True

LIVE_DEBUG_DOOR_LINE_PER_FRAME_FORCE = False
LIVE_DEBUG_DOOR_LINE_PER_FRAME = False

LIVE_MIN_TRACK_HISTORY = 1
LIVE_LINE_MARGIN_PX = 18.0
LIVE_RELAXED_ROI_EXPAND_X = 0.45
LIVE_RELAXED_ROI_EXPAND_Y = 0.28
LIVE_RELAXED_MAX_PERSON_CENTER_DIST_FACTOR = 1.55

EVENT_CLASS_WINDOW = 7

# ---------- размеры двери (обновлено для отсечения ложных срабатываний) ----------
MIN_DOOR_WIDTH_PX = 280       # было 200
MIN_DOOR_HEIGHT_PX = 440      # было 450 → снижено чтобы не отсекать дверь с h=447
MIN_DOOR_AREA_PX2 = 130000    # было 70000
MAX_DOOR_AREA_PX2 = 0
MAX_DOOR_ASPECT_HW = 3.0
MAX_DOOR_ASPECT_WH = 1.3      # было 3.0 — ширина не должна превышать высоту более чем в 1.3 раза

MIN_TRIM_WIDTH_PX = 20
MIN_TRIM_HEIGHT_PX = 80
MIN_TRIM_AREA_PX2 = 2500

FILTER_SMALL_OBJECTS_BEFORE_TRACKING = False

MIN_DOOR_CONFIDENCE = 0.30
MIN_TRIM_CONFIDENCE = 0.30

EVENT_MIN_TRACK_HISTORY = 2
EVENT_MIN_MEAN_CONFIDENCE = 0.35

EVENT_MIN_MASK_OVERLAP_PX = 40
EVENT_MIN_MASK_OVERLAP_PER_BBOX_W = 0.15
EVENT_MIN_MASK_FILL_RATIO = 0.20

EVENT_PRIMARY_CLASSES = (rv.DOOR_CLASS_NAME,)

LINE_EVENT_REJECT_DOOR_OVERLAPPING_PERSON = False
DOOR_PERSON_OVERLAP_IOU_THR = 0.45
DOOR_PERSON_OVERLAP_CONTAINMENT_THR = 0.60
DOOR_PERSON_OVERLAP_MASK_RATIO_THR = 0.55
DOOR_PERSON_AREA_RATIO_MAX = 1.5

LINE_EVENT_PERSON_OVERLAP_HISTORY_FRAMES = 5

LINE_EVENT_REQUIRE_MOTION = True
EVENT_MOTION_LOOKBACK_FRAMES = 8
EVENT_STATIC_CENTER_TRAVEL_PX = 8.0
EVENT_STATIC_SIZE_DELTA_PX = 12.0
EVENT_STATIC_CENTER_TRAVEL_REL = 0.04
EVENT_STATIC_SIZE_DELTA_REL = 0.05

# ---------- направление: минимальная разница signed_dist для подтверждения ----------
DIRECTION_MIN_DELTA = 1.0  # было 3.0 → при коротких треках (2-3 кадра) дельта меньше

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
    lines = [f"Не удалось открыть источник {src!r}."]
    if isinstance(src, int):
        lines.extend([
            "  Локальная камера по индексу:",
            "    • Есть ли устройство: ls -l /dev/video*",
            "    • Доступ: пользователь в группе «video».",
            "    • Виртуалка/WSL: используйте RTSP или видеофайл.",
        ])
    else:
        lines.extend([
            "  URL или путь:",
            "    • RTSP: проверьте сеть, учётные данные.",
            "    • Файл: существует ли путь и права на чтение.",
        ])
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


def _annotate_frame(frame: np.ndarray, last_dets: list, frame_idx: int) -> np.ndarray:
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
            scene=vis, detections=dets, labels=labels,
        )
    else:
        for d in last_dets:
            label = f"{d['class_name']} {d['confidence']:.2f}"
            rv.draw_box_with_label(vis, d["box"], label, _color_bgr(d["class_name"]), thickness=2)
    if SHOW_TRIPWIRE:
        line_px = rv.denorm_line(STATIC_TRIPWIRE_LINE_NORM, w, h)
        cv2.line(
            vis,
            tuple(line_px[0].astype(int)),
            tuple(line_px[1].astype(int)),
            (255, 255, 255), 2,
        )
    cv2.putText(
        vis, f"frame {frame_idx} | dets {len(last_dets)}",
        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
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
        if MAX_DOOR_AREA_PX2 > 0 and area > MAX_DOOR_AREA_PX2:
            return False, f"door_area_gt_{MAX_DOOR_AREA_PX2}"
        if MAX_DOOR_ASPECT_HW > 0 and w > 0 and (h / w) > MAX_DOOR_ASPECT_HW:
            return False, f"door_aspect_h/w_gt_{MAX_DOOR_ASPECT_HW}"
        if MAX_DOOR_ASPECT_WH > 0 and h > 0 and (w / h) > MAX_DOOR_ASPECT_WH:
            return False, f"door_aspect_w/h_gt_{MAX_DOOR_ASPECT_WH}"
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

        self.line = None
        self.top_side_sign_value = None
        self.bottom_side_sign_value = None

        self.tracks = []
        self.next_track_id = 0
        self.events_count = 0
        self.rejected_count = 0
        self._last_primary_line_event_frame: int | None = None
        self._last_primary_line_event_center: np.ndarray | None = None
        self._last_primary_line_event_box: np.ndarray | None = None
        self._last_primary_line_event_track_id: int | None = None
        self._last_person_dets: list = []
        self._person_dets_history: list[list] = []

    def _push_person_dets_history(self, person_dets: list) -> None:
        self._person_dets_history.append(person_dets or [])
        if len(self._person_dets_history) > LINE_EVENT_PERSON_OVERLAP_HISTORY_FRAMES:
            self._person_dets_history.pop(0)

    def _door_overlaps_any_recent_person(self, door_box, door_mask) -> tuple[bool, str]:
        if not LINE_EVENT_REJECT_DOOR_OVERLAPPING_PERSON:
            return False, ""
        n = len(self._person_dets_history)
        for i, pdets in enumerate(self._person_dets_history):
            if not pdets:
                continue
            ov, reason = self._door_overlaps_person(door_box, door_mask, pdets)
            if ov:
                age = n - 1 - i
                return True, f"{reason} | person_age={age}f"
        return False, ""

    def _track_motion_metrics(self, tr: dict) -> dict | None:
        history = tr.get("history", [])
        if len(history) < EVENT_MOTION_LOOKBACK_FRAMES:
            return None
        window = history[-EVENT_MOTION_LOOKBACK_FRAMES:]
        boxes = [item.get("box") for item in window if item.get("box") is not None]
        if len(boxes) < 2:
            return None
        first = boxes[0]
        c0 = rv.center(first)
        w0, h0, _ = rv.box_wh_area(first)
        scale = max(1.0, float(max(w0, h0)))
        max_center_travel = 0.0
        max_size_delta = 0.0
        for b in boxes[1:]:
            c = rv.center(b)
            w, h, _ = rv.box_wh_area(b)
            d = float(np.linalg.norm(c - c0))
            sd = max(abs(float(w) - float(w0)), abs(float(h) - float(h0)))
            if d > max_center_travel:
                max_center_travel = d
            if sd > max_size_delta:
                max_size_delta = sd
        return {
            "max_center_travel_px": max_center_travel,
            "max_size_delta_px": max_size_delta,
            "max_center_travel_rel": max_center_travel / scale,
            "max_size_delta_rel": max_size_delta / scale,
            "window": len(boxes),
        }

    def _track_is_static(self, tr: dict) -> tuple[bool, str]:
        if not LINE_EVENT_REQUIRE_MOTION:
            return False, ""
        m = self._track_motion_metrics(tr)
        if m is None:
            return False, ""
        ct_abs_static = m["max_center_travel_px"] < EVENT_STATIC_CENTER_TRAVEL_PX
        sd_abs_static = m["max_size_delta_px"] < EVENT_STATIC_SIZE_DELTA_PX
        ct_rel_static = m["max_center_travel_rel"] < EVENT_STATIC_CENTER_TRAVEL_REL
        sd_rel_static = m["max_size_delta_rel"] < EVENT_STATIC_SIZE_DELTA_REL
        is_static = (ct_abs_static or ct_rel_static) and (sd_abs_static or sd_rel_static)
        if not is_static:
            return False, ""
        reason = (
            f"center_travel={m['max_center_travel_px']:.1f}px "
            f"({m['max_center_travel_rel']*100:.1f}%) | "
            f"size_delta={m['max_size_delta_px']:.1f}px "
            f"({m['max_size_delta_rel']*100:.1f}%) | "
            f"window={m['window']}f"
        )
        return True, reason

    def _door_overlaps_person(self, door_box, door_mask, person_dets: list) -> tuple[bool, str]:
        if not LINE_EVENT_REJECT_DOOR_OVERLAPPING_PERSON or not person_dets:
            return False, ""
        if door_box is None:
            return False, ""
        dw, dh, darea = rv.box_wh_area(door_box)
        if darea <= 0:
            return False, ""
        door_mask_arr = np.asarray(door_mask, dtype=bool) if door_mask is not None else None
        door_mask_area = float(np.count_nonzero(door_mask_arr)) if door_mask_arr is not None else 0.0
        for pdet in person_dets:
            if pdet.get("class_name") != rv.PERSON_CLASS_NAME:
                continue
            pbox = pdet.get("box")
            if pbox is None:
                continue
            pw, ph, parea = rv.box_wh_area(pbox)
            if parea <= 0:
                continue
            x1 = max(float(door_box[0]), float(pbox[0]))
            y1 = max(float(door_box[1]), float(pbox[1]))
            x2 = min(float(door_box[2]), float(pbox[2]))
            y2 = min(float(door_box[3]), float(pbox[3]))
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if inter <= 0:
                continue
            iou = inter / (darea + parea - inter + 1e-6)
            cont_door = inter / (darea + 1e-6)
            cont_person = inter / (parea + 1e-6)
            area_ratio = darea / (parea + 1e-6)
            door_much_bigger = area_ratio > DOOR_PERSON_AREA_RATIO_MAX
            if not door_much_bigger:
                if iou >= DOOR_PERSON_OVERLAP_IOU_THR:
                    return True, f"iou={iou:.2f}>={DOOR_PERSON_OVERLAP_IOU_THR:.2f} (area_ratio={area_ratio:.2f})"
            if cont_door >= DOOR_PERSON_OVERLAP_CONTAINMENT_THR:
                return True, f"door⊂person={cont_door:.2f}>={DOOR_PERSON_OVERLAP_CONTAINMENT_THR:.2f} (person⊂door={cont_person:.2f})"
            if door_mask_arr is not None and "mask" in pdet:
                pmask = np.asarray(pdet["mask"], dtype=bool)
                if pmask.shape == door_mask_arr.shape and door_mask_area > 0:
                    inter_pixels = float(np.count_nonzero(door_mask_arr & pmask))
                    ratio = inter_pixels / door_mask_area
                    if ratio >= DOOR_PERSON_OVERLAP_MASK_RATIO_THR:
                        return True, f"mask_ratio={ratio:.2f}>={DOOR_PERSON_OVERLAP_MASK_RATIO_THR:.2f}"
        return False, ""

    def _should_skip_duplicate_primary_line_event(
        self, frame_idx: int, obj_class: str, obj_box, track_id: int
    ) -> bool:
        if not DOOR_LINE_EVENT_DEDUP_ENABLED:
            return False
        if obj_class not in (rv.DOOR_CLASS_NAME, rv.TRIM_CLASS_NAME):
            return False
        if self._last_primary_line_event_frame is None or self._last_primary_line_event_center is None:
            return False
        if frame_idx - self._last_primary_line_event_frame > DOOR_LINE_EVENT_DEDUP_FRAMES:
            return False
        c = rv.center(obj_box)
        d = float(np.linalg.norm(c - self._last_primary_line_event_center))
        if d >= DOOR_LINE_EVENT_DEDUP_CENTER_DIST_PX:
            return False
        if self._last_primary_line_event_track_id is not None and int(track_id) == int(
            self._last_primary_line_event_track_id
        ):
            return True
        # Другой track_id — та же физическая дверь после краткой потери детекции / дробления трека.
        if self._last_primary_line_event_box is not None:
            iou = float(rv.iou_xyxy(self._last_primary_line_event_box, obj_box))
            if iou >= DOOR_LINE_EVENT_DEDUP_MIN_IOU_DIFFERENT_TRACK:
                return True
        if d < DOOR_LINE_EVENT_DEDUP_STRICT_CENTER_PX:
            return True
        return False

    def _find_overlapping_latched_track(self, tr: dict, frame_idx: int) -> dict | None:
        if not LINE_EVENT_SUPPRESS_FRAGMENT_OF_LATCHED_TRACK:
            return None
        if tr.get("tracking_group") != "door_trim_group":
            return None
        last_frame = self._last_primary_line_event_frame
        last_tid = self._last_primary_line_event_track_id
        if last_frame is None or last_tid is None:
            return None
        if frame_idx - last_frame > LINE_EVENT_FRAGMENT_RECENT_FRAMES:
            return None
        if int(last_tid) == int(tr["id"]):
            return None
        latched = next((t for t in self.tracks if int(t["id"]) == int(last_tid)), None)
        if latched is None:
            return None
        if not latched.get("updated_this_frame"):
            return None
        if not latched.get("counted_up"):
            return None
        if latched.get("tracking_group") != "door_trim_group":
            return None
        x1a, y1a, x2a, y2a = map(float, tr["box"])
        x1b, y1b, x2b, y2b = map(float, latched["box"])
        area_a = max(0.0, x2a - x1a) * max(0.0, y2a - y1a)
        area_b = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
        if area_a <= 0 or area_b <= 0:
            return None
        ix1 = max(x1a, x1b)
        iy1 = max(y1a, y1b)
        ix2 = min(x2a, x2b)
        iy2 = min(y2a, y2b)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return None
        iou = inter / (area_a + area_b - inter)
        containment = inter / min(area_a, area_b)
        if iou >= LINE_EVENT_FRAGMENT_IOU_THR or containment >= LINE_EVENT_FRAGMENT_CONTAINMENT_THR:
            return latched
        return None

    def _record_primary_line_event(self, frame_idx: int, obj_class: str, obj_box, track_id: int) -> None:
        if obj_class in (rv.DOOR_CLASS_NAME, rv.TRIM_CLASS_NAME):
            self._last_primary_line_event_frame = frame_idx
            self._last_primary_line_event_center = np.asarray(rv.center(obj_box), dtype=np.float32).copy()
            self._last_primary_line_event_box = np.asarray(obj_box, dtype=np.float32).copy()
            self._last_primary_line_event_track_id = int(track_id)

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

    def annotate_frame(self, frame: np.ndarray, all_dets: list, frame_idx: int) -> np.ndarray:
        self._ensure_line(frame)
        return _annotate_frame(frame, all_dets, frame_idx)

    def _signed_distance_to_line(self, point: np.ndarray) -> float:
        a, b = self.line
        line_len = float(np.linalg.norm(b - a))
        if line_len < 1e-6:
            return 0.0
        return float(rv.side_of_line(point, a, b) / line_len)

    def _center_distance(self, box) -> float:
        point = rv.center(box)
        return self._signed_distance_to_line(point) * float(self.bottom_side_sign_value)

    def _bottom_distance(self, box) -> float:
        point = rv.box_bottom_center(box)
        return self._signed_distance_to_line(point) * float(self.bottom_side_sign_value)

    def _box_line_metrics(self, box) -> dict:
        x1, y1, x2, y2 = map(float, box)
        points = [
            np.array([x1, y1], dtype=np.float32),
            np.array([x2, y1], dtype=np.float32),
            np.array([x1, y2], dtype=np.float32),
            np.array([x2, y2], dtype=np.float32),
        ]
        distances = [
            self._signed_distance_to_line(p) * float(self.bottom_side_sign_value)
            for p in points
        ]
        center_d = self._center_distance(box)
        bottom_d = self._bottom_distance(box)
        min_d = min(distances)
        max_d = max(distances)
        if min_d > LIVE_LINE_MARGIN_PX:
            state = "below"
        elif max_d < -LIVE_LINE_MARGIN_PX:
            state = "above"
        else:
            state = "intersects"
        return {
            "state": state, "min_d": float(min_d), "max_d": float(max_d),
            "center_d": float(center_d), "bottom_d": float(bottom_d),
        }

    def _bbox_geometric_straddles_line(self, box) -> bool:
        x1, y1, x2, y2 = map(float, box)
        corners = [
            np.array([x1, y1], dtype=np.float32), np.array([x2, y1], dtype=np.float32),
            np.array([x1, y2], dtype=np.float32), np.array([x2, y2], dtype=np.float32),
        ]
        scale = float(self.bottom_side_sign_value)
        vals = [self._signed_distance_to_line(p) * scale for p in corners]
        eps = 0.25
        return any(v > eps for v in vals) and any(v < -eps for v in vals)

    def _mask_line_overlap_px(self, mask: np.ndarray) -> int:
        if self.line is None:
            return 0
        m = np.asarray(mask)
        if m.size == 0 or not np.any(m):
            return 0
        h, w = m.shape[:2]
        stripe = np.zeros((h, w), dtype=np.uint8)
        p0 = tuple(np.clip(self.line[0].astype(int), [0, 0], [w - 1, h - 1]))
        p1 = tuple(np.clip(self.line[1].astype(int), [0, 0], [w - 1, h - 1]))
        cv2.line(stripe, p0, p1, 255, TRIPWIRE_MASK_LINE_THICKNESS_PX, cv2.LINE_8)
        return int(np.count_nonzero(m.astype(bool) & (stripe > 0)))

    def _mask_intersects_tripwire(self, mask: np.ndarray) -> bool:
        return self._mask_line_overlap_px(mask) >= TRIPWIRE_MASK_MIN_OVERLAP_PX

    def _object_intersects_tripwire(self, hist_item: dict) -> bool:
        mask = hist_item.get("mask")
        if mask is not None:
            mk = np.asarray(mask, dtype=bool)
            if not mk.any():
                return False
            return self._mask_intersects_tripwire(mk)
        box = hist_item["box"]
        bm = self._box_line_metrics(box)
        return bm["state"] == "intersects" or self._bbox_geometric_straddles_line(box)

    def _update_tripwire_leave_streak_and_rearm(self, tr: dict, hist_item: dict) -> None:
        hit = self._object_intersects_tripwire(hist_item)
        if hit:
            tr["tripwire_leave_streak"] = 0
        else:
            tr["tripwire_leave_streak"] = tr.get("tripwire_leave_streak", 0) + 1
        if LINE_EVENT_SIMPLE_FIRST_HIT_PER_TRACK:
            return
        if tr["tripwire_leave_streak"] >= LINE_TRIPWIRE_REARM_FRAMES:
            tr["counted_up"] = False

    def _track_direction(self, tr: dict) -> str:
        """
        Определяет направление пересечения линии: up / down / unknown.
        Использует среднее signed_dist первой и второй половин истории — устойчиво к шуму.
        Для коротких треков (2-3 точки) используется простая разница первая/последняя.
        """
        hist = tr.get("history", [])
        boxes = [it["box"] for it in hist if it.get("box") is not None]
        if len(boxes) < 2:
            return "unknown"

        dists = [self._center_distance(b) for b in boxes]

        if len(dists) >= 4:
            mid = len(dists) // 2
            first_mean = sum(dists[:mid]) / mid
            last_mean = sum(dists[mid:]) / (len(dists) - mid)
            delta = last_mean - first_mean
        else:
            # короткий трек: берём дельту первой и последней точки
            delta = dists[-1] - dists[0]

        if abs(delta) < DIRECTION_MIN_DELTA:
            return "unknown"

        # bottom_side_sign_value уже учтён в _center_distance:
        # положительный delta = движение в сторону "bottom" = вниз (down)
        return "down" if delta > 0 else "up"

    def _find_best_person_for_object_live(self, obj_det, person_dets, frame_shape):
        best = rv.find_best_person_for_object(obj_det, person_dets, frame_shape)
        if best is not None:
            return _copy_person_info_with_mask(best, person_dets)
        obj_box = obj_det["box"]
        obj_c = rv.center(obj_box)
        ow, oh, _ = rv.box_wh_area(obj_box)
        obj_diag = float(np.hypot(ow, oh))
        fh, fw = frame_shape[:2]
        roi = rv.expand_box(obj_box, fw, fh, LIVE_RELAXED_ROI_EXPAND_X, LIVE_RELAXED_ROI_EXPAND_Y)
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
                    "box": p_box.copy(), "confidence": p["confidence"],
                    "w": int(pw), "h": int(ph), "area": int(parea),
                    "center_dist": dist, "iou_with_roi": overlap, "relaxed": True,
                }
        return _copy_person_info_with_mask(best, person_dets)

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

    def _evaluate_track(self, track: dict) -> dict:
        curr_person = track["person_info_current"]
        hist_has_person, best_hist_person = rv.validate_person_near_track(track)
        has_person = hist_has_person or curr_person is not None
        person_info = curr_person if curr_person is not None else best_hist_person

        prev_d = None
        prev_center_d = None
        prev_box_state = None
        hist = track["history"]
        if len(hist) >= 2:
            prev_metrics = self._box_line_metrics(hist[-2]["box"])
            prev_d = prev_metrics["bottom_d"]
            prev_center_d = prev_metrics["center_d"]
            prev_box_state = prev_metrics["state"]
        last_item = hist[-1]
        curr_metrics = self._box_line_metrics(last_item["box"])
        curr_d = curr_metrics["bottom_d"]

        line_hit = self._object_intersects_tripwire(last_item)
        prev_hit = len(hist) >= 2 and self._object_intersects_tripwire(hist[-2])
        fresh_entry = len(hist) == 1 or not prev_hit
        latched = track["counted_up"]

        if LINE_EVENT_SIMPLE_FIRST_HIT_PER_TRACK:
            fresh_line_touch_ready = line_hit and not latched
        else:
            fresh_line_touch_ready = line_hit and not latched and fresh_entry

        reason = None
        if latched:
            reason = "LATCHED_ALREADY_COUNTED_THIS_PASS"
        elif len(hist) < LIVE_MIN_TRACK_HISTORY:
            reason = f"HISTORY_LT_{LIVE_MIN_TRACK_HISTORY}"
        elif not line_hit:
            reason = "NO_LINE_INTERSECTION"
        elif not LINE_EVENT_SIMPLE_FIRST_HIT_PER_TRACK and not fresh_entry:
            reason = "STILL_ON_LINE_NO_FRESH_ENTRY"
        else:
            reason = "READY_EVENT"

        return {
            "reason": reason,
            "curr_person": curr_person, "person_info": person_info,
            "has_person": has_person, "hist_has_person": hist_has_person,
            "prev_bottom_dist": prev_d, "curr_bottom_dist": curr_d,
            "prev_center_dist": prev_center_d, "curr_center_dist": curr_metrics["center_d"],
            "prev_box_state": prev_box_state, "curr_box_state": curr_metrics["state"],
            "curr_box_min_d": curr_metrics["min_d"], "curr_box_max_d": curr_metrics["max_d"],
            "line_intersects": line_hit, "prev_line_intersects": prev_hit,
            "fresh_entry": fresh_entry, "latched": latched,
            "tripwire_leave_streak": int(track.get("tripwire_leave_streak", 0)),
            "fresh_line_touch_ready": fresh_line_touch_ready,
        }

    def _save_debug_frame(self, filename: str, image: np.ndarray) -> None:
        out_path = self.events_dir / filename
        if cv2.imwrite(str(out_path), image):
            self.log(f"  сохранён кадр: {out_path}")

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
        direction: str = "unknown",
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
            else:
                rv.draw_box_with_label(
                    out, person_info["box"],
                    f"PERSON {person_info['w']}x{person_info['h']}",
                    (0, 255, 0), thickness=3,
                )
        cv2.putText(
            out, f"{direction.upper()} | {assoc_str}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.line(
            out,
            tuple(self.line[0].astype(int)),
            tuple(self.line[1].astype(int)),
            (255, 255, 255), 2,
        )
        return out

    def process_frame(self, frame: np.ndarray, all_dets: list, frame_idx: int) -> np.ndarray:
        self._ensure_line(frame)
        debug_frame = frame.copy()
        vis = _annotate_frame(debug_frame, all_dets, frame_idx)

        primary_dets, person_dets = rv.split_detections(all_dets)
        self._last_person_dets = person_dets
        self._push_person_dets_history(person_dets)

        enriched_primary = []
        for det in primary_dets:
            cls_name = det["class_name"]
            conf = float(det.get("confidence", 0.0))
            min_conf = (
                MIN_DOOR_CONFIDENCE if cls_name == rv.DOOR_CLASS_NAME
                else MIN_TRIM_CONFIDENCE if cls_name == rv.TRIM_CLASS_NAME
                else 0.0
            )
            if min_conf > 0.0 and conf < min_conf:
                continue

            if cls_name == rv.DOOR_CLASS_NAME:
                is_overlap, overlap_reason = self._door_overlaps_any_recent_person(
                    det["box"], det.get("mask")
                )
                if is_overlap:
                    ow, oh, oarea = rv.box_wh_area(det["box"])
                    self.log(
                        f"[SKIP DOOR≈PERSON] {self.source_name} | frame={frame_idx} | "
                        f"w={int(ow)} h={int(oh)} area={int(oarea)} | reason={overlap_reason}"
                    )
                    continue

            is_valid_size, invalid_reason = _is_valid_object_size(cls_name, det["box"])
            if FILTER_SMALL_OBJECTS_BEFORE_TRACKING and not is_valid_size:
                continue

            person_info = (
                self._find_best_person_for_object_live(det, person_dets, frame.shape)
                if LINE_EVENT_USE_PERSON_ASSOCIATION else None
            )
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
                dist = float(np.linalg.norm(obj_c - tr["centers"][-1]))
                if dist > rv.TRACK_DISTANCE:
                    continue
                overlap = rv.iou_xyxy(obj_box, tr["box"])
                score = overlap * 1000.0 - dist
                if score > best_score:
                    best_score = score
                    best_track = tr

            hist_item = {
                "frame_id": frame_idx,
                "box": obj_box.copy(),
                "confidence": det["confidence"],
                "class_name": obj_class,
                "person_info": det["person_info"],
            }
            if "mask" in det:
                hist_item["mask"] = det["mask"].copy()

            if best_track is not None:
                best_track["box"] = obj_box.copy()
                best_track["confidence"] = det["confidence"]
                best_track["class_name_current"] = obj_class
                best_track["person_info_current"] = det["person_info"]
                best_track["lost"] = 0
                best_track["centers"].append(obj_c)
                best_track["updated_this_frame"] = True
                if len(best_track["centers"]) > rv.TRACK_HISTORY:
                    best_track["centers"].pop(0)
                best_track["history"].append(hist_item)
                if len(best_track["history"]) > rv.TRACK_HISTORY:
                    best_track["history"].pop(0)
                self._update_tripwire_leave_streak_and_rearm(best_track, hist_item)
                updated_ids.add(best_track["id"])
            else:
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
                    "updated_this_frame": True,
                    "tripwire_leave_streak": 0,
                    "centers": [obj_c],
                    "history": [hist_item],
                }
                self._update_tripwire_leave_streak_and_rearm(new_track, hist_item)
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

        for tr in self.tracks:
            if not tr["updated_this_frame"]:
                continue
            eval_data = self._evaluate_track(tr)

            if len(tr["history"]) < LIVE_MIN_TRACK_HISTORY:
                continue
            if not eval_data.get("fresh_line_touch_ready"):
                continue

            obj_box = tr["box"]
            obj_class = self._get_event_class_recent(tr, EVENT_CLASS_WINDOW)

            if obj_class not in EVENT_PRIMARY_CLASSES:
                if not tr.get("rejected_class_skip", False):
                    tr["rejected_class_skip"] = True
                    self.log(
                        f"[SKIP CLASS ✖] {self.source_name} | frame={frame_idx} | "
                        f"class={obj_class} | track_id={tr['id']}"
                    )
                continue

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

            if EVENT_MIN_TRACK_HISTORY > 1 and len(tr["history"]) < EVENT_MIN_TRACK_HISTORY:
                if not tr.get("rejected_short_hist", False):
                    tr["rejected_short_hist"] = True
                    self.log(
                        f"[REJECT SHORT HIST ✖] {self.source_name} | frame={frame_idx} | "
                        f"hist_len={len(tr['history'])} < {EVENT_MIN_TRACK_HISTORY}"
                    )
                continue

            if EVENT_MIN_MEAN_CONFIDENCE > 0.0:
                recent_confs = [
                    float(it.get("confidence", 0.0))
                    for it in tr["history"][-EVENT_CLASS_WINDOW:]
                    if it.get("class_name") == obj_class
                ]
                mean_conf = sum(recent_confs) / len(recent_confs) if recent_confs else 0.0
                if mean_conf < EVENT_MIN_MEAN_CONFIDENCE:
                    if not tr.get("rejected_low_mean_conf", False):
                        tr["rejected_low_mean_conf"] = True
                        self.log(
                            f"[REJECT LOW CONF ✖] {self.source_name} | frame={frame_idx} | "
                            f"mean_conf={mean_conf:.2f} < {EVENT_MIN_MEAN_CONFIDENCE:.2f}"
                        )
                    continue

            last_hist = tr["history"][-1]
            mk = last_hist.get("mask")
            if mk is not None:
                mk_arr = np.asarray(mk, dtype=bool)
                if mk_arr.size and np.any(mk_arr):
                    overlap_px = self._mask_line_overlap_px(mk_arr)
                    mask_area = int(np.count_nonzero(mk_arr))
                    bbox_area = max(1.0, float(ow) * float(oh))
                    fill_ratio = mask_area / bbox_area
                    overlap_per_w = overlap_px / max(1.0, float(ow))
                    weak_abs = EVENT_MIN_MASK_OVERLAP_PX > 0 and overlap_px < EVENT_MIN_MASK_OVERLAP_PX
                    weak_rel = EVENT_MIN_MASK_OVERLAP_PER_BBOX_W > 0 and overlap_per_w < EVENT_MIN_MASK_OVERLAP_PER_BBOX_W
                    weak_fill = EVENT_MIN_MASK_FILL_RATIO > 0 and fill_ratio < EVENT_MIN_MASK_FILL_RATIO
                    if weak_abs or weak_rel or weak_fill:
                        if not tr.get("rejected_weak_overlap", False):
                            tr["rejected_weak_overlap"] = True
                            self.log(
                                f"[REJECT WEAK MASK ✖] {self.source_name} | frame={frame_idx} | "
                                f"overlap_px={overlap_px} fill={fill_ratio:.2f}"
                            )
                        continue

            if self._should_skip_duplicate_primary_line_event(frame_idx, obj_class, obj_box, tr["id"]):
                if not tr.get("rejected_dedup_line", False):
                    tr["rejected_dedup_line"] = True
                    self.log(
                        f"[REJECT DUP LINE ✖] {self.source_name} | frame={frame_idx} | "
                        f"reason=PRIMARY_LINE_EVENT_DEDUP_NEAR_TIME"
                    )
                tr["counted_up"] = True
                continue

            other = self._find_overlapping_latched_track(tr, frame_idx)
            if other is not None:
                if not tr.get("rejected_fragment_of_latched", False):
                    tr["rejected_fragment_of_latched"] = True
                    self.log(
                        f"[REJECT FRAGMENT ✖] {self.source_name} | frame={frame_idx} | "
                        f"overlaps_track_id={other['id']}"
                    )
                tr["counted_up"] = True
                continue

            if obj_class == rv.DOOR_CLASS_NAME:
                last_mask_for_check = last_hist.get("mask")
                is_overlap_evt, overlap_evt_reason = self._door_overlaps_any_recent_person(
                    obj_box, last_mask_for_check
                )
                if is_overlap_evt:
                    if not tr.get("rejected_door_person_overlap", False):
                        tr["rejected_door_person_overlap"] = True
                        self.log(
                            f"[REJECT DOOR≈PERSON ✖] {self.source_name} | frame={frame_idx} | "
                            f"reason={overlap_evt_reason}"
                        )
                    continue

            is_static, static_reason = self._track_is_static(tr)
            if is_static:
                if not tr.get("rejected_static_object", False):
                    tr["rejected_static_object"] = True
                    self.log(
                        f"[REJECT STATIC ✖] {self.source_name} | frame={frame_idx} | "
                        f"reason={static_reason}"
                    )
                continue

            # --- определяем направление стабильным способом ---
            direction = self._track_direction(tr)

            self._record_primary_line_event(frame_idx, obj_class, obj_box, tr["id"])
            tr["counted_up"] = True
            self.events_count += 1

            person_info = eval_data["person_info"] if LINE_EVENT_USE_PERSON_ASSOCIATION else None
            assoc_str = f"{obj_class}+line+{direction}"
            recent_classes = [item.get("class_name") for item in tr["history"][-EVENT_CLASS_WINDOW:]]

            log_data = {
                "video": self.source_name,
                "frame": frame_idx,
                "direction": direction,
                "event_type": "LINE_CROSS",
                "assoc": assoc_str,
                "object_class": obj_class,
                "object_class_window": EVENT_CLASS_WINDOW,
                "recent_classes": recent_classes,
                "object_w": int(ow),
                "object_h": int(oh),
                "object_area": int(oarea),
                "line_hit": True,
            }
            if person_info is not None:
                log_data["person_w"] = person_info["w"]
                log_data["person_h"] = person_info["h"]
                log_data["person_dist"] = round(person_info["center_dist"], 1)

            self.log("[EVENT ✔] " + json.dumps(log_data, ensure_ascii=False))

            if rv.SAVE:
                out = self._draw_event_frame(
                    debug_frame, tr, obj_box, obj_class, ow, oh,
                    person_info, assoc_str,
                    tr["history"][-1].get("mask"),
                    direction=direction,
                )
                filename = f"{self.source_stem}_LINE_{direction}_{obj_class}_{frame_idx:06d}.jpg"
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
        video_opened = pyqtSignal(int)
        frame_progress = pyqtSignal(int)

        def __init__(self, src, model, class_names, save_mp4: bool, mp4_path: Path, out_fps: float, run_dir: Path):
            super().__init__()
            self.src = src
            self.model = model
            self.class_names = class_names
            self._save_mp4 = save_mp4
            self._mp4_path = mp4_path
            self._out_fps = out_fps
            self._run_dir = run_dir
            self._stop = False
            self.video_writer: cv2.VideoWriter | None = None
            self._jsonl_fp: object | None = None
            self._seek_lock = threading.Lock()
            self._seek_frame_1based: int | None = None
            self._current_file_total_frames = 0

        def request_stop(self) -> None:
            self._stop = True

        def request_seek_frame(self, frame_1based: int) -> None:
            with self._seek_lock:
                self._seek_frame_1based = int(frame_1based)

        def run(self) -> None:
            _write_run_info(
                self._run_dir / "run_info.txt", self.src,
                {"out_fps": self._out_fps, "model": "RFDETRSegMedium", "seg_checkpoint": SEG_CHECKPOINT_PATH},
            )
            if LOG_EVERY_N_FRAMES > 0:
                self._jsonl_fp = open(self._run_dir / "detections.jsonl", "w", encoding="utf-8")
            self.log.emit(f"Папка прогона: {self._run_dir}")
            sources = _expand_sources(self.src)
            if not sources:
                self.log.emit(f"Не найдены видео: {self.src!r}")
                self.finished_ok.emit()
                return
            if len(sources) > 1:
                self.log.emit(f"Автопереключение: {len(sources)} файлов.")

            total_events = 0
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

                self.log.emit(f"[{source_idx}/{len(sources)}] {_source_label(current_src)}")
                event_processor = LiveEventProcessor(current_src, self._run_dir, self.log.emit)
                self.log.emit(event_processor.describe_output())

                self._current_file_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                if _is_video_file(current_src) and self._current_file_total_frames > 1:
                    self.video_opened.emit(self._current_file_total_frames)
                else:
                    self.video_opened.emit(0)

                last_dets: list = []
                frame_idx = 0

                while not self._stop:
                    pending_seek = None
                    with self._seek_lock:
                        if self._seek_frame_1based is not None:
                            pending_seek = self._seek_frame_1based
                            self._seek_frame_1based = None
                    forced_idx = None
                    if pending_seek is not None and _is_video_file(current_src):
                        t = max(1, int(pending_seek))
                        if self._current_file_total_frames > 0:
                            t = min(t, self._current_file_total_frames)
                        event_processor = LiveEventProcessor(current_src, self._run_dir, self.log.emit)
                        last_dets = []
                        cap.set(cv2.CAP_PROP_POS_FRAMES, t - 1)
                        forced_idx = t
                        self.log.emit(f"Перемотка → кадр {t}")

                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if LOOP_VIDEO_FILE and len(sources) == 1 and _is_video_file(current_src):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_idx = 0
                            continue
                        self.log.emit(f"Конец: {_source_label(current_src)}")
                        break

                    if forced_idx is not None:
                        frame_idx = forced_idx
                    else:
                        frame_idx += 1

                    did_detect = DETECT_EVERY_N <= 1 or (frame_idx % DETECT_EVERY_N == 1)
                    if did_detect:
                        last_dets = _load_frame_detections_seg(self.model, frame, self.class_names)

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

                    if self.video_writer is not None and self.video_writer.isOpened():
                        self.video_writer.write(vis)

                    if LOG_EVERY_N_FRAMES > 0 and frame_idx % LOG_EVERY_N_FRAMES == 0:
                        stats = _det_stats(last_dets)
                        self.log.emit(f"[кадр {frame_idx}] всего {len(last_dets)} | по классам: {stats['counts']}")
                        if self._jsonl_fp:
                            self._jsonl_fp.write(
                                json.dumps({"video": _source_label(current_src), "frame": frame_idx, **stats},
                                           ensure_ascii=False) + "\n"
                            )
                            self._jsonl_fp.flush()

                    if SAVE_ANNOTATED_JPEG_EVERY_N > 0 and frame_idx % SAVE_ANNOTATED_JPEG_EVERY_N == 0:
                        jpeg_path = event_processor.events_dir / f"annotated_{frame_idx:06d}.jpg"
                        if cv2.imwrite(str(jpeg_path), vis):
                            self.log.emit(f"  сохранён кадр: {jpeg_path}")

                    self.frame_ready.emit(vis)
                    if wait_ms > 0:
                        self.msleep(wait_ms)
                    if self._current_file_total_frames > 1 and _is_video_file(current_src):
                        self.frame_progress.emit(frame_idx)

                cap.release()
                total_events += event_processor.events_count
                self.log.emit(f"Итог {_source_label(current_src)}: событий {event_processor.events_count}")
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None

            if self._jsonl_fp is not None:
                self._jsonl_fp.close()
            self.log.emit(f"Итого событий: {total_events}")
            self.finished_ok.emit()


    class MainWindow(QMainWindow):
        def __init__(self, worker: DetWorker):
            super().__init__()
            self.setWindowTitle(WINDOW_TITLE)
            self._worker = worker
            self._full_pix: QPixmap | None = None
            self._video_total: int = 0
            self._pause_seek_sync = False

            central = QWidget()
            main_l = QVBoxLayout(central)
            self._label = QLabel()
            self._label.setAlignment(Qt.AlignCenter)
            self._label.setMinimumSize(960, 540)
            self._label.setStyleSheet("background-color: #1a1a1a;")
            main_l.addWidget(self._label, 1)

            seek_row = QHBoxLayout()
            self._lbl_seek = QLabel("Перемотка: ожидание файла…")
            self._slider = QSlider(Qt.Horizontal)
            self._slider.setRange(1, 1)
            self._slider.setEnabled(False)
            self._spin = QSpinBox()
            self._spin.setRange(1, 1)
            self._spin.setEnabled(False)
            self._btn_mid = QPushButton("Середина")
            self._btn_mid.setEnabled(False)
            seek_row.addWidget(self._lbl_seek)
            seek_row.addWidget(self._slider, 1)
            seek_row.addWidget(self._spin)
            seek_row.addWidget(self._btn_mid)
            main_l.addLayout(seek_row)
            self.setCentralWidget(central)

            worker.frame_ready.connect(self._on_frame)
            worker.log.connect(lambda m: print(m, flush=True))
            worker.video_opened.connect(self._on_video_opened)
            worker.frame_progress.connect(self._on_frame_progress)

            self._slider.sliderPressed.connect(self._on_seek_slider_pressed)
            self._slider.sliderReleased.connect(self._on_seek_slider_released)
            self._spin.editingFinished.connect(self._on_seek_spin)
            self._btn_mid.clicked.connect(self._on_seek_middle)
            self._spin.lineEdit().installEventFilter(self)

            QShortcut(QKeySequence("Q"), self, activated=self.close)
            QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.close)
            QShortcut(QKeySequence("M"), self, activated=self._on_seek_middle)

        def _on_video_opened(self, total_frames: int) -> None:
            self._video_total = int(total_frames)
            if self._video_total <= 1:
                self._lbl_seek.setText("Перемотка недоступна (поток/камера)")
                self._slider.setEnabled(False)
                self._spin.setEnabled(False)
                self._btn_mid.setEnabled(False)
                return
            self._lbl_seek.setText(f"Кадр 1…{self._video_total}")
            self._slider.setRange(1, self._video_total)
            self._spin.setRange(1, self._video_total)
            self._slider.setEnabled(True)
            self._spin.setEnabled(True)
            self._btn_mid.setEnabled(True)

        def eventFilter(self, obj, event):
            if hasattr(self, "_spin") and obj == self._spin.lineEdit():
                if event.type() == QEvent.FocusIn:
                    self._pause_seek_sync = True
                elif event.type() == QEvent.FocusOut:
                    self._pause_seek_sync = False
            return super().eventFilter(obj, event)

        def _on_frame_progress(self, frame_idx: int) -> None:
            if self._video_total <= 1 or self._pause_seek_sync:
                return
            self._slider.blockSignals(True)
            self._spin.blockSignals(True)
            v = max(1, min(int(frame_idx), self._video_total))
            self._slider.setValue(v)
            self._spin.setValue(v)
            self._slider.blockSignals(False)
            self._spin.blockSignals(False)

        def _on_seek_slider_pressed(self) -> None:
            self._pause_seek_sync = True

        def _on_seek_slider_released(self) -> None:
            if not self._slider.isEnabled():
                self._pause_seek_sync = False
                return
            v = self._slider.value()
            self._spin.setValue(v)
            self._worker.request_seek_frame(v)
            self._pause_seek_sync = False

        def _on_seek_spin(self) -> None:
            if not self._spin.isEnabled() or self._video_total <= 1:
                return
            v = max(1, min(int(self._spin.value()), self._video_total))
            self._spin.setValue(v)
            self._slider.setValue(v)
            self._worker.request_seek_frame(v)
            self._spin.clearFocus()

        def _on_seek_middle(self) -> None:
            if self._video_total <= 1:
                return
            mid = max(1, self._video_total // 2)
            self._slider.setValue(mid)
            self._spin.setValue(mid)
            self._worker.request_seek_frame(mid)

        def _apply_scale(self) -> None:
            if self._full_pix is None or self._full_pix.isNull():
                return
            self._label.setPixmap(
                self._full_pix.scaled(self._label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        def _on_frame(self, vis: np.ndarray) -> None:
            self._full_pix = QPixmap.fromImage(_bgr_to_qimage(vis))
            self._apply_scale()

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            self._apply_scale()

        def closeEvent(self, event) -> None:
            self._worker.request_stop()
            self._worker.wait(15000)
            if self._worker.video_writer is not None:
                self._worker.video_writer.release()
            event.accept()


def run_headless_mp4(model, class_names, src) -> int:
    run_dir = _new_run_dir()
    _write_run_info(run_dir / "run_info.txt", src,
                    {"mode": "headless_seg", "model": "RFDETRSegMedium", "seg_checkpoint": SEG_CHECKPOINT_PATH})
    jsonl_fp = None
    if LOG_EVERY_N_FRAMES > 0:
        jsonl_fp = open(run_dir / "detections.jsonl", "w", encoding="utf-8")
    print(f"Логи прогона: {run_dir}", flush=True)
    rc = 0
    total_events = 0
    sources = _expand_sources(src)
    if not sources:
        print(f"Не найдены видео: {src!r}", flush=True)
        if jsonl_fp:
            jsonl_fp.close()
        return 1

    try:
        for source_idx, current_src in enumerate(sources, start=1):
            cap = _try_open_video_capture(current_src)
            if cap is None:
                print(_format_capture_open_error(current_src), flush=True)
                rc = 1
                continue
            out_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            if out_fps < 1.0 or out_fps > 120.0:
                out_fps = 25.0
            wait_ms = _playback_wait_ms(cap, current_src)
            print(f"[{source_idx}/{len(sources)}] {_source_label(current_src)} | пауза ~{wait_ms} ms", flush=True)

            writer = None
            last_dets: list = []
            frame_idx = 0
            event_processor = LiveEventProcessor(current_src, run_dir, lambda msg: print(msg, flush=True))

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
                vis = (event_processor.process_frame(frame, last_dets, frame_idx) if did_detect
                       else event_processor.annotate_frame(frame, last_dets, frame_idx))
                h, w = vis.shape[:2]
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    target_mp4 = _recording_mp4_path(current_src, sources, event_processor.events_dir, MP4_OUTPUT)
                    writer = cv2.VideoWriter(str(target_mp4), fourcc, out_fps, (w, h))
                    if not writer.isOpened():
                        print("Не удалось открыть VideoWriter", flush=True)
                        rc = 1
                        break
                writer.write(vis)
                if LOG_EVERY_N_FRAMES > 0 and frame_idx % LOG_EVERY_N_FRAMES == 0:
                    stats = _det_stats(last_dets)
                    print(f"[кадр {frame_idx}] всего {len(last_dets)} | {stats['counts']}", flush=True)
                    if jsonl_fp:
                        jsonl_fp.write(json.dumps({"video": _source_label(current_src), "frame": frame_idx, **stats},
                                                   ensure_ascii=False) + "\n")
                        jsonl_fp.flush()
                if wait_ms > 0:
                    cv2.waitKey(wait_ms)

            total_events += event_processor.events_count
            print(f"Итог {_source_label(current_src)}: событий {event_processor.events_count}", flush=True)
            cap.release()
            if writer:
                writer.release()
    finally:
        if jsonl_fp:
            jsonl_fp.close()

    print(f"Итого событий: {total_events}", flush=True)
    return rc


def main() -> int:
    if not Path(SEG_CHECKPOINT_PATH).exists():
        print(f"Чекпоинт не найден: {SEG_CHECKPOINT_PATH}")
        return 1
    if not Path(rv.DATA_YAML_PATH).exists():
        print(f"data.yaml не найден: {rv.DATA_YAML_PATH}")
        return 1

    class_names = rv.load_class_names_from_yaml(rv.DATA_YAML_PATH)
    print("RF-DETR Seg | классы:", ", ".join(f"{i}:{n}" for i, n in enumerate(class_names)))
    model = _build_seg_model(SEG_CHECKPOINT_PATH, num_classes=len(class_names))

    if os.environ.get("RFDETR_LIVE_HEADLESS") == "1":
        return run_headless_mp4(model, class_names, LIVE_SOURCE)

    if not HAS_PYQT:
        print("Нужен PyQt5: pip install PyQt5")
        return 1

    out_fps = _probe_fps(LIVE_SOURCE)
    run_dir = _new_run_dir()
    print(f"Логи и артефакты прогона: {run_dir}", flush=True)

    app = QApplication(sys.argv)
    worker = DetWorker(LIVE_SOURCE, model, class_names, ALSO_SAVE_MP4, MP4_OUTPUT, out_fps, run_dir)
    win = MainWindow(worker)
    win.show()
    worker.start()
    return int(app.exec_())


if __name__ == "__main__":
    sys.exit(main())