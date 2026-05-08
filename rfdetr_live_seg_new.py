#!/usr/bin/env python3
"""
Эфир с RF-DETR Seg: маски на кадре, пересечение tripwire отслеживается по классу person.

Источник (LIVE_SOURCE): int — камера; str — один RTSP/файл; list — несколько URL.

Зависимость: pip install PyQt5 supervision pyyaml

Без окна: RFDETR_LIVE_HEADLESS=1 python rfdetr_live_seg_new.py

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
import cv2
import numpy as np
import supervision as sv
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
import yaml

# ---------- локально: data.yaml, линия, трекинг (без rfdetr_video_events) ----------
DATA_YAML_PATH = str(Path(__file__).resolve().parent / "new_dataset" / "data.yaml")
CONF_THRESHOLD = 0.5

PERSON_CLASS_NAME = "person"
DOOR_CLASS_NAME = "door"
TRIM_CLASS_NAME = "trim"

# Tripwire: нормализованные концы отрезка [0,1] × [0,1]
LINE: list[tuple[float, float]] = [
    (0.931721, 0.382750),
    (0.013977, 0.396481),
    (0.928862, 0.379318),
]

TRACK_DISTANCE = 185
TRACK_HISTORY = 20
MAX_LOST = 12

DRAW = True
SAVE = True
SAVE_REJECTED = True


def load_class_names_from_yaml(yaml_path: str) -> list[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "names" not in data:
        raise ValueError(f"В файле {yaml_path} нет поля 'names'.")
    names = data["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if isinstance(names, list):
        return names
    raise ValueError("Поле 'names' должно быть list или dict.")


def center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def box_bottom_center(box):
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


def box_wh_area(box):
    w = max(0.0, float(box[2] - box[0]))
    h = max(0.0, float(box[3] - box[1]))
    return w, h, w * h


def denorm_line(line, w, h):
    return np.array(
        [
            (float(line[0][0] * w), float(line[0][1] * h)),
            (float(line[1][0] * w), float(line[1][1] * h)),
        ],
        dtype=np.float32,
    )


def side_of_line(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def iou_xyxy(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    a1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    a2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union = a1 + a2 - inter + 1e-6
    return inter / union


def point_side_sign(point, line_a, line_b):
    v = side_of_line(point, line_a, line_b)
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def infer_top_bottom_sides(frame_w, frame_h, line):
    top_point = np.array([frame_w * 0.5, frame_h * 0.05], dtype=np.float32)
    bottom_point = np.array([frame_w * 0.5, frame_h * 0.95], dtype=np.float32)
    top_side = point_side_sign(top_point, line[0], line[1])
    bottom_side = point_side_sign(bottom_point, line[0], line[1])
    if top_side == 0:
        top_side = -1
    if bottom_side == 0:
        bottom_side = 1
    if top_side == bottom_side:
        top_side = -1
        bottom_side = 1
    return top_side, bottom_side


def frame_to_model_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.shape[2] == 3:
        bgr = frame
    else:
        raise ValueError(f"Неподдерживаемая форма кадра: {frame.shape}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_box_with_label(image, box, label, color, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


# ---------- источник (выбери один) ----------
# int — индекс USB-камеры; str — один .mp4 / RTSP; list — несколько URL или файлов подряд.
RECORDINGS_CAM0_DIR = str(Path(__file__).resolve().parent / "recordings_2" / "cam3")
_rtsp_one = os.environ.get("RFDETR_LIVE_RTSP", "").strip()
if _rtsp_one:
    LIVE_SOURCE: int | str | list = _rtsp_one
else:
    LIVE_SOURCE = RECORDINGS_CAM0_DIR
# LIVE_SOURCE = 0  # только если камера есть: ls /dev/video*

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
# Только person в треке и на превью событий (остальные классы детектора игнорируются для линии).
TRACK_PERSON_ONLY = True
# Если True — при пересечении линии не проверяем mean conf и движение (только геометрия трека).
PERSON_SIMPLE_CROSSING = True

# Папка с видео / один .mp4: замедление прокрутки (RTSP и камеры из списка — без этого)
FILE_PLAYBACK_SLOWDOWN = 2.0  # множитель паузы между кадрами (>1 — медленнее)
FILE_PLAYBACK_CAP_FPS = 18.0  # не быстрее этого FPS по таймеру (часто завышено в метаданных)

ALSO_SAVE_MP4 = False
MP4_OUTPUT = Path(__file__).resolve().parent / "rfdetr_live_out.mp4"

# ---------- логи и сохранение кадров ----------
# Для каждого запуска создаётся папка run_YYYY-mm-dd_HH-MM-SS с:
#   run_info.txt      — источник и настройки
#   detections.jsonl  — одна строка JSON на «отчётный» кадр (см. ниже)
#   annotated_*.jpg   — если включено сохранение картинок
LIVE_RUN_ROOT = Path(__file__).resolve().parent / "rfdetr_live_logs_3"
# Каждые N кадров: сообщение в консоль + строка в jsonl (0 = отключить периодику)
LOG_EVERY_N_FRAMES = 5
# Сохранять размеченный кадр JPEG каждые N кадров (0 = не сохранять изображения)
SAVE_ANNOTATED_JPEG_EVERY_N = 0

# ---------- debug / relaxed live event logic ----------
LIVE_DEBUG_EVENT_DECISIONS = True
LIVE_MIN_TRACK_HISTORY = 2
LIVE_LINE_MARGIN_PX = 18.0

# Доп. критерий DOWN, когда bbox долго «intersects», но нога по метрикам уходит вниз
# (иначе cs==want уже на обоих кадрах и point_side_sign не даёт перехода).
DOWN_INTERSECTS_MIN_BOTTOM_DELTA = 12.0
DOWN_INTERSECTS_MIN_CENTER_DELTA = 4.0
DOWN_INTERSECTS_BOTTOM_DEEP = 160.0
# Для intersects→intersects: «вход» только из полосы у линии сверху (нога ещё не глубоко внизу)
# или из уже глубокого пересечения с центром у линии. Между ними — типичный джиттер сидящего
# (prev_bottom ~20–100 без входа сверху), даёт ложный DOWN без этого разделения.
DOWN_INTERSECTS_PREV_BOTTOM_FROM_ABOVE_MAX = 18.0
DOWN_INTERSECTS_PREV_BOTTOM_DEEP_MIN = 115.0
DOWN_INTERSECTS_MAX_ABS_CENTER_DEEP = 108.0

# ---------- стабилизация класса события ----------
# Класс события определяется по последним N кадрам истории трека.
EVENT_CLASS_WINDOW = 7

# Жёстче засчитывать UP/DOWN: слабый conf / почти без движения (если PERSON_SIMPLE_CROSSING = False)
EVENT_UP_MIN_PRIMARY_MEAN_CONF = 0.45
EVENT_DOWN_MIN_PRIMARY_MEAN_CONF = 0.45
EVENT_MOTION_LOOKBACK_FRAMES = 15
EVENT_MIN_BOTTOM_TRAVEL_PX = 20.0

# ---------- минимальный размер person на момент события ----------
MIN_PERSON_WIDTH_PX = 24
MIN_PERSON_HEIGHT_PX = 40
MIN_PERSON_AREA_PX2 = 1200

# Если True — до трекинга отсекаем слишком мелких людей по порогам выше.
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
    """
    Открывает камеру (индекс), файл или RTSP/URL.
    Для числового индекса на Linux дополнительно пробует CAP_V4L2, если бэкенд по умолчанию не сработал.
    """
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
    """Сообщение в лог, если источник не открылся."""
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
    """Детекции для пайплайна событий; при Seg добавляется mask [H,W] bool."""
    detections = model.predict(frame_to_model_rgb(frame), threshold=CONF_THRESHOLD)
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


def _keep_only_person_detections(dets: list) -> list:
    if not TRACK_PERSON_ONLY:
        return dets
    return [d for d in dets if d.get("class_name") == PERSON_CLASS_NAME]


def _color_bgr(class_name: str):
    if class_name == DOOR_CLASS_NAME:
        return (255, 255, 0)
    if class_name == TRIM_CLASS_NAME:
        return (255, 0, 255)
    if class_name == PERSON_CLASS_NAME:
        return (0, 255, 0)
    return (180, 180, 180)


def _annotate_frame(
        frame: np.ndarray,
        last_dets: list,
        frame_idx: int,
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
            draw_box_with_label(vis, d["box"], label, _color_bgr(d["class_name"]), thickness=2)
    if SHOW_TRIPWIRE:
        line = denorm_line(LINE, w, h)
        cv2.line(
            vis,
            tuple(line[0].astype(int)),
            tuple(line[1].astype(int)),
            (255, 255, 255),
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
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = LIVE_RUN_ROOT / f"run_{stamp}"
    run_dir.mkdir(parents=False)
    return run_dir


def _det_stats(last_dets: list) -> dict:
    if TRACK_PERSON_ONLY:
        last_dets = _keep_only_person_detections(last_dets)
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


def _tracking_group_name(class_name: str) -> str:
    """Один трекинг-класс — person."""
    if class_name == PERSON_CLASS_NAME:
        return "person"
    return class_name


def _is_valid_object_size(class_name: str, box) -> tuple[bool, str]:
    if class_name != PERSON_CLASS_NAME:
        return True, "ok"
    x1, y1, x2, y2 = map(float, box)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    if w < MIN_PERSON_WIDTH_PX:
        return False, f"person_width_lt_{MIN_PERSON_WIDTH_PX}"
    if h < MIN_PERSON_HEIGHT_PX:
        return False, f"person_height_lt_{MIN_PERSON_HEIGHT_PX}"
    if area < MIN_PERSON_AREA_PX2:
        return False, f"person_area_lt_{MIN_PERSON_AREA_PX2}"
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
        self.events_count_down = 0
        self.rejected_count = 0
        self.rejected_count_down = 0
        self.prev_frame_for_event: np.ndarray | None = None

    def describe_output(self) -> str:
        return f"События и reject-кадры: {self.events_dir}"

    def _ensure_line(self, frame: np.ndarray) -> None:
        if self.line is not None:
            return
        h, w = frame.shape[:2]
        self.line = denorm_line(LINE, w, h)
        (
            self.top_side_sign_value,
            self.bottom_side_sign_value,
        ) = infer_top_bottom_sides(w, h, self.line)

    def annotate_frame(self, frame: np.ndarray, all_dets: list, frame_idx: int) -> np.ndarray:
        self._ensure_line(frame)
        vis = _annotate_frame(
            frame,
            _keep_only_person_detections(all_dets),
            frame_idx,
        )
        self.prev_frame_for_event = frame.copy()
        return vis

    def _signed_distance_to_line(self, point: np.ndarray) -> float:
        a, b = self.line
        line_len = float(np.linalg.norm(b - a))
        if line_len < 1e-6:
            return 0.0
        return float(side_of_line(point, a, b) / line_len)

    def _bottom_distance(self, box) -> float:
        point = box_bottom_center(box)
        return self._signed_distance_to_line(point) * float(self.bottom_side_sign_value)

    def _center_distance(self, box) -> float:
        point = center(box)
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
            "state": state,
            "min_d": float(min_d),
            "max_d": float(max_d),
            "center_d": float(center_d),
            "bottom_d": float(bottom_d),
        }

    def _crossed_down_loose(self, track) -> bool:
        """
        DOWN: нижняя середина бокса за кадр перешла на сторону линии, ближайшую к «низу» кадра.
        Повторно не сработает: см. counted_down на треке.
        """
        if len(track["history"]) < 2:
            return False
        if self.top_side_sign_value is None or self.bottom_side_sign_value is None:
            return False
        a, b = self.line
        prev_b = track["history"][-2]["box"]
        curr_b = track["history"][-1]["box"]
        prev_pt = box_bottom_center(prev_b)
        curr_pt = box_bottom_center(curr_b)
        ps = point_side_sign(prev_pt, a, b)
        cs = point_side_sign(curr_pt, a, b)
        want = int(self.bottom_side_sign_value)
        top_s = int(self.top_side_sign_value)
        pm = self._box_line_metrics(prev_b)
        cm = self._box_line_metrics(curr_b)

        # Весь бокс ушёл в полуплоскость «низ»
        if pm["state"] in {"above", "intersects"} and cm["state"] == "below":
            return True

        # Долго intersects: нога и центр за кадр смещаются «к низу», но без явного below.
        if pm["state"] == "intersects" and cm["state"] == "intersects":
            if (
                cm["bottom_d"] >= pm["bottom_d"] + DOWN_INTERSECTS_MIN_BOTTOM_DELTA
                and cm["center_d"] >= pm["center_d"] + DOWN_INTERSECTS_MIN_CENTER_DELTA
            ):
                from_above = pm["bottom_d"] <= DOWN_INTERSECTS_PREV_BOTTOM_FROM_ABOVE_MAX
                deep_straddle = pm["bottom_d"] >= DOWN_INTERSECTS_PREV_BOTTOM_DEEP_MIN
                if from_above or deep_straddle:
                    center_ok = from_above or abs(pm["center_d"]) <= DOWN_INTERSECTS_MAX_ABS_CENTER_DEEP
                    if center_ok and (cs == want or cm["bottom_d"] >= DOWN_INTERSECTS_BOTTOM_DEEP):
                        return True

        # Явный переход ноги: верх линии / на линии → сторона «низ»
        if cs == want and (ps == top_s or ps == 0):
            return True
        return False

    def _get_event_class_recent(self, track: dict, n: int = EVENT_CLASS_WINDOW) -> str:
        """
        Берём класс по последним N кадрам истории трека.
        Если есть ничья — побеждает более свежий класс.
        """
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

    def _track_primary_mean_conf_recent(self, track: dict, n: int) -> float:
        """Средняя уверенность по кадрам окна, где класс door или trim."""
        hist = track.get("history", [])[-n:]
        confs = [
            float(item["confidence"])
            for item in hist
            if item.get("class_name") == PERSON_CLASS_NAME
        ]
        if not confs:
            return 0.0
        return sum(confs) / len(confs)

    def _track_bottom_travel_px(self, track: dict, n: int) -> float:
        """
        «Пройденный» размах позиций нижней середины бокса за последние n кадров (диагональ bbox всех точек).
        Статичный объект → близко к 0; реально движущаяся дверь/профиль — обычно больше порога.
        """
        hist = track.get("history", [])[-n:]
        if len(hist) < 2:
            return 0.0
        pts = np.array(
            [box_bottom_center(item["box"]) for item in hist],
            dtype=np.float32,
        )
        span = pts.max(axis=0) - pts.min(axis=0)
        return float(np.linalg.norm(span))

    def _evaluate_track_down(self, track: dict) -> dict:
        curr_person = None
        hist_has_person = False
        has_person = True
        person_info = None

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

        loose_cross = self._crossed_down_loose(track)

        reason = None
        if track["counted_down"]:
            reason = "ALREADY_COUNTED_DOWN"
        elif len(track["history"]) < LIVE_MIN_TRACK_HISTORY:
            reason = f"HISTORY_LT_{LIVE_MIN_TRACK_HISTORY}"
        elif not loose_cross:
            reason = "NO_CROSS_DOWN_LOOSE"
        else:
            reason = "READY_EVENT_DOWN"

        return {
            "reason": reason,
            "curr_person": curr_person,
            "person_info": person_info,
            "has_person": has_person,
            "hist_has_person": hist_has_person,
            "prev_bottom_dist": prev_d,
            "curr_bottom_dist": curr_d,
            "prev_center_dist": prev_center_d,
            "curr_center_dist": curr_metrics["center_d"],
            "prev_box_state": prev_box_state,
            "curr_box_state": curr_metrics["state"],
            "curr_box_min_d": curr_metrics["min_d"],
            "curr_box_max_d": curr_metrics["max_d"],
            "loose_cross_down": loose_cross,
        }

    def _log_track_decision_down(self, track: dict, frame_idx: int, eval_data: dict) -> None:
        if not LIVE_DEBUG_EVENT_DECISIONS:
            return

        ow, oh, oarea = box_wh_area(track["box"])
        recent_event_class = self._get_event_class_recent(track)
        parts = [
            f"[TRACK DN] {self.source_name}",
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
            f"loose_down={int(eval_data['loose_cross_down'])}",
            f"has_person={int(eval_data['has_person'])}",
            f"reason={eval_data['reason']}",
        ]
        if eval_data["prev_box_state"] is not None:
            parts.append(f"prev_box={eval_data['prev_box_state']}")
        parts.append(f"curr_box={eval_data['curr_box_state']}")
        if eval_data["prev_bottom_dist"] is not None:
            parts.append(f"prev_line_d={eval_data['prev_bottom_dist']:.1f}")
        parts.append(f"curr_line_d={eval_data['curr_bottom_dist']:.1f}")
        if eval_data["prev_center_dist"] is not None:
            parts.append(f"prev_center_d={eval_data['prev_center_dist']:.1f}")
        parts.append(f"curr_center_d={eval_data['curr_center_dist']:.1f}")
        parts.append(f"box_min_d={eval_data['curr_box_min_d']:.1f}")
        parts.append(f"box_max_d={eval_data['curr_box_max_d']:.1f}")

        person_info = eval_data["person_info"]
        if person_info is not None:
            parts.append(f"person_dist={person_info['center_dist']:.1f}")
            parts.append(f"person_wh={person_info['w']}x{person_info['h']}")
            parts.append(f"person_conf={person_info['confidence']:.2f}")
            if person_info.get("relaxed"):
                parts.append("person_match=relaxed")

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
            banner: str = "REJECT | person event rules",
    ) -> np.ndarray:
        rej = frame.copy()
        if not DRAW:
            return rej
        obj_color = (255, 0, 0)
        if obj_mask is not None and np.any(obj_mask):
            _blend_mask_contour_bgr(rej, obj_mask, obj_color, LIVE_MASK_OPACITY, LIVE_POLYGON_THICKNESS)
            cx, cy = _mask_centroid(obj_mask)
            cv2.putText(
                rej,
                f"{obj_class.upper()} REJECT {int(ow)}x{int(oh)}",
                (max(5, cx - 100), max(22, cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                obj_color,
                2,
                cv2.LINE_AA,
            )
        else:
            draw_box_with_label(
                rej,
                obj_box,
                f"{obj_class.upper()} REJECT {int(ow)}x{int(oh)}",
                obj_color,
                thickness=3,
            )
        cv2.putText(
            rej,
            banner,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            rej,
            tuple(self.line[0].astype(int)),
            tuple(self.line[1].astype(int)),
            (255, 255, 255),
            2,
        )
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
        if not DRAW:
            return out
        obj_color = _color_bgr(obj_class)
        if obj_mask is not None and np.any(obj_mask):
            _blend_mask_contour_bgr(out, obj_mask, obj_color, LIVE_MASK_OPACITY, LIVE_POLYGON_THICKNESS)
            cx, cy = _mask_centroid(obj_mask)
            cv2.putText(
                out,
                f"{obj_class.upper()} {int(ow)}x{int(oh)} conf={track['confidence']:.2f}",
                (max(5, cx - 120), max(22, cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                obj_color,
                2,
                cv2.LINE_AA,
            )
        else:
            draw_box_with_label(
                out,
                obj_box,
                f"{obj_class.upper()} {int(ow)}x{int(oh)} conf={track['confidence']:.2f}",
                obj_color,
                thickness=3,
            )
        cv2.putText(
            out,
            f"{direction} | {assoc_str}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            out,
            tuple(self.line[0].astype(int)),
            tuple(self.line[1].astype(int)),
            (255, 255, 255),
            2,
        )
        return out

    def process_frame(self, frame: np.ndarray, all_dets: list, frame_idx: int) -> np.ndarray:
        self._ensure_line(frame)
        debug_frame = frame.copy()
        person_dets = _keep_only_person_detections(all_dets)
        vis = _annotate_frame(debug_frame, person_dets, frame_idx)
        primary_dets = person_dets

        enriched_primary = []
        for det in primary_dets:
            is_valid_size, invalid_reason = _is_valid_object_size(det["class_name"], det["box"])

            if FILTER_SMALL_OBJECTS_BEFORE_TRACKING and not is_valid_size:
                ow, oh, oarea = box_wh_area(det["box"])
                self.log(
                    f"[SKIP SMALL] {self.source_name} | frame={frame_idx} | "
                    f"class={det['class_name']} | conf={det['confidence']:.2f} | "
                    f"w={int(ow)} h={int(oh)} area={int(oarea)} | reason={invalid_reason}"
                )
                continue

            person_info = None
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
            obj_c = center(obj_box)

            best_track = None
            best_score = -1e18

            for tr in self.tracks:
                if tr["tracking_group"] != tracking_group:
                    continue

                prev_box = tr["box"]
                prev_c = tr["centers"][-1]

                dist = float(np.linalg.norm(obj_c - prev_c))
                if dist > TRACK_DISTANCE:
                    continue

                overlap = iou_xyxy(obj_box, prev_box)
                score = overlap * 1000.0 - dist
                if score > best_score:
                    best_score = score
                    best_track = tr

            if best_track is not None:
                best_track["box"] = obj_box.copy()
                best_track["confidence"] = det["confidence"]
                best_track["class_name_current"] = obj_class
                best_track["person_info_current"] = det["person_info"]
                best_track["lost"] = 0
                best_track["updated_this_frame"] = True
                best_track["centers"].append(obj_c)

                if len(best_track["centers"]) > TRACK_HISTORY:
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
                if len(best_track["history"]) > TRACK_HISTORY:
                    best_track["history"].pop(0)

                updated_ids.add(best_track["id"])
            else:
                hist_item = {
                    "frame_id": frame_idx,
                    "box": obj_box.copy(),
                    "confidence": det["confidence"],
                    "class_name": obj_class,
                    "person_info": det["person_info"],
                }
                if "mask" in det:
                    hist_item["mask"] = det["mask"].copy()
                self.tracks.append(
                    {
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
                        "centers": [obj_c],
                        "history": [hist_item],
                    }
                )
                updated_ids.add(self.next_track_id)
                self.next_track_id += 1

        alive_tracks = []
        for tr in self.tracks:
            if tr["id"] not in updated_ids:
                tr["lost"] += 1
                tr["updated_this_frame"] = False
            if tr["lost"] <= MAX_LOST:
                alive_tracks.append(tr)
        self.tracks = alive_tracks

        for tr in self.tracks:
            if tr["counted_down"]:
                continue
            eval_down = self._evaluate_track_down(tr)
            if tr["updated_this_frame"]:
                self._log_track_decision_down(tr, frame_idx, eval_down)

            if len(tr["history"]) < LIVE_MIN_TRACK_HISTORY:
                continue
            if not eval_down["loose_cross_down"]:
                continue

            event_hist_item = tr["history"][-2] if len(tr["history"]) >= 2 else tr["history"][-1]
            obj_box = event_hist_item["box"]
            obj_class = event_hist_item.get("class_name", self._get_event_class_recent(tr, EVENT_CLASS_WINDOW))
            event_conf = float(event_hist_item.get("confidence", tr["confidence"]))

            ow, oh, oarea = box_wh_area(obj_box)

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

            if not eval_down["has_person"]:
                if not tr["rejected_down"]:
                    tr["rejected_down"] = True
                    self.rejected_count_down += 1
                    self.log(
                        f"[REJECT ✖ DOWN] {self.source_name} | frame={frame_idx} | "
                        f"class={obj_class} | obj_w={int(ow)} obj_h={int(oh)} obj_area={int(oarea)} | "
                        "reason=NO_PERSON_NEAR_OBJECT"
                    )
                    if SAVE and SAVE_REJECTED:
                        rej = self._draw_reject_frame(
                            debug_frame,
                            obj_box,
                            obj_class,
                            ow,
                            oh,
                            tr["history"][-1].get("mask"),
                            banner="REJECT DOWN | no person near object",
                        )
                        filename = f"{self.source_stem}_REJECT_DOWN_{obj_class}_{frame_idx:06d}.jpg"
                        self._save_debug_frame(filename, rej)
                        vis = rej
                continue

            if not PERSON_SIMPLE_CROSSING:
                look_m = max(2, min(EVENT_MOTION_LOOKBACK_FRAMES, len(tr["history"])))
                mean_conf = self._track_primary_mean_conf_recent(tr, EVENT_CLASS_WINDOW)
                if mean_conf < EVENT_DOWN_MIN_PRIMARY_MEAN_CONF:
                    if not tr.get("rejected_low_conf_down", False):
                        tr["rejected_low_conf_down"] = True
                        self.rejected_count_down += 1
                        self.log(
                            f"[REJECT ✖ DOWN] {self.source_name} | frame={frame_idx} | "
                            f"class={obj_class} | mean_conf={mean_conf:.2f} < {EVENT_DOWN_MIN_PRIMARY_MEAN_CONF} | "
                            "reason=LOW_MEAN_CONF_STABILIZE"
                        )
                        if SAVE and SAVE_REJECTED:
                            rej = self._draw_reject_frame(
                                debug_frame,
                                obj_box,
                                obj_class,
                                ow,
                                oh,
                                tr["history"][-1].get("mask"),
                                banner="REJECT DOWN | LOW_MEAN_CONF_STABILIZE",
                            )
                            fn = f"{self.source_stem}_REJECT_DOWN_LOWCONF_{obj_class}_{frame_idx:06d}.jpg"
                            self._save_debug_frame(fn, rej)
                            vis = rej
                    continue

                travel = self._track_bottom_travel_px(tr, look_m)
                if travel < EVENT_MIN_BOTTOM_TRAVEL_PX:
                    if not tr.get("rejected_static_object_down", False):
                        tr["rejected_static_object_down"] = True
                        self.rejected_count_down += 1
                        self.log(
                            f"[REJECT ✖ DOWN] {self.source_name} | frame={frame_idx} | "
                            f"class={obj_class} | bottom_travel_px={travel:.1f} < {EVENT_MIN_BOTTOM_TRAVEL_PX} | "
                            "reason=STATIC_OBJECT_NO_TRAVEL"
                        )
                        if SAVE and SAVE_REJECTED:
                            rej = self._draw_reject_frame(
                                debug_frame,
                                obj_box,
                                obj_class,
                                ow,
                                oh,
                                tr["history"][-1].get("mask"),
                                banner="REJECT DOWN | STATIC_OBJECT_NO_TRAVEL",
                            )
                            fn = f"{self.source_stem}_REJECT_DOWN_STATIC_{obj_class}_{frame_idx:06d}.jpg"
                            self._save_debug_frame(fn, rej)
                            vis = rej
                    continue

            tr["counted_down"] = True
            self.events_count_down += 1

            person_info = None
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
                "object_conf": round(event_conf, 3),
            }
            self.log("[EVENT ✔ DOWN] " + json.dumps(log_data, ensure_ascii=False))

            if SAVE:
                event_frame = self.prev_frame_for_event if self.prev_frame_for_event is not None else debug_frame
                out = self._draw_event_frame(
                    event_frame,
                    tr,
                    obj_box,
                    obj_class,
                    ow,
                    oh,
                    person_info,
                    assoc_str,
                    event_hist_item.get("mask"),
                    direction="DOWN",
                )
                filename = f"{self.source_stem}_DOWN_{assoc_str}_{frame_idx:06d}.jpg"
                self._save_debug_frame(filename, out)
                vis = out

        return vis


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class DetWorker(QThread):
    frame_ready = pyqtSignal(object)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal()
    """total_frames, 0 = перемотка не поддерживается (камера/RTSP)"""
    video_opened = pyqtSignal(int)
    """текущий номер кадра (1..N) для синхронизации слайдера"""
    frame_progress = pyqtSignal(int)

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
        self._seek_lock = threading.Lock()
        self._seek_frame_1based: int | None = None
        self._current_file_total_frames = 0

    def request_stop(self) -> None:
        self._stop = True

    def request_seek_frame(self, frame_1based: int) -> None:
        """Перемотка только для открытого видеофайла (см. video_opened)."""
        with self._seek_lock:
            self._seek_frame_1based = int(frame_1based)

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
                if pending_seek is not None:
                    if _is_video_file(current_src):
                        t = max(1, int(pending_seek))
                        if self._current_file_total_frames > 0:
                            t = min(t, self._current_file_total_frames)
                        event_processor = LiveEventProcessor(
                            current_src, self._run_dir, self.log.emit
                        )
                        last_dets = []
                        cap.set(cv2.CAP_PROP_POS_FRAMES, t - 1)
                        forced_idx = t
                        self.log.emit(
                            f"Перемотка → кадр {t}"
                            + (
                                f" / {self._current_file_total_frames}"
                                if self._current_file_total_frames > 0
                                else ""
                            )
                        )
                ok, frame = cap.read()
                if not ok or frame is None:
                    if LOOP_VIDEO_FILE and len(sources) == 1 and _is_video_file(current_src):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        self.log.emit("Повтор с начала файла.")
                        continue
                    self.log.emit(f"Конец источника: {_source_label(current_src)}")
                    break

                if forced_idx is not None:
                    frame_idx = forced_idx
                else:
                    frame_idx += 1
                did_detect = DETECT_EVERY_N <= 1 or (frame_idx % DETECT_EVERY_N == 1)
                if did_detect:
                    last_dets = _keep_only_person_detections(
                        _load_frame_detections_seg(self.model, frame, self.class_names)
                    )

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
                    target_mp4 = (
                        self._mp4_path
                        if len(sources) == 1
                        else event_processor.events_dir / f"{_source_stem(current_src)}.mp4"
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

                if self._current_file_total_frames > 1 and _is_video_file(current_src):
                    self.frame_progress.emit(frame_idx)

            cap.release()
            total_events += event_processor.events_count_down
            total_rejected += event_processor.rejected_count_down
            self.log.emit(
                f"Итог по {_source_label(current_src)}: DOWN {event_processor.events_count_down}; "
                f"отклонено DOWN {event_processor.rejected_count_down}"
            )
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

        if self._jsonl_fp is not None:
            self._jsonl_fp.close()
            self._jsonl_fp = None
        self.log.emit(
            f"Итого DOWN: найдено {total_events}, отклонено {total_rejected}"
        )
        self.finished_ok.emit()


class MainWindow(QMainWindow):
    def __init__(self, worker: DetWorker):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self._worker = worker
        self._full_pix: QPixmap | None = None
        self._video_total: int = 0
        self._pause_seek_sync = False  # не перебивать ползунок/поле, пока пользователь вводит кадр

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
        self._btn_mid.setToolTip("Перейти к ~50% длины ролика")
        seek_row.addWidget(self._lbl_seek)
        seek_row.addWidget(self._slider, 1)
        seek_row.addWidget(self._spin)
        seek_row.addWidget(self._btn_mid)
        main_l.addLayout(seek_row)

        self.setCentralWidget(central)

        worker.frame_ready.connect(self._on_frame)
        worker.log.connect(self._on_log)
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
            self._lbl_seek.setText("Перемотка недоступна (поток/камера или длина неизвестна)")
            self._slider.setEnabled(False)
            self._spin.setEnabled(False)
            self._btn_mid.setEnabled(False)
            return
        self._lbl_seek.setText(
            f"Кадр 1…{self._video_total} (клик в число — ввод; ползунок — тяни; Enter — перейти)"
        )
        self._slider.setRange(1, self._video_total)
        self._spin.setRange(1, self._video_total)
        self._slider.setEnabled(True)
        self._spin.setEnabled(True)
        self._btn_mid.setEnabled(True)

    def eventFilter(self, obj, event):  # noqa: ANN001
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
                    last_dets = _keep_only_person_detections(
                        _load_frame_detections_seg(model, frame, class_names)
                    )
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
                    target_mp4 = MP4_OUTPUT if len(
                        sources) == 1 else event_processor.events_dir / f"{_source_stem(current_src)}.mp4"
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

            total_events += event_processor.events_count_down
            total_rejected += event_processor.rejected_count_down
            print(
                f"Итог по {_source_label(current_src)}: DOWN {event_processor.events_count_down}; "
                f"отклонено DOWN {event_processor.rejected_count_down}",
                flush=True,
            )
            cap.release()
            if writer is not None:
                writer.release()
    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()

    print(
        f"Итого DOWN: найдено {total_events}, отклонено {total_rejected}",
        flush=True,
    )
    if rc == 0:
        print(f"Готово: {MP4_OUTPUT} | логи: {run_dir}", flush=True)
    return rc


def main() -> int:
    if not Path(SEG_CHECKPOINT_PATH).exists():
        print(f"Чекпоинт сегментации не найден: {SEG_CHECKPOINT_PATH}")
        return 1
    if not Path(DATA_YAML_PATH).exists():
        print(f"data.yaml не найден: {DATA_YAML_PATH}")
        return 1

    if os.environ.get("RFDETR_LIVE_HEADLESS") == "1":
        class_names = load_class_names_from_yaml(DATA_YAML_PATH)
        print("RF-DETR Seg | классы:", ", ".join(f"{i}:{n}" for i, n in enumerate(class_names)))
        model = _build_seg_model(SEG_CHECKPOINT_PATH, num_classes=len(class_names))
        return run_headless_mp4(model, class_names, LIVE_SOURCE)

    class_names = load_class_names_from_yaml(DATA_YAML_PATH)
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
