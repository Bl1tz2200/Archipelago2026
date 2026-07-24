"""Распознавание ArUco-меток на кадре камеры.

Своим детектором, а не топиком `/markers`: тип сообщения этого топика в документации
Обрика не описан, а кадры `drone.image` дают ровно то, что нужно — ID метки и её центр
в пикселях. Ни ROS, ни `sverk_interfaces` здесь не нужны, поэтому модуль одинаково
работает на дроне, на записи и в тестах.

Отсюда же берётся ширина полосы обзора — **в метках, а не в метрах**: сколько столбцов
и строк меток помещается в кадр на рабочей высоте. Это и есть шаг разрежения проходов
поискового маршрута.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import field
from .field import Node, Numbering


@dataclass
class Sight:
    """Одна метка, увиденная на кадре."""

    id: int
    center: Tuple[float, float]        # пиксели
    corners: np.ndarray                # 4×2, пиксели
    area: float                        # пиксели
    size_percent: float                # доля площади кадра

    def distance_to(self, point: Tuple[float, float]) -> float:
        return math.hypot(self.center[0] - point[0], self.center[1] - point[1])


def _dictionary(name: str):
    """Словарь ArUco по имени, с понятной ошибкой вместо AttributeError."""
    attr = getattr(cv2.aruco, name, None)
    if attr is None:
        available = [a for a in dir(cv2.aruco) if a.startswith("DICT_")]
        raise ValueError(f"неизвестный словарь ArUco '{name}'; доступны: {', '.join(available)}")
    getter = getattr(cv2.aruco, "getPredefinedDictionary", None)
    if getter is None:  # OpenCV < 4.7
        getter = cv2.aruco.Dictionary_get
    return getter(attr)


class MarkerDetector:
    """Детектор ArUco с шимом над двумя поколениями API OpenCV.

    4.7+ / 5.x — класс `cv2.aruco.ArucoDetector`; 4.5 / 4.6 — функция
    `cv2.aruco.detectMarkers` с объектом параметров. На борту вероятнее второе.
    """

    def __init__(self, dictionary: str = "DICT_4X4_50", min_marker_percent: float = 0.02) -> None:
        self.dictionary_name = dictionary
        self.min_marker_percent = float(min_marker_percent)
        self._dict = _dictionary(dictionary)

        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
        else:  # OpenCV < 4.7
            params = cv2.aruco.DetectorParameters_create()
        self._params = params

        self._detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)

    # ------------------------------------------------------------------ детекция

    def detect(self, frame: np.ndarray) -> List[Sight]:
        """Метки на кадре, отсортированные по убыванию площади."""
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dict, parameters=self._params)
        if ids is None or len(ids) == 0:
            return []

        frame_area = float(frame.shape[0] * frame.shape[1])
        sights: List[Sight] = []
        for quad, mid in zip(corners, ids.flatten()):
            pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
            area = float(cv2.contourArea(pts))
            size_percent = 100.0 * area / frame_area if frame_area else 0.0
            if size_percent < self.min_marker_percent:
                continue
            cx, cy = pts.mean(axis=0)
            sights.append(Sight(id=int(mid), center=(float(cx), float(cy)),
                                corners=pts, area=area, size_percent=size_percent))

        sights.sort(key=lambda s: s.area, reverse=True)
        return sights


# ------------------------------------------------------------------- выбор метки


def frame_center(frame_size: Tuple[int, int]) -> Tuple[float, float]:
    width, height = frame_size
    return width / 2.0, height / 2.0


def frame_diagonal(frame_size: Tuple[int, int]) -> float:
    width, height = frame_size
    return math.hypot(width, height)


def find(sights: Sequence[Sight], marker_id: int) -> Optional[Sight]:
    """Метка с нужным ID среди увиденных."""
    for sight in sights:
        if sight.id == marker_id:
            return sight
    return None


def nearest_to_center(
    sights: Sequence[Sight], frame_size: Tuple[int, int], ids: Sequence[int] = ()
) -> Optional[Sight]:
    """Ближайшая к центру кадра метка — та, над которой висит дрон.

    `ids` сужает выбор: например, только угловые метки.
    """
    center = frame_center(frame_size)
    allowed = set(ids) if ids else None
    candidates = [s for s in sights if allowed is None or s.id in allowed]
    if not candidates:
        return None
    return min(candidates, key=lambda s: s.distance_to(center))


def start_marker(
    sights: Sequence[Sight],
    frame_size: Tuple[int, int],
    numbering: Numbering = field.DEFAULT_NUMBERING,
) -> Optional[Sight]:
    """Стартовый маркер: ближайшая к центру кадра **угловая** метка.

    Регламент, п. 2.1.1: дрон-лидер ставят в назначенный угол, ID метки этого угла
    задаёт фигуру. В кадре на рабочей высоте видно несколько меток, поэтому угловая
    выбирается явно, а не «первая попавшаяся».
    """
    return nearest_to_center(sights, frame_size, ids=field.corner_ids(numbering))


def offset_from_center(
    sight: Sight, frame_size: Tuple[int, int]
) -> Tuple[float, float, float]:
    """Смещение метки от центра кадра: (dx, dy, доля диагонали кадра)."""
    cx, cy = frame_center(frame_size)
    dx = sight.center[0] - cx
    dy = sight.center[1] - cy
    return dx, dy, math.hypot(dx, dy) / frame_diagonal(frame_size)


# --------------------------------------------------------- ширина обзора в метках


def marker_pitch(
    sights: Sequence[Sight], numbering: Numbering = field.DEFAULT_NUMBERING
) -> Optional[float]:
    """Шаг сетки меток в пикселях кадра — по парам соседних меток.

    Единственная величина, которая связывает кадр с полем, и она измеряется, а не
    берётся из метража: сколько пикселей от метки до соседней метки на этой высоте.
    """
    indexed = []
    for sight in sights:
        try:
            indexed.append((numbering.node_of(sight.id), sight))
        except ValueError:
            continue

    pitches = []
    for i, (node_a, a) in enumerate(indexed):
        for node_b, b in indexed[i + 1:]:
            dcol = abs(node_a[0] - node_b[0])
            drow = abs(node_a[1] - node_b[1])
            # Только соседи по стороне: диагональ дала бы шаг, умноженный на √2.
            if dcol + drow != 1:
                continue
            pitches.append(a.distance_to(b.center))
    if not pitches:
        return None
    pitches.sort()
    return pitches[len(pitches) // 2]


def visible_span(
    sights: Sequence[Sight],
    numbering: Numbering = field.DEFAULT_NUMBERING,
    frame_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """Полоса обзора камеры в метках: сколько рядов меток охватывает кадр.

    Считается по шагу сетки в пикселях, а не по разбросу найденных ID: над угловой
    меткой половина поля просто отсутствует, и разброс ID занизил бы полосу обзора,
    хотя камера охватывает столько же пола, сколько в середине поля.

    Возвращается нечётное число рядов — полоса симметрична относительно метки,
    над которой висит дрон. Без `frame_size` работает старым способом, по разбросу.
    """
    pitch = marker_pitch(sights, numbering) if frame_size else None
    if pitch and pitch > 1.0:
        width, height = frame_size
        return (2 * int(width / 2 // pitch) + 1, 2 * int(height / 2 // pitch) + 1)

    nodes = to_nodes(sights, numbering)
    if not nodes:
        return 0, 0
    cols = {c for c, _ in nodes}
    rows = {r for _, r in nodes}
    return max(cols) - min(cols) + 1, max(rows) - min(rows) + 1


def to_nodes(
    sights: Sequence[Sight], numbering: Numbering = field.DEFAULT_NUMBERING
) -> List[Node]:
    """Узлы поля для увиденных меток; чужие ID молча отбрасываются."""
    nodes: List[Node] = []
    for sight in sights:
        try:
            nodes.append(numbering.node_of(sight.id))
        except ValueError:
            continue
    return nodes


def layout(
    sights: Sequence[Sight], frame_size: Tuple[int, int]
) -> str:
    """ID меток в том порядке, в каком они лежат в кадре.

    Печатается `tools/marker_check.py`: по этой картинке сразу видно, совпадает ли
    физическая раскладка с `markers.row_major` / `flip_x` / `flip_y` в конфиге.
    """
    if not sights:
        return "меток не видно"

    rows: Dict[int, List[Sight]] = {}
    tolerance = frame_size[1] * 0.15  # метки одной строки кадра лежат примерно на одной высоте
    for sight in sorted(sights, key=lambda s: s.center[1]):
        for key, bucket in rows.items():
            if abs(sight.center[1] - key) <= tolerance:
                bucket.append(sight)
                break
        else:
            rows[int(sight.center[1])] = [sight]

    lines = []
    for key in sorted(rows):
        line = " ".join(f"{s.id:3d}" for s in sorted(rows[key], key=lambda s: s.center[0]))
        lines.append(f"  {line}")
    return "\n".join(lines)


def draw(frame: np.ndarray, sights: Sequence[Sight], target: Optional[int] = None) -> np.ndarray:
    """Разметка меток для публикации в `/out_detection`."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    cv2.drawMarker(canvas, (width // 2, height // 2), (255, 255, 255),
                   cv2.MARKER_CROSS, 20, 1)
    for sight in sights:
        color = (0, 200, 255) if sight.id == target else (0, 220, 0)
        cv2.polylines(canvas, [sight.corners.astype(np.int32)], True, color, 2)
        cx, cy = int(sight.center[0]), int(sight.center[1])
        cv2.putText(canvas, str(sight.id), (cx - 12, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if sight.id == target:
            cv2.line(canvas, (width // 2, height // 2), (cx, cy), color, 2)
    return canvas
