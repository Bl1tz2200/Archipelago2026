"""Синтетический вид поля меток сверху — самопроверка без камеры и без полёта.

Рисует кусок сетки 7×7 так, как его видит камера, висящая над заданным узлом:
в центре кадра — метка этого узла, вокруг — соседние. Позволяет прогнать
детектор, выбор стартового маркера и измерение полосы обзора на любой машине.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from . import field
from .field import Node, Numbering


def _marker_image(marker_id: int, size: int, dictionary: str = "DICT_4X4_50") -> np.ndarray:
    attr = getattr(cv2.aruco, dictionary)
    getter = getattr(cv2.aruco, "getPredefinedDictionary", None) or cv2.aruco.Dictionary_get
    aruco_dict = getter(attr)
    draw = getattr(cv2.aruco, "generateImageMarker", None)
    if draw is not None:  # OpenCV 4.7+
        image = draw(aruco_dict, marker_id, size)
    else:  # OpenCV < 4.7
        image = cv2.aruco.drawMarker(aruco_dict, marker_id, size)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def make_view(
    center_node: Node = (3, 3),
    spacing_px: int = 150,
    marker_px: int = 0,
    size: Tuple[int, int] = (640, 480),
    numbering: Numbering = field.DEFAULT_NUMBERING,
    offset_px: Tuple[int, int] = (0, 0),
    brightness: int = 170,
    seed: int = 0,
) -> np.ndarray:
    """Кадр «вид сверху» с меткой `center_node` в центре.

    `spacing_px` задаёт, как далеко метки друг от друга в кадре — то есть высоту:
    чем меньше шаг, тем больше меток попадает в кадр. Размер самой метки по
    умолчанию пропорционален шагу — как на настоящем поле, где метки не наезжают
    друг на друга ни с какой высоты.
    `offset_px` сдвигает всю картинку — имитация того, что дрон висит не точно
    над меткой (нужно для проверки стабилизации).
    """
    marker_px = int(marker_px) if marker_px else max(24, int(spacing_px * 0.6))
    rng = np.random.default_rng(seed)
    width, height = size
    frame = np.full((height, width, 3), brightness, np.uint8)
    frame = np.clip(frame.astype(np.float32) + rng.normal(0, 5, frame.shape), 0, 255).astype(np.uint8)

    ccol, crow = center_node
    reach = int(max(width, height) / spacing_px) + 1
    for drow in range(-reach, reach + 1):
        for dcol in range(-reach, reach + 1):
            node = (ccol + dcol, crow + drow)
            if not field.inside(node):
                continue
            # Ось Y кадра направлена вниз, ось row поля — вверх, отсюда минус.
            x = int(width / 2 + dcol * spacing_px + offset_px[0] - marker_px / 2)
            y = int(height / 2 - drow * spacing_px + offset_px[1] - marker_px / 2)
            if x + marker_px < 0 or y + marker_px < 0 or x >= width or y >= height:
                continue
            tile = _marker_image(numbering.marker_id(node), marker_px)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + marker_px), min(height, y + marker_px)
            frame[y0:y1, x0:x1] = tile[y0 - y:y1 - y, x0 - x:x1 - x]
    return frame


def make_apple_view(
    center_node: Node = (3, 3),
    apple_color: Optional[str] = "red",
    apple_offset: Tuple[int, int] = (140, -90),
    **kwargs,
) -> np.ndarray:
    """То же поле меток плюс «яблоко» на полу — сквозная проверка двух детекторов."""
    from apple_vision.synthetic import APPLE_BGR  # локальный импорт: нужен только здесь

    frame = make_view(center_node=center_node, **kwargs)
    if apple_color:
        height, width = frame.shape[:2]
        center = (width // 2 + apple_offset[0], height // 2 + apple_offset[1])
        bgr = APPLE_BGR.get(apple_color, (200, 200, 200))
        cv2.circle(frame, center, 30, bgr, -1)
        cv2.circle(frame, (center[0] - 10, center[1] - 10), 6,
                   tuple(min(255, c + 90) for c in bgr), -1)
    return frame
