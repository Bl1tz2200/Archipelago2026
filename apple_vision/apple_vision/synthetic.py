"""Синтетические кадры — самопроверка детектора без камеры и без полёта.

Рисует пол с шумом, ArUco-подобные чёрно-белые квадраты и цветные «яблоки»,
чтобы можно было прогнать всю цепочку (детектор → зачёт) на любой машине.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Цвета «яблок» в BGR, примерно соответствующие профилям конфига.
APPLE_BGR: Dict[str, Tuple[int, int, int]] = {
    "red": (40, 40, 200),
    "green": (60, 170, 60),
    "yellow": (40, 210, 230),
}


def _floor(width: int, height: int, brightness: int, rng: np.random.Generator) -> np.ndarray:
    frame = np.full((height, width, 3), brightness, np.uint8)
    noise = rng.normal(0, 6, (height, width, 3))
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # Пара чёрно-белых квадратов — имитация ArUco-поля на полу.
    for cx, cy in ((int(width * 0.18), int(height * 0.2)), (int(width * 0.8), int(height * 0.78))):
        size = int(min(width, height) * 0.12)
        cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), (250, 250, 250), -1)
        cv2.rectangle(frame, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2),
                      (10, 10, 10), -1)
    return frame


def _draw_apple(frame: np.ndarray, center: Tuple[int, int], radius: int,
                bgr: Tuple[int, int, int]) -> None:
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, bgr, -1)
    cv2.addWeighted(overlay, 0.97, frame, 0.03, 0, frame)
    # Блик и затенение — как у настоящего яблока под лампами зала.
    highlight = (min(255, bgr[0] + 90), min(255, bgr[1] + 90), min(255, bgr[2] + 90))
    cv2.circle(frame, (center[0] - radius // 3, center[1] - radius // 3),
               max(2, radius // 5), highlight, -1)
    cv2.circle(frame, center, radius, tuple(int(c * 0.6) for c in bgr), 2)


def make_frame(
    apples: Sequence[Tuple[str, Tuple[int, int], int]] = (),
    size: Tuple[int, int] = (640, 480),
    brightness: int = 150,
    seed: int = 0,
    distractors: bool = True,
) -> np.ndarray:
    """Кадр с «яблоками»: список (цвет, центр, радиус)."""
    rng = np.random.default_rng(seed)
    width, height = size
    frame = _floor(width, height, brightness, rng)

    if distractors:
        # Вытянутая цветная полоса и тонкая линия: по цвету похожи на яблоко,
        # по форме — нет. Фильтры circularity/fill обязаны их отбросить.
        cv2.rectangle(frame, (int(width * 0.05), int(height * 0.55)),
                      (int(width * 0.42), int(height * 0.62)), (50, 50, 190), -1)
        cv2.line(frame, (0, int(height * 0.9)), (width, int(height * 0.86)), (60, 180, 60), 5)

    for color, center, radius in apples:
        _draw_apple(frame, center, radius, APPLE_BGR.get(color, (200, 200, 200)))
    return frame


def make_flight(
    colors: Sequence[str] = ("red", "green", "yellow"),
    frames_per_apple: int = 10,
    gap_frames: int = 5,
    size: Tuple[int, int] = (640, 480),
    seed: int = 0,
) -> List[Tuple[np.ndarray, Optional[str]]]:
    """Имитация пролёта: яблоки появляются по очереди, между ними — пустые кадры.

    Возвращает список (кадр, ожидаемый цвет или None).
    """
    width, height = size
    sequence: List[Tuple[np.ndarray, Optional[str]]] = []
    for i, color in enumerate(colors):
        for _ in range(gap_frames):
            sequence.append((make_frame(size=size, seed=seed + len(sequence)), None))
        for k in range(frames_per_apple):
            # Яблоко «проплывает» по кадру, как при движении дрона над ним.
            t = k / max(1, frames_per_apple - 1)
            cx = int(width * (0.3 + 0.4 * t))
            cy = int(height * (0.5 + 0.1 * math.sin(t * math.pi)))
            frame = make_frame([(color, (cx, cy), 34)], size=size, seed=seed + len(sequence))
            sequence.append((frame, color))
    return sequence
