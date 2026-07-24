"""Поиск «яблок» в кадре по цвету и форме.

Чистый OpenCV/numpy — ни ROS, ни sverk_interfaces здесь не нужны, поэтому
детектор одинаково работает на дроне, на записанном видео и в тестах.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import (
    AppleVisionConfig,
    ColorProfile,
    DetectorConfig,
    enabled_profiles,
    load_config,
)


@dataclass
class Detection:
    """Одно найденное «яблоко» в кадре (координаты — в пикселях исходного кадра)."""

    color: str
    label: str
    center: Tuple[float, float]
    radius: float
    area: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    circularity: float
    solidity: float
    fill: float
    score: float
    draw_bgr: Tuple[int, int, int] = (255, 255, 255)
    contour: Optional[np.ndarray] = field(default=None, repr=False)

    def distance_to(self, other: "Detection | Tuple[float, float]") -> float:
        p = other.center if isinstance(other, Detection) else other
        return math.hypot(self.center[0] - p[0], self.center[1] - p[1])


def _odd(value: int) -> int:
    """Ядра OpenCV требуют нечётного размера."""
    value = int(value)
    return value if value % 2 == 1 else value + 1


class AppleDetector:
    """Детектор «яблок» по цветовым профилям из конфига."""

    def __init__(
        self,
        config: AppleVisionConfig | str | None = None,
        colors: Sequence[str] = (),
    ) -> None:
        if config is None or isinstance(config, str):
            config = load_config(config) if config else load_config()
        self.config: AppleVisionConfig = config
        self.params: DetectorConfig = config.detector
        self.profiles: List[ColorProfile] = enabled_profiles(config, colors)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd(self.params.morph_ksize), _odd(self.params.morph_ksize))
        )
        self.last_masks: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- маски

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Уменьшение + сглаживание. Возвращает кадр и масштаб (исходный/рабочий)."""
        scale = 1.0
        work = frame
        target = int(self.params.resize_width)
        if target and frame.shape[1] > target:
            scale = frame.shape[1] / float(target)
            height = max(1, int(round(frame.shape[0] / scale)))
            work = cv2.resize(frame, (target, height), interpolation=cv2.INTER_AREA)
        if self.params.blur_ksize and self.params.blur_ksize >= 3:
            work = cv2.medianBlur(work, _odd(self.params.blur_ksize))
        return work, scale

    def color_mask(self, hsv: np.ndarray, profile: ColorProfile) -> np.ndarray:
        """Бинарная маска пикселей профиля (объединение всех его HSV-диапазонов)."""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in profile.ranges:
            mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        if self.params.morph_iterations > 0:
            it = int(self.params.morph_iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=it)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=it)
        return mask

    # ------------------------------------------------------------ детекция

    def _contour_detection(
        self, contour: np.ndarray, mask: np.ndarray, profile: ColorProfile, scale: float
    ) -> Optional[Detection]:
        area = float(cv2.contourArea(contour))
        area_full = area * scale * scale  # площадь в пикселях исходного кадра
        if area_full < profile.min_area or area_full > profile.max_area:
            return None

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            return None
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < profile.min_circularity:
            return None

        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < profile.min_solidity:
            return None

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circle_area = math.pi * radius * radius
        # Насколько плотно маска заполняет описанную окружность: у яблока — почти
        # полностью, у вытянутой полосы или ломаного пятна — заметно меньше.
        fill = area / circle_area if circle_area > 0 else 0.0
        if fill < profile.min_fill:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        score = float(np.clip(0.4 * circularity + 0.3 * solidity + 0.3 * fill, 0.0, 1.0))
        if score < self.params.min_score:
            return None

        return Detection(
            color=profile.name,
            label=profile.label,
            center=(cx * scale, cy * scale),
            radius=radius * scale,
            area=area_full,
            bbox=(int(x * scale), int(y * scale), int(w * scale), int(h * scale)),
            circularity=circularity,
            solidity=solidity,
            fill=fill,
            score=score,
            draw_bgr=profile.draw_bgr,
            contour=(contour.astype(np.float32) * scale).astype(np.int32),
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Возвращает найденные «яблоки», отсортированные по убыванию площади."""
        if frame is None or frame.size == 0:
            return []

        work, scale = self._preprocess(frame)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

        found: List[Detection] = []
        self.last_masks = {}
        for profile in self.profiles:
            mask = self.color_mask(hsv, profile)
            self.last_masks[profile.name] = mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            per_color: List[Detection] = []
            for contour in contours:
                det = self._contour_detection(contour, mask, profile, scale)
                if det is not None:
                    per_color.append(det)
            per_color.sort(key=lambda d: d.area, reverse=True)
            limit = max(1, int(self.params.max_per_color))
            found.extend(per_color[:limit])

        found = self._suppress_overlaps(found)
        found.sort(key=lambda d: d.area, reverse=True)
        return found

    @staticmethod
    def _suppress_overlaps(detections: List[Detection]) -> List[Detection]:
        """Один объект мог попасть в два соседних профиля (красный/жёлтый) — оставляем лучший."""
        kept: List[Detection] = []
        for det in sorted(detections, key=lambda d: (d.score, d.area), reverse=True):
            overlap = False
            for other in kept:
                if det.color == other.color:
                    continue
                if det.distance_to(other) < max(det.radius, other.radius):
                    overlap = True
                    break
            if not overlap:
                kept.append(det)
        return kept
