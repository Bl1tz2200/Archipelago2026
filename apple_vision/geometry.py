"""Перевод пикселя в координаты пола.

Нужен, чтобы записать в лог, где именно лежит «яблоко», и чтобы не засчитать
одно и то же яблоко дважды, когда цепочка пролетает над ним повторно.
На само распознавание не влияет: если интринсиков нет, детектор работает как есть.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import CameraConfig


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0
    source: str = "fov"

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """Пересчёт под другое разрешение кадра (калибровка могла быть в ином)."""
        if not self.width or not self.height or (width == self.width and height == self.height):
            return self
        sx = width / float(self.width)
        sy = height / float(self.height)
        return CameraIntrinsics(
            fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy,
            width=width, height=height, source=self.source,
        )


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> CameraIntrinsics:
    """Грубая оценка матрицы камеры по горизонтальному углу обзора."""
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return CameraIntrinsics(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
                            width=width, height=height, source="fov")


def load_intrinsics(path: str) -> Optional[CameraIntrinsics]:
    """Читает `latest_calibration.yaml` (формат camera_calibration / camera_info)."""
    import yaml

    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None

    matrix = data.get("camera_matrix") or data.get("K") or data.get("k")
    if isinstance(matrix, dict):
        matrix = matrix.get("data")
    if not matrix:
        return None
    flat = [float(v) for v in (matrix if isinstance(matrix[0], (int, float)) else sum(matrix, []))]
    if len(flat) < 9:
        return None

    return CameraIntrinsics(
        fx=flat[0], fy=flat[4], cx=flat[2], cy=flat[5],
        width=int(data.get("image_width", 0) or 0),
        height=int(data.get("image_height", 0) or 0),
        source=path,
    )


class GroundProjector:
    """Проецирует пиксель на пол, считая, что камера смотрит вертикально вниз.

    Наклоны корпуса не компенсируются: на скорости «змейки» и высоте ~1.5 м
    ошибка от крена в пару градусов заметно меньше радиуса дедупликации.
    """

    def __init__(self, camera: CameraConfig, intrinsics: Optional[CameraIntrinsics] = None) -> None:
        self.camera = camera
        self.intrinsics = intrinsics or load_intrinsics(camera.calibration_file)

    def ensure_intrinsics(self, width: int, height: int) -> CameraIntrinsics:
        if self.intrinsics is None:
            self.intrinsics = intrinsics_from_fov(width, height, self.camera.fallback_hfov_deg)
        return self.intrinsics.scaled_to(width, height)

    def pixel_to_body(
        self, pixel: Tuple[float, float], altitude: float, frame_size: Tuple[int, int]
    ) -> Optional[Tuple[float, float]]:
        """Пиксель → смещение относительно дрона в метрах, FLU (x вперёд, y влево)."""
        if altitude is None or altitude <= 0.05:
            return None
        width, height = frame_size
        k = self.ensure_intrinsics(width, height)

        u, v = pixel
        if self.camera.mirror_x:
            u = width - 1 - u

        # Плоскость пола: смещение линейно по высоте.
        right = (u - k.cx) / k.fx * altitude      # вправо по кадру
        forward = -(v - k.cy) / k.fy * altitude   # вверх по кадру = вперёд

        yaw = math.radians(self.camera.yaw_offset_deg)
        if yaw:
            c, s = math.cos(yaw), math.sin(yaw)
            forward, right = forward * c - right * s, forward * s + right * c

        return forward, -right  # y в FLU направлена влево

    def pixel_to_map(
        self,
        pixel: Tuple[float, float],
        telemetry_xy: Tuple[float, float],
        yaw: float,
        altitude: float,
        frame_size: Tuple[int, int],
    ) -> Optional[Tuple[float, float]]:
        """Пиксель → координаты пола во фрейме `map` (ENU)."""
        body = self.pixel_to_body(pixel, altitude, frame_size)
        if body is None:
            return None
        bx, by = body
        c, s = math.cos(yaw), math.sin(yaw)
        return telemetry_xy[0] + bx * c - by * s, telemetry_xy[1] + bx * s + by * c
