"""Симулятор дрона и поля: прогон всей миссии без железа.

Подменяет объект `sverk_interfaces.init(...)`: те же `drone.control`, `drone.image`,
но полёт считается на месте, а кадры рисуются — поле меток 7×7 плюс «яблоки» на полу.
Через него прогоняется вся цепочка целиком: поток кадров → распознавание меток и
«яблок» → навигация → зависания → возврат и посадка.

Метраж здесь появляется впервые и только внутри симулятора: чтобы нарисовать кадр,
нужно знать, с какой высоты и с каким шагом смотрит камера. Сама миссия по-прежнему
не знает ни одного метра.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field as dc_field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import field
from .field import Node, Numbering
from .synthetic import make_view

# Ось столбцов поля — это map X, ось строк — map Y (ENU, как в ROS).
# Нос дрона в симуляторе смотрит вдоль строк, то есть курс равен +90°: тогда
# «вперёд» по корпусу — это верх кадра, а «влево» — левый край, и пересчёт
# пикселя в координаты пола (`apple_vision.geometry`) даёт настоящие координаты
# яблока в map. На площадке ту же роль играет `camera.yaw_offset_deg`.
HEADING = math.pi / 2


@dataclass
class SimWorld:
    """Физика площадки: шаг меток, размер меток и яблок, угол обзора камеры."""

    step_m: float = 1.0              # расстояние между соседними метками
    marker_m: float = 0.3            # сторона метки
    apple_radius_m: float = 0.09     # радиус «яблока» на полу
    hfov_deg: float = 65.0           # горизонтальный угол обзора камеры
    frame_size: Tuple[int, int] = (640, 480)
    speed: float = 0.6               # скорость дрона, м/с
    climb_speed: float = 0.8         # скорость набора высоты, м/с
    fps: float = 20.0                # частота кадров симулятора
    time_scale: float = 6.0          # во сколько раз симуляция быстрее реального времени
    apples: Dict[str, Tuple[float, float]] = dc_field(default_factory=dict)

    def focal_px(self) -> float:
        width = self.frame_size[0]
        return (width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)


class _Telemetry:
    def __init__(self, x: float, y: float, z: float, yaw: float,
                 vx: float, vy: float, vz: float) -> None:
        self.x, self.y, self.z, self.yaw = x, y, z, yaw
        self.vx, self.vy, self.vz = vx, vy, vz
        self.connected = True
        self.armed = z > 0.05
        self.mode = "OFFBOARD" if z > 0.05 else "MANUAL"
        self.voltage = 12.4
        self.lat = self.lon = self.alt = 0.0


class SimControl:
    def __init__(self, sim: "SimDrone") -> None:
        self.sim = sim
        self.commands = 0
        self.landed = False

    def navigate(self, x=0.0, y=0.0, z=0.0, yaw=0.0, speed=0.0,
                 frame_id="map", auto_arm=False, **_) -> None:
        self.commands += 1
        self.sim.command(frame_id, x, y, z, auto_arm)

    def navigate_wait(self, x=0.0, y=0.0, z=0.0, timeout=30.0, **kwargs) -> None:
        self.navigate(x=x, y=y, z=z, **{k: v for k, v in kwargs.items()
                                        if k in ("yaw", "speed", "frame_id", "auto_arm")})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.sim.at_target():
            time.sleep(0.05)

    def set_position(self, x=0.0, y=0.0, z=0.0, yaw=0.0, frame_id="map", **_) -> None:
        self.commands += 1
        self.sim.command(frame_id, x, y, z, False)

    def get_telemetry(self, frame_id="map"):
        return self.sim.telemetry(frame_id)

    def set_altitude(self, z=1.0, frame_id="terrain", **_) -> None:
        self.sim.command("altitude", 0.0, 0.0, z, False)

    def land(self):
        self.landed = True
        self.sim.land()
        return type("Resp", (), {"success": True, "message": "landed"})()


class SimImage:
    def __init__(self, sim: "SimDrone") -> None:
        self.sim = sim
        self.published = 0
        self._stop = threading.Event()

    def take_picture(self, timeout: float = 2.0, raw: bool = False):
        return self.sim.render()

    def stream(self, callback: Callable[[np.ndarray], None], duration: Optional[float] = None):
        self._stop.clear()
        period = 1.0 / (self.sim.world.fps * self.sim.world.time_scale)
        deadline = None if duration is None else time.monotonic() + duration
        while not self._stop.is_set():
            if deadline is not None and time.monotonic() > deadline:
                break
            callback(self.sim.render())
            time.sleep(period)

    def stop_stream(self) -> None:
        self._stop.set()

    def publish(self, image: np.ndarray) -> None:
        self.published += 1
        self.sim.last_overlay = image


class SimDrone:
    """Дрон, поле и камера в одном объекте — подставляется вместо `sverk_interfaces.init()`."""

    def __init__(
        self,
        world: Optional[SimWorld] = None,
        start: Node = (0, 0),
        numbering: Optional[Numbering] = None,
        apples: Sequence[Tuple[str, Tuple[float, float]]] = (),
    ) -> None:
        self.world = world or SimWorld()
        self.numbering = numbering or field.DEFAULT_NUMBERING
        self.control = SimControl(self)
        self.image = SimImage(self)
        self.last_overlay: Optional[np.ndarray] = None

        self.pos = [float(start[0]), float(start[1])]     # в шагах сетки
        self.altitude = 0.0
        self.target = list(self.pos)
        self.target_alt = 0.0
        self.velocity = [0.0, 0.0, 0.0]
        self.apples = {name: (float(c), float(r)) for name, (c, r) in apples}
        self.track: List[Tuple[float, float]] = []
        self.frames = 0
        self.closed = False

        self._lock = threading.Lock()
        self._last_tick = time.monotonic()

    # --------------------------------------------------------------- команды

    def command(self, frame_id: str, x: float, y: float, z: float, auto_arm: bool) -> None:
        with self._lock:
            self._tick()
            step = self.world.step_m
            if frame_id == "body":
                # FLU: x вперёд (ось строк), y влево (ось столбцов, в минус).
                self.target = [self.pos[0] - y / step, self.pos[1] + x / step]
                if z:
                    self.target_alt = max(0.0, self.altitude + z)
            elif frame_id == "map":
                self.target = [x / step, y / step]
                self.target_alt = z
            elif frame_id == "altitude":
                self.target_alt = z
            elif frame_id.startswith("aruco_"):
                marker_id = int(frame_id.split("_", 1)[1])
                node = self.numbering.node_of(marker_id)
                self.target = [float(node[0]) + x / step, float(node[1]) + y / step]
                self.target_alt = z
            else:                                   # aruco_map и прочее — как map
                self.target = [x / step, y / step]
                self.target_alt = z
            if auto_arm and self.target_alt <= 0:
                self.target_alt = z

    def at_target(self, tolerance: float = 0.05) -> bool:
        with self._lock:
            self._tick()
            return math.dist(self.pos, self.target) <= tolerance

    def land(self) -> None:
        with self._lock:
            self.target_alt = 0.0

    def close(self) -> None:
        self.closed = True
        self.image.stop_stream()

    # ---------------------------------------------------------------- физика

    def _tick(self) -> None:
        now = time.monotonic()
        dt = (now - self._last_tick) * self.world.time_scale
        self._last_tick = now
        if dt <= 0:
            return

        step = self.world.step_m
        dx = (self.target[0] - self.pos[0]) * step
        dy = (self.target[1] - self.pos[1]) * step
        distance = math.hypot(dx, dy)
        travel = min(distance, self.world.speed * dt)
        if distance > 1e-6:
            self.pos[0] += dx / distance * travel / step
            self.pos[1] += dy / distance * travel / step
            self.velocity[0] = dx / distance * self.world.speed
            self.velocity[1] = dy / distance * self.world.speed
        else:
            self.velocity[0] = self.velocity[1] = 0.0

        dz = self.target_alt - self.altitude
        climb = min(abs(dz), self.world.climb_speed * dt)
        self.altitude += math.copysign(climb, dz) if abs(dz) > 1e-6 else 0.0
        self.velocity[2] = math.copysign(self.world.climb_speed, dz) if abs(dz) > 0.02 else 0.0

    def telemetry(self, frame_id: str = "map") -> _Telemetry:
        with self._lock:
            self._tick()
            step = self.world.step_m
            x, y = self.pos[0] * step, self.pos[1] * step
            if frame_id.startswith("aruco_"):
                node = self.numbering.node_of(int(frame_id.split("_", 1)[1]))
                x -= node[0] * step
                y -= node[1] * step
            return _Telemetry(x, y, self.altitude, HEADING, *self.velocity)

    # ----------------------------------------------------------------- камера

    def render(self) -> Optional[np.ndarray]:
        with self._lock:
            self._tick()
            altitude = self.altitude
            pos = tuple(self.pos)
        if altitude < 0.15:
            return None                     # на земле камера смотрит в пол вплотную

        focal = self.world.focal_px()
        spacing = focal * self.world.step_m / altitude
        marker_px = max(12, int(focal * self.world.marker_m / altitude))
        center_node = (int(round(pos[0])), int(round(pos[1])))
        center_node = (min(max(center_node[0], 0), field.SIDE - 1),
                       min(max(center_node[1], 0), field.SIDE - 1))
        offset = (int(-(pos[0] - center_node[0]) * spacing),
                  int((pos[1] - center_node[1]) * spacing))

        frame = make_view(center_node=center_node, spacing_px=int(spacing),
                          marker_px=marker_px, size=self.world.frame_size,
                          numbering=self.numbering, offset_px=offset, seed=self.frames)
        self._draw_apples(frame, pos, spacing, altitude)
        self.frames += 1
        self.track.append(pos)
        return frame

    def _draw_apples(self, frame: np.ndarray, pos: Tuple[float, float],
                     spacing: float, altitude: float) -> None:
        from apple_vision.synthetic import APPLE_BGR

        height, width = frame.shape[:2]
        radius = max(3, int(self.world.focal_px() * self.world.apple_radius_m / altitude))
        for color, (acol, arow) in self.apples.items():
            x = int(width / 2 + (acol - pos[0]) * spacing)
            y = int(height / 2 - (arow - pos[1]) * spacing)
            if not (-radius < x < width + radius and -radius < y < height + radius):
                continue
            bgr = APPLE_BGR.get(color, (200, 200, 200))
            cv2.circle(frame, (x, y), radius, bgr, -1)
            cv2.circle(frame, (x - radius // 3, y - radius // 3), max(1, radius // 5),
                       tuple(min(255, c + 90) for c in bgr), -1)
            cv2.circle(frame, (x, y), radius, tuple(int(c * 0.6) for c in bgr), 1)


def default_apples() -> List[Tuple[str, Tuple[float, float]]]:
    """Три «яблока» в разных концах поля — как их раскладывают организаторы."""
    return [("red", (1.5, 0.5)), ("green", (4.5, 3.5)), ("yellow", (2.5, 5.5))]
