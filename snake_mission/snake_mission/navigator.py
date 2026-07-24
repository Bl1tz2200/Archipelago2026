"""Перелёт от метки к метке и стабилизация над меткой.

Никакого метража поля: цель перелёта — не координата, а конкретная ArUco-метка,
а признак «прилетели» — метка в центре кадра. Метры появляются только там, где они
принадлежат самому дрону: его высота из телеметрии при переводе пикселей в смещение
корпуса (`apple_vision.geometry.GroundProjector`).

Две стратегии управления, выбор автоматический:

  1. `aruco_frame` — `navigate(x=0, y=0, z=alt, frame_id="aruco_<ID>")`: пересчёт делает
     сам Обрик через tf2 (`docs-sverk/obrik-ros-2/24-coordinate-frames.md`).
  2. `visual` — метка ищется в кадре, пиксельное смещение переводится в смещение
     корпуса и отправляется как `navigate(frame_id="body")`. Работает даже если
     фреймы `aruco_<N>` в системе не публикуются.

Все циклы управления на каждой итерации спрашивают `should_pause()`. Поэтому событие
«яблоко» останавливает дрон посреди перелёта, а не по прилёте в узел, — как требует
регламент, п. 2.1.2: «цепочка останавливается и зависает на месте».
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from . import markers
from .config import MissionConfig
from .field import Node, Numbering
from .markers import MarkerDetector, Sight

PauseCheck = Callable[[], bool]


@dataclass
class NavResult:
    """Чем закончился перелёт или стабилизация."""

    ok: bool
    reason: str                      # arrived | paused | timeout | no_marker | no_frame
    offset: Optional[float] = None   # смещение метки от центра кадра, доля диагонали
    elapsed: float = 0.0

    @property
    def paused(self) -> bool:
        return self.reason == "paused"

    def __bool__(self) -> bool:
        return self.ok


class MarkerNavigator:
    """Управление полётом по меткам поля.

    Кадры не запрашивает сам — их отдаёт миссия через `feed()`, чтобы на камеру была
    ровно одна подписка на всю программу (её же слушает распознавание «яблок»).
    """

    def __init__(
        self,
        drone,
        config: MissionConfig,
        detector: Optional[MarkerDetector] = None,
        projector=None,
        numbering: Optional[Numbering] = None,
        should_pause: Optional[PauseCheck] = None,
        verbose: bool = True,
    ) -> None:
        self.drone = drone
        self.config = config
        self.nav = config.navigation
        self.detector = detector or MarkerDetector(
            config.markers.dictionary, config.markers.min_marker_percent
        )
        self.numbering = numbering or config.markers.numbering()
        self.projector = projector
        self.should_pause = should_pause or (lambda: False)
        self.verbose = verbose

        self.strategy = config.navigation.strategy
        self.last_sights: List[Sight] = []
        self.frames_seen = 0

        self._frame: Optional[np.ndarray] = None
        self._sights: List[Sight] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._last_command_at = 0.0

    # --------------------------------------------------------------- кадры

    def feed(self, frame: np.ndarray) -> List[Sight]:
        """Принимает очередной кадр с камеры и обновляет список видимых меток."""
        if frame is None:
            return []
        sights = self.detector.detect(frame)
        with self._lock:
            self._frame = frame
            self._sights = sights
            self._seq += 1
            self.frames_seen += 1
        self.last_sights = sights
        return sights

    def snapshot(self) -> Tuple[Optional[np.ndarray], List[Sight], int]:
        with self._lock:
            return self._frame, list(self._sights), self._seq

    def wait_frame(self, after: int = -1, timeout: Optional[float] = None
                   ) -> Tuple[Optional[np.ndarray], List[Sight], int]:
        """Ждёт кадр новее `after`. Возвращает (кадр, метки, номер кадра)."""
        deadline = time.monotonic() + (timeout if timeout is not None else self.nav.frame_timeout)
        while True:
            frame, sights, seq = self.snapshot()
            if seq > after and frame is not None:
                return frame, sights, seq
            if time.monotonic() >= deadline:
                return None, [], after
            time.sleep(0.02)

    # ---------------------------------------------------------- телеметрия

    def telemetry(self, frame_id: str = "map"):
        if self.drone is None:
            return None
        try:
            return self.drone.control.get_telemetry(frame_id=frame_id)
        except Exception:
            return None

    def altitude(self) -> Optional[float]:
        t = self.telemetry("terrain") or self.telemetry("map")
        try:
            return float(t.z) if t is not None else None
        except (TypeError, ValueError):
            return None

    def horizontal_speed(self) -> Optional[float]:
        t = self.telemetry("map")
        try:
            return math.hypot(float(t.vx), float(t.vy)) if t is not None else None
        except (TypeError, ValueError, AttributeError):
            return None

    # ---------------------------------------------------------- стратегия

    def probe_strategy(self) -> str:
        """Определяет, доступны ли фреймы `aruco_<N>`; фиксирует стратегию на всю миссию."""
        if self.strategy in ("aruco_frame", "visual"):
            return self.strategy
        if self.drone is None:
            self.strategy = "visual"
            return self.strategy

        _, sights, _ = self.wait_frame(timeout=2.0)
        for sight in sights[:3]:
            t = self.telemetry(f"aruco_{sight.id}")
            if t is None:
                continue
            try:
                if all(math.isfinite(float(v)) for v in (t.x, t.y, t.z)):
                    self.strategy = "aruco_frame"
                    self._log(f"стратегия перелёта: aruco_frame (фрейм aruco_{sight.id} доступен)")
                    return self.strategy
            except (TypeError, ValueError, AttributeError):
                continue

        self.strategy = "visual"
        self._log("стратегия перелёта: visual (фреймы aruco_<N> недоступны, наводимся по кадру)")
        return self.strategy

    # ------------------------------------------------------------ команды

    def _command_marker_frame(self, marker_id: int) -> bool:
        if self.drone is None:
            return False
        try:
            self.drone.control.navigate(
                x=0.0, y=0.0, z=self.config.flight.altitude,
                yaw=self.config.flight.yaw, speed=self.config.flight.speed,
                frame_id=f"aruco_{marker_id}",
            )
            return True
        except Exception as exc:
            self._log(f"navigate(aruco_{marker_id}) не прошёл: {exc}")
            return False

    def _command_visual(self, sight: Sight, frame_size: Tuple[int, int]) -> bool:
        """Доводка по кадру: смещение метки в пикселях → смещение корпуса в метрах."""
        if self.drone is None or self.projector is None:
            return False
        altitude = self.altitude()
        if altitude is None:
            return False
        body = self.projector.pixel_to_body(sight.center, altitude, frame_size)
        if body is None:
            return False
        forward, left = body
        try:
            self.drone.control.navigate(
                x=float(forward), y=float(left), z=0.0,
                yaw=self.config.flight.yaw, speed=self.config.flight.speed,
                frame_id="body",
            )
            return True
        except Exception as exc:
            self._log(f"navigate(body) не прошёл: {exc}")
            return False

    def _steer(self, marker_id: int, sight: Optional[Sight],
               frame_size: Optional[Tuple[int, int]], force: bool = False) -> None:
        """Одна команда управления в сторону метки — по выбранной стратегии.

        Команды отправляются не чаще `poll_interval`: кадры приходят десятками в
        секунду, а сервис навигации дёргать с такой частотой незачем.
        """
        now = time.monotonic()
        if not force and now - self._last_command_at < self.nav.poll_interval:
            return
        self._last_command_at = now

        if self.strategy == "aruco_frame":
            if self._command_marker_frame(marker_id):
                return
        if sight is not None and frame_size is not None:
            self._command_visual(sight, frame_size)

    # -------------------------------------------------------------- полёт

    def goto(self, node: Node, tolerance: Optional[float] = None,
             timeout: Optional[float] = None) -> NavResult:
        """Перелёт на метку узла `node`. Признак прилёта — метка у центра кадра."""
        marker_id = self.numbering.marker_id(node)
        tolerance = self.nav.arrive_tolerance if tolerance is None else tolerance
        timeout = self.nav.goto_timeout if timeout is None else timeout
        return self._approach(marker_id, tolerance, timeout, hold_frames=1, settle=False)

    def stabilize(self, node: Node, timeout: Optional[float] = None) -> NavResult:
        """Стабилизация над меткой: центр кадра и низкая скорость несколько кадров подряд."""
        marker_id = self.numbering.marker_id(node)
        timeout = self.nav.stabilize_timeout if timeout is None else timeout
        return self._approach(marker_id, self.nav.center_tolerance, timeout,
                              hold_frames=self.nav.hold_frames, settle=True)

    def _approach(self, marker_id: int, tolerance: float, timeout: float,
                  hold_frames: int, settle: bool) -> NavResult:
        started = time.monotonic()
        deadline = started + timeout
        seq = -1
        in_tolerance = 0
        last_seen = started
        last_offset: Optional[float] = None
        commanded = False

        while True:
            if self.should_pause():
                return NavResult(False, "paused", last_offset, time.monotonic() - started)
            if time.monotonic() >= deadline:
                reason = "timeout" if last_offset is not None else "no_marker"
                return NavResult(False, reason, last_offset, time.monotonic() - started)

            frame, sights, seq = self.wait_frame(after=seq, timeout=self.nav.frame_timeout)
            if frame is None:
                if not commanded:
                    # Кадров нет — хотя бы отправим команду, чтобы дрон не стоял.
                    self._steer(marker_id, None, None, force=True)
                    commanded = True
                continue

            frame_size = (frame.shape[1], frame.shape[0])
            sight = markers.find(sights, marker_id)

            if sight is None:
                if time.monotonic() - last_seen > self.nav.lost_marker_timeout:
                    return NavResult(False, "no_marker", last_offset, time.monotonic() - started)
                self._steer(marker_id, None, None)
                in_tolerance = 0
                continue

            last_seen = time.monotonic()
            _, _, last_offset = markers.offset_from_center(sight, frame_size)

            if last_offset <= tolerance and (not settle or self._settled()):
                in_tolerance += 1
                if in_tolerance >= hold_frames:
                    return NavResult(True, "arrived", last_offset, time.monotonic() - started)
                continue

            in_tolerance = 0
            # Первая команда на цель уходит сразу, дальше — не чаще poll_interval.
            self._steer(marker_id, sight, frame_size, force=not commanded)
            commanded = True

    def _settled(self) -> bool:
        speed = self.horizontal_speed()
        return speed is None or speed <= self.nav.settle_speed

    # ------------------------------------------------------------ удержание

    def freeze(self) -> bool:
        """Мгновенное зависание на месте: текущая точка становится целевой."""
        if self.drone is None:
            return False
        t = self.telemetry("map")
        if t is None:
            return False
        try:
            self.drone.control.set_position(
                x=float(t.x), y=float(t.y), z=float(t.z),
                yaw=float(getattr(t, "yaw", self.config.flight.yaw) or 0.0),
                frame_id="map",
            )
            return True
        except Exception as exc:
            self._log(f"зависание не прошло: {exc}")
            return False

    def hold(self, duration: float, until: Optional[Callable[[], bool]] = None,
             refresh: float = 0.5) -> bool:
        """Держит текущую точку `duration` секунд, обновляя цель.

        `until` — необязательное условие досрочного выхода (например, «хвост встал в строй»).
        Возвращает True, если вышли по условию, False — если по времени.
        """
        if self.drone is None:
            time.sleep(min(duration, 0.1))
            return False
        t = self.telemetry("map")
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if until is not None and until():
                return True
            if t is not None:
                try:
                    self.drone.control.set_position(
                        x=float(t.x), y=float(t.y), z=float(t.z),
                        yaw=float(getattr(t, "yaw", self.config.flight.yaw) or 0.0),
                        frame_id="map",
                    )
                except Exception:
                    pass
            time.sleep(refresh)
        return bool(until is not None and until())

    # ------------------------------------------------------------- разное

    def visible_span(self, sights: Optional[Sequence[Sight]] = None) -> Tuple[int, int]:
        """Полоса обзора в метках по последнему кадру."""
        frame, last, _ = self.snapshot()
        size = (frame.shape[1], frame.shape[0]) if frame is not None else None
        return markers.visible_span(last if sights is None else sights, self.numbering, size)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[navigator] {message}", flush=True)
