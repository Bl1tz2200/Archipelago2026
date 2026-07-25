"""Перелёт от метки к метке и стабилизация над меткой.

Никакого метража поля: цель перелёта — не координата, а конкретная ArUco-метка,
а признак «прилетели» — метка в центре кадра. Метры появляются только там, где они
принадлежат самому дрону: его высота из телеметрии при переводе пикселей в смещение
корпуса (`apple_vision.geometry.GroundProjector`).

Три стратегии управления:

  0. `body_step` (штатная) — перелёт как в базовом примере полёта: одна команда
     `navigate(x, y, z=0, frame_id="body")` на шаг сетки и пауза на время перелёта.
     Никаких низкоуровневых setpoint'ов (`set_position`) и никакого непрерывного
     цикла управления: команда ушла — дрон летит сам. Камера при этом продолжает
     работать на «яблоки» и на разметку, но в управление не вмешивается.
  1. `aruco_frame` — `navigate(x=0, y=0, z=alt, frame_id="aruco_<ID>")`: пересчёт делает
     сам Обрик через tf2 (`docs-sverk/obrik-ros-2/24-coordinate-frames.md`).
  2. `visual` — метка ищется в кадре, пиксельное смещение переводится в смещение
     корпуса и отправляется как `navigate(frame_id="body")`. Работает даже если
     фреймы `aruco_<N>` в системе не публикуются.

Зависание (`freeze`/`hold`) во всех стратегиях — тоже обычный `navigate` на нулевое
смещение по корпусу: текущая точка становится целевой.

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

# Сколько ждать ответа `get_telemetry`, прежде чем считать телеметрию недоступной.
TELEMETRY_TIMEOUT_S = 2.0


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
        # Где дрон находится по счислению — нужно `body_step`, чтобы знать, на сколько
        # узлов смещаться. Обновляется через `set_node()` и после каждого перелёта.
        # `_pos` — то же самое, но дробное: зависание может застать дрон между узлами.
        self.node: Optional[Node] = None
        self._pos: Optional[List[float]] = None

        self._frame: Optional[np.ndarray] = None
        self._sights: List[Sight] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._last_command_at = 0.0
        self._telemetry_stuck = False

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
        """Телеметрия со сторожем: без данных от FCU `get_telemetry` не возвращается никогда.

        Убить зависший вызов внутри ROS/DDS нельзя — его можно только бросить (поток
        демонический). Первого зависания достаточно, чтобы больше не спрашивать: на
        `body_step` телеметрия не нужна, а миссия не должна из-за неё стоять.
        """
        if self.drone is None or self._telemetry_stuck:
            return None

        box: dict = {}

        def worker() -> None:
            try:
                box["value"] = self.drone.control.get_telemetry(frame_id=frame_id)
            except BaseException:            # noqa: BLE001 — сюда же ошибки ROS
                box["value"] = None

        thread = threading.Thread(target=worker, name="snake_telemetry", daemon=True)
        thread.start()
        thread.join(TELEMETRY_TIMEOUT_S)
        if thread.is_alive():
            self._telemetry_stuck = True
            self._log(f"телеметрия не отвечает (> {TELEMETRY_TIMEOUT_S:.0f} с) — "
                      "дальше летим без неё")
            return None
        return box.get("value")

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
        if self.strategy == "body_step":
            self._log(f"стратегия перелёта: body_step "
                      f"(navigate по корпусу, шаг сетки {self.nav.step_m:.2f} м)")
            return self.strategy
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

    def set_node(self, node: Optional[Node]) -> None:
        """Отмечает, над каким узлом дрон находится (счисление для `body_step`)."""
        if node is None:
            self.node = None
            self._pos = None
            return
        self.node = (int(node[0]), int(node[1]))
        self._pos = [float(node[0]), float(node[1])]

    def goto(self, node: Node, tolerance: Optional[float] = None,
             timeout: Optional[float] = None) -> NavResult:
        """Перелёт на метку узла `node`. Признак прилёта — метка у центра кадра."""
        if self.strategy == "body_step":
            return self.step_body(node)
        marker_id = self.numbering.marker_id(node)
        tolerance = self.nav.arrive_tolerance if tolerance is None else tolerance
        timeout = self.nav.goto_timeout if timeout is None else timeout
        return self._approach(marker_id, tolerance, timeout, hold_frames=1, settle=False)

    def stabilize(self, node: Node, timeout: Optional[float] = None) -> NavResult:
        """Стабилизация над меткой: центр кадра и низкая скорость несколько кадров подряд."""
        if self.strategy == "body_step":
            # Перелёт уже закончился паузой на торможение — доводить нечем и незачем.
            self.set_node(node)
            return NavResult(True, "arrived", None, 0.0)
        marker_id = self.numbering.marker_id(node)
        timeout = self.nav.stabilize_timeout if timeout is None else timeout
        return self._approach(marker_id, self.nav.center_tolerance, timeout,
                              hold_frames=self.nav.hold_frames, settle=True)

    # ------------------------------------------------------ перелёт по корпусу

    def step_body(self, target: Node) -> NavResult:
        """Перелёт в узел одной командой `navigate(frame_id="body")` — как в примере.

        Смещение считается по сетке от текущего узла: строки идут по «вперёд» корпуса,
        столбцы — по «влево» с минусом (FLU), курс весь полёт постоянный, поэтому
        корпус остаётся сонаправлен полю. Дальше — пауза на время перелёта; метка,
        телеметрия и tf2 в этом пути не участвуют вообще.
        """
        started = time.monotonic()
        if self._pos is None:
            # Откуда летим — неизвестно, смещать не от чего: считаем, что уже на месте.
            self.set_node(target)
            return NavResult(False, "no_origin", None, 0.0)

        dcol = target[0] - self._pos[0]
        drow = target[1] - self._pos[1]
        x = drow * self.nav.step_m
        y = -dcol * self.nav.step_m
        distance = math.hypot(x, y)
        if distance < 1e-6:
            self.set_node(target)
            return NavResult(True, "arrived", None, 0.0)

        if self.drone is None:
            self.set_node(target)
            return NavResult(False, "no_drone", None, time.monotonic() - started)

        try:
            self.drone.control.navigate(
                x=float(x), y=float(y), z=0.0,
                yaw=self.config.flight.yaw, speed=self.config.flight.speed,
                frame_id="body", auto_arm=False,
            )
        except Exception as exc:
            self._log(f"navigate(body) в {target} не прошёл: {exc}")
            return NavResult(False, "command_failed", None, time.monotonic() - started)

        # Ждём столько, сколько занимает перелёт на этой скорости, плюс запас на разгон
        # и торможение. Пауза дробная: «яблоко» должно останавливать дрон посреди
        # перелёта, а не по прилёте в узел (регламент, п. 2.1.2).
        # (`time_scale` — только для симулятора, где время идёт быстрее реального;
        # на дроне он равен единице и на расчёт не влияет.)
        flying = distance / max(self.config.flight.speed, 0.05) / self.nav.time_scale
        if not self._sleep_until(started + flying + self.nav.settle_pause / self.nav.time_scale):
            # Прервались на полпути: положение по счислению — доля пути, пройденная
            # к этому моменту. Без неё следующая команда ушла бы от старого узла и
            # дала бы двойное смещение.
            done = min(1.0, (time.monotonic() - started) / flying)
            self._advance(dcol * done, drow * done)
            return NavResult(False, "paused", None, time.monotonic() - started)

        self.set_node(target)
        return NavResult(True, "arrived", None, time.monotonic() - started)

    def _advance(self, dcol: float, drow: float) -> None:
        """Сдвигает счисляемое положение на долю шага сетки."""
        if self._pos is None:
            return
        self._pos = [self._pos[0] + dcol, self._pos[1] + drow]
        self.node = (int(round(self._pos[0])), int(round(self._pos[1])))

    def _sleep_until(self, deadline: float, tick: float = 0.05) -> bool:
        """Спит до `deadline`. False — если пришлось прерваться по `should_pause()`."""
        while time.monotonic() < deadline:
            if self.should_pause():
                return False
            time.sleep(min(tick, max(0.0, deadline - time.monotonic())))
        return True

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
        """Мгновенное зависание на месте: нулевое смещение по корпусу — текущая точка.

        Раньше здесь был `set_position(frame_id="map")` — низкоуровневый setpoint ноды
        `offboard_control`. Он требует телеметрии map и на площадке дрон по нему не
        летит, поэтому останов идёт тем же `navigate`, что и весь остальной полёт.
        """
        if self.drone is None:
            return False
        try:
            self.drone.control.navigate(
                x=0.0, y=0.0, z=0.0,
                yaw=self.config.flight.yaw, speed=self.config.flight.speed,
                frame_id="body", auto_arm=False,
            )
            return True
        except Exception as exc:
            self._log(f"зависание не прошло: {exc}")
            return False

    def hold(self, duration: float, until: Optional[Callable[[], bool]] = None,
             refresh: float = 0.2) -> bool:
        """Висит на месте `duration` секунд: одна команда останова и ожидание.

        `until` — необязательное условие досрочного выхода (например, «хвост встал в строй»).
        Возвращает True, если вышли по условию, False — если по времени.
        """
        if self.drone is None:
            time.sleep(min(duration, 0.1))
            return False
        self.freeze()
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if until is not None and until():
                return True
            time.sleep(min(refresh, max(0.0, deadline - time.monotonic())))
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
