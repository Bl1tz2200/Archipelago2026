"""Миссия лидера: взлёт → чтение задания → поиск «яблок» → возврат → посадка.

Порядок и все паузы продиктованы регламентом:

  * п. 2.1.1 — лидер взлетает первым, считывает ArUco-маркер стартового угла,
    печатает ID и название фигуры в терминал (это условие начисления баллов);
  * п. 2.1.2 — при распознавании каждого «яблока» цепочка **останавливается и
    зависает**, с земли взлетает следующий дрон, движение продолжается только после
    того, как он встал в хвост;
  * п. 2.1.3 — на сборку формации не более 3 минут, поиск идёт перелётами между
    соседними узлами сетки на постоянной высоте, после сборки лидер приводит цепочку
    на стартовую позицию.

Кадры с камеры берутся один раз и раздаются обоим потребителям — распознаванию
«яблок» (`apple_vision`) и навигации по меткам, чтобы на камеру была ровно одна подписка.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field as dc_field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from apple_vision import AppleVision, GroundProjector, load_config as load_apples_config
from apple_vision.overlay import draw_detections

from . import field, markers, search
from .config import ALTITUDE_CEILING_M, MissionConfig, load_config
from .field import Node
from .markers import MarkerDetector
from .navigator import MarkerNavigator
from .search import SearchPlan
from .swarm import ConsoleSwarmLink, SwarmLink


class FrameHub:
    """Одна подписка на камеру, несколько потребителей кадра."""

    def __init__(self, drone, consumers: Sequence[Callable[[np.ndarray], None]]) -> None:
        self.drone = drone
        self.consumers = list(consumers)
        self.frames = 0
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def _on_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        self.frames += 1
        for consumer in self.consumers:
            try:
                consumer(frame)
            except Exception as exc:   # один потребитель не должен ронять поток кадров
                print(f"[mission] ошибка обработки кадра: {exc}", flush=True)

    def start(self) -> None:
        if self._running or self.drone is None:
            return
        self._running = True

        def worker() -> None:
            try:
                self.drone.image.stream(self._on_frame)
            except Exception as exc:
                print(f"[mission] поток кадров остановлен: {exc}", flush=True)
            finally:
                self._running = False

        self._thread = threading.Thread(target=worker, name="snake_frames", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self.drone is None:
            return
        try:
            self.drone.image.stop_stream()
        except Exception:
            pass


@dataclass
class MissionResult:
    start_marker: Optional[int] = None
    figure: Optional[str] = None
    apples: int = 0
    steps_done: int = 0
    elapsed: float = 0.0
    returned: bool = False
    landed: bool = False
    notes: List[str] = dc_field(default_factory=list)

    def describe(self) -> str:
        lines = [
            "═" * 62,
            "ИТОГ ПОПЫТКИ",
            f"  стартовый маркер: {self.start_marker if self.start_marker is not None else '—'}",
            f"  назначенная фигура: {self.figure or '—'}",
            f"  засчитано яблок: {self.apples}/3",
            f"  перелётов выполнено: {self.steps_done}",
            f"  время сборки: {self.elapsed:.0f} с",
            f"  возврат на старт: {'да' if self.returned else 'нет'}",
            f"  посадка: {'да' if self.landed else 'нет'}",
        ]
        lines.extend(f"  ! {note}" for note in self.notes)
        lines.append("═" * 62)
        return "\n".join(lines)


class LeaderMission:
    """Автомат миссии лидера."""

    def __init__(
        self,
        drone,
        config: Optional[MissionConfig] = None,
        swarm: Optional[SwarmLink] = None,
        verbose: bool = True,
    ) -> None:
        self.config = config or load_config()
        self.drone = drone
        self.verbose = verbose
        self.swarm: SwarmLink = swarm or ConsoleSwarmLink(self.config.swarm.join_wait_s)

        apples_cfg = (load_apples_config(self.config.apples_config)
                      if self.config.apples_config else load_apples_config())
        self.vision = AppleVision(drone, on_apple=self._on_apple, config=apples_cfg,
                                  publish=False, verbose=verbose)
        self.numbering = self.config.markers.numbering()
        self.detector = MarkerDetector(self.config.markers.dictionary,
                                       self.config.markers.min_marker_percent)
        self.navigator = MarkerNavigator(
            drone, self.config, detector=self.detector,
            projector=GroundProjector(apples_cfg.camera),
            numbering=self.numbering, should_pause=self._paused, verbose=verbose,
        )
        self.hub = FrameHub(drone, [self.vision.process, self.navigator.feed, self._publish])

        self.result = MissionResult()
        self.plan: Optional[SearchPlan] = None
        self.estimate: Optional[search.TimeEstimate] = None
        self.start_node: Optional[Node] = None
        self.current_node: Optional[Node] = None

        self._pause = threading.Event()
        self._pending: List = []
        self._pending_lock = threading.Lock()
        self._started_at: Optional[float] = None
        self._search_active = False
        self._publish_every = 3
        self._publishing = True

    # ------------------------------------------------------------- события

    def _on_apple(self, event) -> None:
        """Коллбек распознавания. Вызывается в потоке кадров — здесь ничего долгого."""
        with self._pending_lock:
            self._pending.append(event)
        self._pause.set()
        # Зависание отдаём сразу, не дожидаясь основного цикла: регламент требует
        # остановиться на месте в момент распознавания. Но только если поиск уже идёт —
        # иначе яблоко, замеченное на взлёте, заморозило бы дрон на полпути к высоте.
        if self._search_active:
            self.navigator.freeze()

    def _paused(self) -> bool:
        return self._pause.is_set()

    def _publish(self, frame: np.ndarray) -> None:
        """Разметка яблок и меток в `/out_detection` — её смотрят в веб-интерфейсе."""
        if self.drone is None or not self._publishing or self.hub.frames % self._publish_every:
            return
        left = self.time_left()
        status = (f"apples {self.vision.count}/3 | "
                  f"node {self.current_node} | "
                  f"budget {left:.0f}s" if left is not None else
                  f"apples {self.vision.count}/3 | node {self.current_node}")
        canvas = draw_detections(frame, self.vision.last_detections, self.vision.registry, status)
        target = (self.numbering.marker_id(self.current_node)
                  if self.current_node is not None else None)
        canvas = markers.draw(canvas, self.navigator.last_sights, target)
        try:
            self.drone.image.publish(canvas)
        except Exception as exc:
            self._publishing = False   # публикация не работает — не спамим ошибкой
            print(f"[mission] разметка в /out_detection отключена: {exc}", flush=True)

    # -------------------------------------------------------------- бюджет

    def time_left(self) -> Optional[float]:
        """Сколько осталось от 3 минут на сборку формации."""
        if self._started_at is None:
            return None
        return self.config.formation_budget - (time.monotonic() - self._started_at)

    def _budget_over(self) -> bool:
        left = self.time_left()
        return left is not None and left <= 0

    # --------------------------------------------------------------- этапы

    def takeoff(self) -> bool:
        alt = self.config.flight.altitude
        self._log(f"взлёт на {alt} м")
        self._started_at = time.monotonic()
        try:
            self.drone.control.navigate(x=0.0, y=0.0, z=alt, yaw=self.config.flight.yaw,
                                        speed=self.config.flight.speed, frame_id="body",
                                        auto_arm=True)
        except Exception as exc:
            self.result.notes.append(f"взлёт не прошёл: {exc}")
            return False

        deadline = time.monotonic() + self.config.flight.takeoff_timeout
        while time.monotonic() < deadline:
            height = self.navigator.altitude()
            if height is not None and height >= alt * 0.85:
                self._log(f"на высоте {height:.2f} м")
                return True
            time.sleep(0.3)
        self.result.notes.append("высота не подтверждена телеметрией, продолжаем")
        return True

    def identify_start(self, timeout: float = 12.0) -> Optional[Node]:
        """Считывает стартовый маркер и печатает ID с названием фигуры."""
        deadline = time.monotonic() + timeout
        seq = -1
        fallback: Optional[Node] = None

        while time.monotonic() < deadline:
            frame, sights, seq = self.navigator.wait_frame(after=seq, timeout=1.0)
            if frame is None:
                continue
            frame_size = (frame.shape[1], frame.shape[0])
            sight = markers.start_marker(sights, frame_size, self.numbering)
            if sight is not None:
                node = self.numbering.node_of(sight.id)
                figure = field.figure_for(sight.id)
                self.result.start_marker = sight.id
                self.result.figure = figure
                print("=" * 62, flush=True)
                print(f">>> СТАРТОВЫЙ МАРКЕР ArUco: ID = {sight.id}", flush=True)
                print(f">>> НАЗНАЧЕННАЯ ФИГУРА: {figure}", flush=True)
                print(f">>> стартовый узел поля: столбец {node[0]}, строка {node[1]}", flush=True)
                print("=" * 62, flush=True)
                return node
            if fallback is None:
                near = markers.nearest_to_center(sights, frame_size)
                if near is not None:
                    try:
                        fallback = self.numbering.node_of(near.id)
                    except ValueError:
                        fallback = None

        if fallback is not None:
            self.result.notes.append(
                f"угловой маркер не найден, стартуем от узла {fallback} — фигура неизвестна")
            self._log(f"угловая метка не видна, стартовый узел взят по ближайшей метке: {fallback}")
            return fallback

        self.result.notes.append("ни одной метки не видно — маршрут построить нельзя")
        return None

    def measure_span(self, timeout: float = 3.0) -> Tuple[int, int]:
        """Полоса обзора в метках на рабочей высоте — от неё зависит разрежение проходов."""
        best = (0, 0)
        deadline = time.monotonic() + timeout
        seq = -1
        while time.monotonic() < deadline:
            frame, sights, seq = self.navigator.wait_frame(after=seq, timeout=1.0)
            if frame is None:
                continue
            span = markers.visible_span(sights, self.numbering,
                                        (frame.shape[1], frame.shape[0]))
            best = (max(best[0], span[0]), max(best[1], span[1]))
        if best == (0, 0):
            default = self.config.search.default_span
            self.result.notes.append(f"полосу обзора измерить не удалось, взято {default}×{default}")
            return default, default
        self._log(f"в кадре видно {best[0]}×{best[1]} меток — это и есть полоса обзора")
        return best

    def build_plan(self, start: Node, span: Tuple[int, int], quiet: bool = False) -> SearchPlan:
        plan = search.plan(start, span, self.config.search)
        estimate = search.estimate(plan, self.config.search, apples=3,
                                   budget_s=self.config.formation_budget)
        if not quiet:
            print(plan.describe(), flush=True)
            print(estimate.describe(), flush=True)
            print(field.render(plan.route, plan.start), flush=True)
            if not estimate.fits:
                self.result.notes.append(
                    "оценка времени превышает бюджет сборки — поиск прервётся по таймеру")
        self.plan = plan
        self.estimate = estimate
        return plan

    def plan_search(self, start: Node) -> SearchPlan:
        """Меряет полосу обзора и строит маршрут, при необходимости поднимаясь выше.

        Подъём выполняется **до начала поиска**: чем выше дрон, тем шире полоса
        обзора камеры и тем меньше проходов нужно. Во время самого поиска высота
        уже не меняется — регламент требует движения в одной горизонтальной плоскости.
        """
        span = self.measure_span()
        plan = self.build_plan(start, span, quiet=True)

        while (self.config.flight.auto_altitude
               and not self.estimate.fits
               and self.config.flight.altitude < ALTITUDE_CEILING_M - 0.05):
            target = min(ALTITUDE_CEILING_M,
                         self.config.flight.altitude + self.config.flight.altitude_step)
            self._log(f"маршрут не влезает в бюджет ({self.estimate.total_s:.0f} с) — "
                      f"поднимаемся с {self.config.flight.altitude:.2f} до {target:.2f} м "
                      "до начала поиска")
            if not self.climb_to(target):
                break
            span = self.measure_span()
            plan = self.build_plan(start, span, quiet=True)

        self._log(f"высота поиска зафиксирована: {self.config.flight.altitude:.2f} м")
        return self.build_plan(start, span)

    def climb_to(self, altitude: float) -> bool:
        """Смена высоты между этапами (не во время поиска)."""
        altitude = min(float(altitude), ALTITUDE_CEILING_M)
        try:
            self.drone.control.set_altitude(z=altitude, frame_id="terrain")
        except Exception as exc:
            self.result.notes.append(f"смена высоты не прошла: {exc}")
            return False
        self.config.flight.altitude = altitude
        deadline = time.monotonic() + self.config.flight.takeoff_timeout
        while time.monotonic() < deadline:
            height = self.navigator.altitude()
            if height is not None and abs(height - altitude) <= 0.15:
                return True
            time.sleep(0.2)
        self.result.notes.append(f"высота {altitude:.2f} м не подтверждена телеметрией")
        return True

    # ---------------------------------------------------------------- поиск

    def search_apples(self, plan: SearchPlan) -> None:
        route = plan.route
        self.current_node = plan.start
        index = 1

        # Встаём точно над стартовой меткой: дальше весь маршрут отсчитывается от неё.
        self.navigator.stabilize(plan.start)
        self._search_active = True

        while index < len(route):
            # Зависание на яблоке — первым делом: даже если это было последнее яблоко
            # и поиск закончен, следующий дрон обязан подняться до возобновления движения.
            if self._paused():
                self.handle_apple()
                continue
            if self.vision.complete:
                self._log("все три яблока засчитаны — поиск закончен досрочно")
                break
            if self._budget_over():
                self.result.notes.append("бюджет 3 минуты исчерпан — поиск прерван")
                self._log("бюджет сборки исчерпан, идём на стартовую позицию")
                break

            node = route[index]
            result = self.navigator.goto(node)
            if result.paused:
                self.handle_apple()
                continue
            if not result.ok:
                self.result.notes.append(
                    f"узел {node}: перелёт не подтверждён ({result.reason})")
                self._log(f"узел {node}: {result.reason}, идём дальше")

            self.current_node = node
            self.result.steps_done += 1

            stabilized = self.navigator.stabilize(node)
            if stabilized.paused:
                self.handle_apple()
                continue
            if not stabilized.ok:
                self._log(f"узел {node}: стабилизация {stabilized.reason} "
                          f"(смещение {stabilized.offset if stabilized.offset is None else round(stabilized.offset, 3)})")
            index += 1

    def handle_apple(self) -> None:
        """Зависание на яблоке и подъём следующего дрона (регламент, п. 2.1.2)."""
        with self._pending_lock:
            events = list(self._pending)
            self._pending.clear()

        self.navigator.freeze()
        for event in events:
            print("=" * 62, flush=True)
            print(f">>> {event.describe()}", flush=True)
            print(">>> ЦЕПОЧКА ЗАВИСАЕТ НА МЕСТЕ", flush=True)
            print("=" * 62, flush=True)
            self.result.apples = self.vision.count

            self.swarm.launch_next(event.index)
            joined = threading.Event()
            outcome: List[bool] = []

            def waiter() -> None:
                try:
                    outcome.append(bool(self.swarm.wait_tail_joined(
                        event.index, self.config.swarm.join_timeout)))
                finally:
                    joined.set()

            worker = threading.Thread(target=waiter, name="swarm_join", daemon=True)
            worker.start()
            # Пока ждём хвост, цель удерживается — дрон не должен «уплыть».
            self.navigator.hold(self.config.swarm.join_timeout, until=joined.is_set)
            worker.join(timeout=1.0)

            if not outcome or not outcome[0]:
                self.result.notes.append(
                    f"яблоко #{event.index}: вход дрона в хвост не подтверждён")
                self._log(f"яблоко #{event.index}: подтверждения строя нет, продолжаем поиск")

        self._pause.clear()

    # -------------------------------------------------------- возврат и посадка

    def return_to_start(self) -> bool:
        """Приводит лидера на стартовую позицию — оттуда начинается фигура."""
        if self.start_node is None or self.current_node is None:
            return False
        path = field.straight_path(self.current_node, self.start_node)
        if not path:
            self.result.returned = True
            return True
        self._log(f"возврат на стартовую метку: {len(path)} перелётов")
        for node in path:
            if self._paused():
                self.handle_apple()
            result = self.navigator.goto(node)
            self.current_node = node
            self.result.steps_done += 1
            if not result.ok:
                self._log(f"возврат, узел {node}: {result.reason}")
            self.navigator.stabilize(node)
        self.result.returned = True
        return True

    def land(self) -> bool:
        # На посадке зависание по яблоку уже не нужно — только помешало бы сесть.
        self._search_active = False
        try:
            self.drone.control.land()
            self.result.landed = True
            self._log("посадка")
            return True
        except Exception as exc:
            self.result.notes.append(f"посадка не прошла: {exc}")
            return False

    # ----------------------------------------------------------------- запуск

    def run(self) -> MissionResult:
        self.hub.start()
        try:
            if not self.takeoff():
                return self._finish()

            self.navigator.probe_strategy()

            start = self.identify_start()
            if start is None:
                self.result.notes.append("миссия прервана: поле не опознано")
                self.land()
                return self._finish()
            self.start_node = start
            self.current_node = start

            plan = self.plan_search(start)
            self.search_apples(plan)
            self.return_to_start()

            if self.config.flight.land_at_end:
                self.land()
            else:
                self._log("посадка отключена (land_at_end: false) — цепочка готова к фигуре")
            return self._finish()

        except KeyboardInterrupt:
            self._log("остановлено оператором, сажаем")
            self.result.notes.append("остановлено оператором")
            self.land()
            return self._finish()
        finally:
            self.hub.stop()
            self.vision.stop()

    def _finish(self) -> MissionResult:
        self.result.apples = self.vision.count
        self.result.elapsed = (time.monotonic() - self._started_at) if self._started_at else 0.0
        print(flush=True)
        print(self.vision.summary(), flush=True)
        print(self.result.describe(), flush=True)
        return self.result

    def _log(self, message: str) -> None:
        if self.verbose:
            left = self.time_left()
            stamp = f"[{left:5.0f}s]" if left is not None else "[     ]"
            print(f"[mission]{stamp} {message}", flush=True)


def dry_run(config: Optional[MissionConfig] = None, start_marker: int = 0,
            span: Optional[Tuple[int, int]] = None) -> SearchPlan:
    """Полный расчёт миссии без дрона: маршрут, проверка регламента, оценка времени."""
    config = config or load_config()
    numbering = config.markers.numbering()
    start = numbering.node_of(start_marker)
    span = span or (config.search.default_span, config.search.default_span)

    figure = field.figure_for(start_marker)
    print(f"Стартовый маркер {start_marker} → узел {start} → фигура: {figure or '—'}")
    plan = search.plan(start, span, config.search)
    estimate = search.estimate(plan, config.search, apples=3, budget_s=config.formation_budget)
    print(plan.describe())
    print(estimate.describe())
    print(field.render(plan.route, plan.start))
    return plan
