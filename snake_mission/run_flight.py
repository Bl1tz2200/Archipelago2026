#!/usr/bin/env python3
"""Полный автономный полёт: арм → взлёт → змейка по плану → посадка.

Отдельно от `run_leader.py`: здесь нет ни распознавания «яблок», ни роя, ни
трёхминутного бюджета — только навигация. Это то, чем проверяют, что дрон вообще
летает по полю, прежде чем гонять зачётную миссию.

Главное отличие от миссии — **ни один вызов к дрону не может подвесить программу
молча**. На стенде без телеметрии FCU `get_telemetry()` не возвращается никогда и не
реагирует на Ctrl+C; здесь каждый вызов идёт через сторож с таймаутом, поэтому
скрипт всегда доходит до посадки и печатает, на каком шаге всё встало.

Запуск (любым интерпретатором — ROS 2 должен быть в окружении этой же оболочки):

    python3 snake_mission/run_flight.py --alt 0.5 --speed 0.3
    python3 snake_mission/run_flight.py --nav markers --start 0
    python3 snake_mission/run_flight.py --dry-run          # только показать маршрут

Два способа лететь:

  --nav blind     (по умолчанию) счисление пути: перелёт = смещение по корпусу на
                  `--step` метров. Камера не нужна вообще. Работает на любой высоте,
                  но накапливает снос — для первого «просто пролететь» это нормально.
  --nav markers   наведение по ArUco-меткам через `MarkerNavigator` — точно, но нужна
                  видимость меток; на 0.5 м в кадр попадает мало меток.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snake_mission import field, search  # noqa: E402
from snake_mission.config import ALTITUDE_CEILING_M, MissionConfig, load_config  # noqa: E402
from snake_mission.field import Node  # noqa: E402

# Сколько ждать ответа на команду дрону, прежде чем считать её зависшей.
CALL_TIMEOUT_S = 6.0
# Отдельно для посадки: она обязана быть отправлена, ждём дольше.
LAND_TIMEOUT_S = 10.0


# --------------------------------------------------------------------- сторож


class Call:
    """Результат вызова к дрону: значение либо причина, по которой его нет."""

    __slots__ = ("value", "status")

    def __init__(self, value=None, status: str = "ok") -> None:
        self.value = value
        self.status = status

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def __bool__(self) -> bool:
        return self.ok


def guarded(fn: Callable, *args, timeout: float = CALL_TIMEOUT_S, **kwargs) -> Call:
    """Зовёт `fn` в потоке-демоне и бросает поток, если тот не вернулся за `timeout`.

    Убить зависший вызов внутри ROS/DDS нельзя — его можно только оставить. Поток
    демонический, поэтому брошенный вызов не помешает программе завершиться.
    """
    box: dict = {}

    def worker() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:            # noqa: BLE001 — сюда же ловим ошибки ROS
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return Call(status=f"завис (> {timeout:.0f} с)")
    if "error" in box:
        return Call(status=f"ошибка: {box['error']}")
    return Call(box.get("value"))


# ----------------------------------------------------------------- сам полёт


class Flight:
    """Полёт по готовому маршруту: взлёт, перелёты, посадка."""

    def __init__(self, drone, config: MissionConfig, step_m: float,
                 nav: str = "blind", verbose: bool = True) -> None:
        self.drone = drone
        self.config = config
        self.step_m = step_m
        self.nav = nav
        self.verbose = verbose

        self.steps_done = 0
        self.notes: List[str] = []
        self.airborne = False
        self.landed = False
        self.started_at = 0.0

        self.navigator = None
        self.hub = None
        if nav == "markers":
            self._setup_marker_nav()

    # ------------------------------------------------------------ навигация

    def _setup_marker_nav(self) -> None:
        """Поднимает камеру и `MarkerNavigator` — только для `--nav markers`."""
        from apple_vision.geometry import GroundProjector
        from apple_vision.config import load_config as load_apples_config
        from snake_mission.markers import MarkerDetector
        from snake_mission.mission import FrameHub
        from snake_mission.navigator import MarkerNavigator

        apples_cfg = load_apples_config()
        detector = MarkerDetector(self.config.markers.dictionary,
                                  self.config.markers.min_marker_percent)
        self.navigator = MarkerNavigator(
            self.drone, self.config, detector=detector,
            projector=GroundProjector(apples_cfg.camera),
            numbering=self.config.markers.numbering(), verbose=self.verbose,
        )
        self.hub = FrameHub(self.drone, [self.navigator.feed])

    # -------------------------------------------------------------- этапы

    def preflight(self) -> bool:
        """Проверяет, что дрон отвечает. Без телеметрии лететь нельзя."""
        self._log("проверка связи с дроном…")
        call = guarded(self.drone.control.get_telemetry, frame_id="map")
        if not call.ok:
            self.notes.append(f"телеметрия недоступна: {call.status}")
            self._log(f"ТЕЛЕМЕТРИИ НЕТ — {call.status}")
            self._log("проверьте FCU: ros2 topic hz /fmu/out/vehicle_status")
            return False

        t = call.value
        mode = getattr(t, "mode", "?")
        armed = getattr(t, "armed", "?")
        volts = getattr(t, "voltage", None)
        self._log(f"связь есть: режим {mode}, моторы {'запущены' if armed else 'выключены'}"
                  + (f", АКБ {volts:.1f} В" if isinstance(volts, (int, float)) else ""))
        if isinstance(volts, (int, float)) and 0 < volts < 11.4:
            self.notes.append(f"АКБ просажен: {volts:.1f} В (норма от 11.4 В)")
            self._log(f"ВНИМАНИЕ: АКБ {volts:.1f} В — ниже нормы 11.4 В")
        return True

    def takeoff(self) -> bool:
        """Арм и взлёт. `auto_arm=True` переводит в OFFBOARD и запускает моторы."""
        alt = self.config.flight.altitude
        self._log(f"АРМ и ВЗЛЁТ на {alt:.2f} м")
        self.started_at = time.monotonic()

        call = guarded(self.drone.control.navigate,
                       x=0.0, y=0.0, z=alt, yaw=self.config.flight.yaw,
                       speed=self.config.flight.speed, frame_id="body", auto_arm=True)
        if not call.ok:
            self.notes.append(f"взлёт не отправлен: {call.status}")
            self._log(f"ВЗЛЁТ НЕ ПРОШЁЛ — {call.status}")
            return False

        self.airborne = True   # команда ушла: дальше обязаны сесть в любом случае
        deadline = time.monotonic() + self.config.flight.takeoff_timeout
        while time.monotonic() < deadline:
            height = self._altitude()
            if height is not None and height >= alt * 0.85:
                self._log(f"на высоте {height:.2f} м")
                return True
            time.sleep(0.3)

        self.notes.append("высота не подтверждена телеметрией — летим дальше")
        self._log(f"высоту {alt:.2f} м телеметрия не подтвердила, продолжаем")
        return True

    def fly(self, route: List[Node]) -> bool:
        """Проходит маршрут узел за узлом. Возвращает False, если сорвались на полпути."""
        if len(route) < 2:
            return True
        if self.hub is not None:
            self.hub.start()
        if self.navigator is not None:
            self.navigator.probe_strategy()

        total = len(route) - 1
        position = route[0]
        self._log(f"маршрут: {total} перелётов, старт из узла {position}")

        for index, node in enumerate(route[1:], start=1):
            moved = self._step_to(position, node)
            if not moved:
                self.notes.append(f"перелёт {index}/{total} в узел {node} не прошёл — прерываем")
                self._log(f"перелёт в {node} не прошёл, идём на посадку")
                return False
            position = node
            self.steps_done += 1
            if self.verbose:
                self._log(f"[{index:3d}/{total}] узел {node} "
                          f"(метка {self.config.markers.numbering().marker_id(node)})")
        self._log("маршрут пройден полностью")
        return True

    def _step_to(self, current: Node, target: Node) -> bool:
        if self.nav == "markers":
            return self._step_markers(target)
        return self._step_blind(current, target)

    def _step_blind(self, current: Node, target: Node) -> bool:
        """Счисление пути: смещение по корпусу (FLU — x вперёд, y влево).

        Соответствие осей сетке взято тем же, что у симулятора: строки идут по «вперёд»,
        столбцы — по «влево» с минусом. Курс весь полёт держим постоянным, поэтому
        корпус остаётся сонаправлен полю.
        """
        dcol = target[0] - current[0]
        drow = target[1] - current[1]
        x = drow * self.step_m
        y = -dcol * self.step_m

        call = guarded(self.drone.control.navigate,
                       x=x, y=y, z=0.0, yaw=self.config.flight.yaw,
                       speed=self.config.flight.speed, frame_id="body", auto_arm=False)
        if not call.ok:
            self.notes.append(f"navigate(body) в {target}: {call.status}")
            return False

        # Ждём столько, сколько занимает перелёт на этой скорости, плюс запас на разгон.
        distance = math.hypot(x, y)
        travel = distance / max(self.config.flight.speed, 0.05)
        time.sleep(travel + 1.0)
        return True

    def _step_markers(self, target: Node) -> bool:
        result = self.navigator.goto(target)
        if not result.ok:
            self.notes.append(f"узел {target}: перелёт не подтверждён ({result.reason})")
            self._log(f"узел {target}: {result.reason}")
        stabilized = self.navigator.stabilize(target)
        if not stabilized.ok:
            self._log(f"узел {target}: стабилизация {stabilized.reason}")
        # Навигация по меткам не считается сорванной: метку могло просто не быть видно,
        # но дрон при этом остаётся управляемым и маршрут можно продолжать.
        return True

    def land(self) -> bool:
        if self.landed:
            return True
        self._log("ПОСАДКА")
        call = guarded(self.drone.control.land, timeout=LAND_TIMEOUT_S)
        if not call.ok:
            self.notes.append(f"посадка не прошла: {call.status}")
            self._log(f"КОМАНДА ПОСАДКИ НЕ ПРОШЛА — {call.status}")
            self._log("сажайте с пульта: тумблер SB в Position, затем газ вниз")
            return False
        self.landed = True
        self._log("команда посадки отправлена")
        return True

    # ------------------------------------------------------------ служебное

    def _altitude(self) -> Optional[float]:
        for frame_id in ("terrain", "map"):
            call = guarded(self.drone.control.get_telemetry, frame_id=frame_id)
            if call.ok and call.value is not None:
                try:
                    return float(call.value.z)
                except (TypeError, ValueError, AttributeError):
                    continue
        return None

    def stop_camera(self) -> None:
        if self.hub is not None:
            self.hub.stop()

    def _log(self, message: str) -> None:
        if not self.verbose:
            return
        stamp = time.monotonic() - self.started_at if self.started_at else 0.0
        print(f"[полёт][{stamp:6.1f}s] {message}", flush=True)

    def report(self) -> str:
        lines = [
            "═" * 62,
            "ИТОГ ПОЛЁТА",
            f"  перелётов выполнено: {self.steps_done}",
            f"  время в воздухе: {time.monotonic() - self.started_at:.0f} с"
            if self.started_at else "  время в воздухе: —",
            f"  посадка: {'да' if self.landed else 'НЕТ'}",
        ]
        lines.extend(f"  ! {note}" for note in self.notes)
        lines.append("═" * 62)
        return "\n".join(lines)


# ---------------------------------------------------------------- маршрут


def build_route(config: MissionConfig, start_marker: int, span: int,
                back_to_start: bool) -> Tuple[List[Node], Node]:
    """Строит змейку по всему полю от стартовой метки."""
    numbering = config.markers.numbering()
    start = numbering.node_of(start_marker)
    plan = search.plan(start, (span, span), config.search)
    route = list(plan.route)
    if back_to_start:
        route.extend(field.straight_path(plan.end, start))
    return route, start


# -------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Полный автономный полёт змейкой: арм, взлёт, маршрут, посадка")
    p.add_argument("--config", default="", help="путь к YAML-конфигу миссии")
    p.add_argument("--alt", type=float, default=0.0,
                   help=f"высота полёта, м (потолок {ALTITUDE_CEILING_M})")
    p.add_argument("--speed", type=float, default=0.0, help="скорость перелёта, м/с")
    p.add_argument("--start", type=int, default=0, help="ID стартовой метки")
    p.add_argument("--step", type=float, default=1.0,
                   help="шаг сетки меток на площадке, м (для --nav blind)")
    p.add_argument("--span", type=int, default=1,
                   help="сколько рядов меток «видно» при планировании: 1 = обойти все узлы")
    p.add_argument("--nav", default="blind", choices=["blind", "markers"],
                   help="blind — счисление пути, markers — наведение по ArUco")
    p.add_argument("--no-return", action="store_true",
                   help="не возвращаться на стартовый узел перед посадкой")
    p.add_argument("--dry-run", action="store_true", help="показать маршрут и выйти")
    p.add_argument("--quiet", action="store_true", help="без подробного вывода")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = load_config(args.config)
        if args.alt:
            config.flight.altitude = args.alt
            config.flight.__post_init__()      # потолок высоты проверяется здесь
        if args.speed:
            config.flight.speed = args.speed
    except ValueError as exc:
        print(f"Конфиг отклонён: {exc}")
        return 1

    try:
        route, start = build_route(config, args.start, args.span, not args.no_return)
    except (ValueError, search.RouteError) as exc:
        print(f"Маршрут построить нельзя: {exc}")
        return 1

    numbering = config.markers.numbering()
    print(f"Старт: метка {args.start} → узел {start}")
    print(f"Маршрут: {len(route) - 1} перелётов, "
          f"высота {config.flight.altitude:.2f} м, скорость {config.flight.speed:.2f} м/с, "
          f"навигация {args.nav}")
    print(field.render(route, start))

    if args.dry_run:
        ids = [numbering.marker_id(n) for n in route]
        print("метки по порядку: " + " → ".join(str(i) for i in ids))
        return 0

    try:
        import sverk_interfaces
    except ImportError:
        print("sverk_interfaces не найден — нужен дрон и окружение ROS 2 в этой оболочке:")
        print("  source /opt/ros/*/setup.bash && source ~/sverk_ws/install/setup.bash")
        print("Проверить маршрут без дрона: --dry-run")
        return 1

    drone = sverk_interfaces.init(Nodename="snake_flight")
    from snake_mission import camera_compat
    camera_compat.patch_image_api(drone)   # камера отдаёт YUV — учим to_cv2 его понимать

    flight = Flight(drone, config, step_m=args.step, nav=args.nav, verbose=not args.quiet)
    completed = False
    try:
        if not flight.preflight():
            print(flight.report())
            return 1
        if flight.takeoff():
            completed = flight.fly(route)
    except KeyboardInterrupt:
        flight.notes.append("прервано оператором")
        print("\n[полёт] остановлено оператором — сажаем", flush=True)
    finally:
        # Взлетели — обязаны сесть, чем бы ни кончился маршрут.
        if flight.airborne:
            flight.land()
        flight.stop_camera()
        guarded(drone.close, timeout=3.0)
        print(flight.report(), flush=True)

    return 0 if (completed and flight.landed) else 2


if __name__ == "__main__":
    raise SystemExit(main())
