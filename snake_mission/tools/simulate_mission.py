#!/usr/bin/env python3
"""Полный прогон миссии лидера в симуляторе — без дрона и без камеры.

    python3 tools/simulate_mission.py                    # старт из угла 0
    python3 tools/simulate_mission.py --start 48 --alt 2.0
    python3 tools/simulate_mission.py --step 1.2 --save трек.png

Симулятор рисует поле 7×7 и «яблоки» на полу, считает полёт и отдаёт кадры той же
программе, что полетит на площадке. Прогоняется всё: чтение стартового маркера,
измерение полосы обзора, маршрут, стабилизация на метках, зависания на «яблоках»,
возврат на старт и посадка.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snake_mission import field  # noqa: E402
from snake_mission.config import load_config  # noqa: E402
from snake_mission.mission import LeaderMission  # noqa: E402
from snake_mission.simulator import SimDrone, SimWorld, default_apples  # noqa: E402
from snake_mission.swarm import ConsoleSwarmLink  # noqa: E402


def draw_track(drone: SimDrone, path: str) -> None:
    """Картинка пройденного трека поверх сетки узлов."""
    import cv2
    import numpy as np

    cell, margin = 90, 40
    size = cell * (field.SIDE - 1) + 2 * margin
    canvas = np.full((size, size, 3), 250, np.uint8)

    def to_px(col: float, row: float):
        return int(margin + col * cell), int(size - margin - row * cell)

    for node in field.all_nodes():
        cv2.circle(canvas, to_px(*node), 4, (200, 200, 200), -1)
    for name, (acol, arow) in drone.apples.items():
        color = {"red": (40, 40, 200), "green": (60, 170, 60),
                 "yellow": (40, 210, 230)}.get(name, (150, 150, 150))
        cv2.circle(canvas, to_px(acol, arow), 12, color, -1)

    points = [to_px(*p) for p in drone.track]
    for a, b in zip(points, points[1:]):
        cv2.line(canvas, a, b, (200, 80, 40), 2)
    if points:
        cv2.circle(canvas, points[0], 8, (0, 150, 0), 2)
        cv2.circle(canvas, points[-1], 8, (0, 0, 200), 2)
    cv2.imwrite(path, canvas)
    print(f"трек полёта: {path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Прогон миссии лидера в симуляторе")
    p.add_argument("--config", default="", help="путь к YAML-конфигу")
    p.add_argument("--start", type=int, default=0, help="ID стартового угла (0, 6, 42, 48)")
    p.add_argument("--alt", type=float, default=0.0, help="стартовая высота поиска, м")
    p.add_argument("--step", type=float, default=1.0, help="шаг сетки меток на площадке, м")
    p.add_argument("--hfov", type=float, default=65.0, help="угол обзора камеры, градусы")
    p.add_argument("--scale", type=float, default=8.0, help="во сколько раз ускорить время")
    p.add_argument("--save", default="", help="куда сохранить картинку трека")
    p.add_argument("--quiet", action="store_true", help="без подробного вывода")
    args = p.parse_args()

    config = load_config(args.config)
    if args.alt:
        config.flight.altitude = args.alt
        config.flight.__post_init__()
    # Симуляция идёт быстрее реального времени, поэтому и бюджет сборки, и паузы
    # ожидания роя сжимаются во столько же раз — иначе 3 минуты регламента
    # оказались бы недостижимо щедрыми.
    config.formation_budget = config.formation_budget / args.scale
    config.swarm.join_wait_s = config.swarm.join_wait_s / args.scale
    config.swarm.join_timeout = config.swarm.join_timeout / args.scale
    config.search.step_time_s /= args.scale
    config.search.stabilize_time_s /= args.scale
    config.search.apple_hold_s = config.swarm.join_wait_s + 1.0 / args.scale

    world = SimWorld(step_m=args.step, hfov_deg=args.hfov, time_scale=args.scale,
                     speed=config.flight.speed)
    start_node = config.markers.numbering().node_of(args.start)
    drone = SimDrone(world, start=start_node, apples=default_apples(),
                     numbering=config.markers.numbering())

    print(f"Симуляция: старт из угла {args.start} {start_node}, шаг сетки {args.step} м, "
          f"высота {config.flight.altitude} м, ускорение времени ×{args.scale}")
    print(f"Яблоки на поле: {dict(drone.apples)}")
    print("─" * 62)

    began = time.monotonic()
    mission = LeaderMission(drone, config=config,
                            swarm=ConsoleSwarmLink(config.swarm.join_wait_s),
                            verbose=not args.quiet)
    result = mission.run()
    wall = time.monotonic() - began

    print(f"Реального времени на прогон: {wall:.1f} с "
          f"(модельного: {wall * args.scale:.0f} с)")
    if args.save:
        draw_track(drone, args.save)

    ok = result.apples >= 3 and result.returned and result.landed
    print("ПРОГОН УСПЕШЕН" if ok else "ПРОГОН С ЗАМЕЧАНИЯМИ — смотрите итог выше")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
