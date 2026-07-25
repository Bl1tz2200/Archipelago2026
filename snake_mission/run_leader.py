#!/usr/bin/env python3
"""Запуск миссии лидера на борту Обрика.

Расчёт без дрона — маршрут, покрытие, время (можно на ноутбуке):

    python3 run_leader.py --dry-run --start 42

Зачётный запуск (на дроне, внутри контейнера sverk_ros2):

    python3 run_leader.py
    python3 run_leader.py --alt 2.0 --no-land

Что происходит: взлёт → чтение стартового ArUco-маркера (ID и фигура печатаются
в терминал) → поиск «яблок» перелётами по меткам со стабилизацией на каждой →
зависание и подъём следующего дрона на каждом яблоке → возврат на стартовую метку.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snake_mission import ConsoleSwarmLink, load_config  # noqa: E402
from snake_mission.mission import LeaderMission, dry_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Миссия лидера «Змейка»")
    p.add_argument("--config", default="", help="путь к YAML-конфигу (по умолчанию config/snake.yaml)")
    p.add_argument("--alt", type=float, default=0.0, help="высота поиска, м (потолок 3.0)")
    p.add_argument("--speed", type=float, default=0.0, help="скорость перелёта, м/с")
    p.add_argument("--strategy", default="",
                   choices=["", "body_step", "auto", "aruco_frame", "visual"],
                   help="как лететь: body_step (navigate по корпусу, штатно) | "
                        "aruco_frame | visual | auto")
    p.add_argument("--step", type=float, default=0.0,
                   help="шаг сетки меток на площадке, м (для --strategy body_step)")
    p.add_argument("--no-land", action="store_true", help="не садиться: зависнуть на старте")
    p.add_argument("--quiet", action="store_true", help="без подробного вывода")
    p.add_argument("--dry-run", action="store_true",
                   help="расчёт без дрона: маршрут, покрытие, проверка регламента, время")
    p.add_argument("--start", type=int, default=0,
                   help="для --dry-run: ID стартового маркера (0, 6, 42 или 48)")
    p.add_argument("--span", type=int, default=0,
                   help="для --dry-run: сколько меток видно в кадре по стороне")
    return p.parse_args()


def apply_overrides(config, args: argparse.Namespace):
    if args.alt:
        config.flight.altitude = args.alt
        config.flight.__post_init__()      # потолок высоты проверяется и здесь
    if args.speed:
        config.flight.speed = args.speed
    if args.strategy:
        config.navigation.strategy = args.strategy
        config.navigation.__post_init__()   # допустимость стратегии проверяется здесь
    if args.step:
        config.navigation.step_m = args.step
        config.navigation.__post_init__()
    if args.no_land:
        config.flight.land_at_end = False
    return config


def main() -> int:
    args = parse_args()
    try:
        config = apply_overrides(load_config(args.config), args)
    except ValueError as exc:
        print(f"Конфиг отклонён: {exc}")
        return 1

    if args.dry_run:
        span = (args.span, args.span) if args.span else None
        dry_run(config, start_marker=args.start, span=span)
        return 0

    try:
        import sverk_interfaces
    except ImportError:
        print("sverk_interfaces не найден — запускайте на дроне, внутри контейнера sverk_ros2.")
        print("Проверить расчёт без дрона: python3 run_leader.py --dry-run --start 42")
        return 1

    drone = sverk_interfaces.init(Nodename="snake_leader")
    from snake_mission import camera_compat
    camera_compat.patch_image_api(drone)  # камера отдаёт YUV — учим to_cv2 его понимать
    mission = LeaderMission(
        drone,
        config=config,
        swarm=ConsoleSwarmLink(config.swarm.join_wait_s),
        verbose=not args.quiet,
    )
    try:
        result = mission.run()
    finally:
        drone.close()

    return 0 if result.apples >= 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
