#!/usr/bin/env python3
"""Запуск распознавания «яблок» на борту Обрика.

Проверка на земле (дрон включён, пропеллеры сняты):

    python3 run_apple_detect.py --duration 60

Кадр с разметкой уходит в топик `/out_detection` — смотреть в веб-интерфейсе
дрона или через `ros2 run rqt_image_view rqt_image_view`.

Полёт с поиском яблок (взлёт, змейка по узлам сетки, зависание на каждой находке):

    python3 run_apple_detect.py --fly --alt 1.5 --step 1.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apple_vision import AppleVision, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Распознавание «яблок» с камеры дрона")
    p.add_argument("--config", default="", help="путь к YAML-конфигу (по умолчанию config/apples.yaml)")
    p.add_argument("--colors", default="", help="ограничить цвета, например red,green")
    p.add_argument("--duration", type=float, default=180.0, help="сколько секунд искать (0 = без ограничения)")
    p.add_argument("--no-publish", action="store_true", help="не публиковать разметку в /out_detection")
    p.add_argument("--quiet", action="store_true", help="без подробного вывода")
    p.add_argument("--fly", action="store_true", help="взлететь и облетать зону змейкой во время поиска")
    p.add_argument("--alt", type=float, default=1.5, help="высота полёта, м")
    p.add_argument("--step", type=float, default=1.0, help="шаг сетки между узлами, м")
    p.add_argument("--grid", type=int, default=3, help="размер облетаемой сетки NxN узлов")
    return p.parse_args()


def on_apple(event) -> None:
    """Реакция на засчитанное яблоко.

    По регламенту здесь цепочка зависает и с земли поднимается следующий дрон.
    Подставьте сюда вызов своей команды роя — важно, что событие приходит
    ровно один раз на каждое яблоко.
    """
    print("=" * 60, flush=True)
    print(f">>> {event.describe()}", flush=True)
    print(f">>> высота {event.altitude if event.altitude is None else round(event.altitude, 2)} м, "
          f"пиксель {int(event.pixel[0])},{int(event.pixel[1])}", flush=True)
    print(">>> ЗАВИСАНИЕ, ПОДЪЁМ СЛЕДУЮЩЕГО ДРОНА", flush=True)
    print("=" * 60, flush=True)


def snake_nodes(grid: int, step: float):
    """Узлы сетки в порядке обхода «змейкой»: перелёты только между соседями."""
    for row in range(grid):
        cols = range(grid) if row % 2 == 0 else reversed(range(grid))
        for col in cols:
            yield col * step, row * step


def main() -> int:
    args = parse_args()

    try:
        import sverk_interfaces
    except ImportError:
        print("sverk_interfaces не найден — запускайте на дроне, внутри контейнера sverk_ros2.")
        print("Проверить детектор без дрона: python3 tools/offline_test.py --demo")
        return 1

    config = load_config(args.config) if args.config else load_config()
    colors = tuple(c.strip() for c in args.colors.split(",") if c.strip())

    drone = sverk_interfaces.init(Nodename="apple_vision")
    vision = AppleVision(
        drone,
        on_apple=on_apple,
        config=config,
        colors=colors,
        publish=not args.no_publish,
        verbose=not args.quiet,
    )

    print(f"Ищем яблоки: {', '.join(p.label for p in vision.detector.profiles)}")
    print(f"Нужно найти: {vision.registry.config.max_apples}, "
          f"подтверждение: {vision.registry.config.confirm_frames} кадра подряд")

    started = time.time()
    try:
        vision.start(duration=None if args.duration <= 0 else args.duration)

        if args.fly:
            drone.control.navigate(x=0, y=0, z=args.alt, yaw=0.0, speed=0.5,
                                   frame_id="body", auto_arm=True)
            time.sleep(6)
            start = drone.control.get_telemetry(frame_id="map")
            for x, y in snake_nodes(args.grid, args.step):
                if vision.complete:
                    break
                drone.control.navigate_wait(x=start.x + x, y=start.y + y, z=args.alt,
                                            yaw=0.0, speed=0.4, frame_id="map",
                                            tolerance=0.2, timeout=15)
                # Зависание в узле: даём детектору набрать кадры подтверждения.
                deadline = time.time() + 1.5
                while time.time() < deadline and not vision.complete:
                    time.sleep(0.1)
            drone.control.land()
        else:
            while not vision.complete:
                if args.duration > 0 and time.time() - started > args.duration:
                    break
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nОстановлено оператором")
    finally:
        vision.stop()
        print()
        print(vision.summary())
        print(f"Обработано кадров: {vision.frames_processed}, средний темп: {vision.fps:.1f} к/с")
        drone.close()

    return 0 if vision.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
