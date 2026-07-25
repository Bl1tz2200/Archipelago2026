#!/usr/bin/env python3
"""Живой просмотр камеры дрона с разметкой — отладка распознавания «яблок».

Работает на дроне (внутри контейнера sverk_ros2): тянет кадры с камеры,
прогоняет их через тот же детектор, что и боевой запуск, и публикует
размеченный кадр в `/out_detection`. Смотреть в браузере — Sverk Drone Tools
(панель «Видео с камеры») или Web video server, топик `/out_detection`.
Без ограничения по времени и без зачёта — только диагностика.

Раз в секунду в консоль печатается по каждому цвету: сколько пикселей поймала
HSV-маска и сколько контуров из неё прошло фильтр формы/размера:

    red: маска 0px, контуров прошло 0        -> проблема в цвете/освещении,
                                                 пороки HSV не совпадают —
                                                 см. tools/calibrate_hsv.py
    red: маска 8400px, контуров прошло 0      -> цвет ловится, но форма/размер
                                                 отсекают — крути min_size_percent,
                                                 min_roundness, min_filling
    red: маска 8400px, контуров прошло 1      -> всё видно, ищи причину в другом
                                                 месте (registry.confirm_frames и т.п.)

Использование:

    python3 apple_vision/tools/live_view.py                 # все цвета из конфига
    python3 apple_vision/tools/live_view.py --colors red    # только красный
    python3 apple_vision/tools/live_view.py --masks         # + маски под кадром
    python3 apple_vision/tools/live_view.py --raw           # без детектора, сырой кадр камеры
    python3 apple_vision/tools/live_view.py --save-every 5  # сохранять кадр раз в 5 с

Ctrl+C — выход.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from apple_vision import AppleDetector, draw_detections, load_config, stack_masks  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Живой просмотр камеры дрона для отладки apple_vision")
    p.add_argument("--config", default="", help="путь к YAML-конфигу (по умолчанию config/apples.yaml)")
    p.add_argument("--colors", default="", help="ограничить цвета, например red,green")
    p.add_argument("--masks", action="store_true", help="приклеить маски по цветам под кадром")
    p.add_argument("--raw", action="store_true", help="не гонять детектор — публиковать сырой кадр камеры")
    p.add_argument("--no-publish", action="store_true", help="не публиковать в /out_detection")
    p.add_argument("--interval", type=float, default=0.0, help="пауза между кадрами, с (0 = максимальный темп)")
    p.add_argument("--save-every", type=float, default=0.0, help="сохранять кадр в out/live_view/ раз в N секунд")
    p.add_argument("--duration", type=float, default=0.0, help="сколько секунд смотреть (0 = без ограничения)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import sverk_interfaces
    except ImportError:
        print("sverk_interfaces не найден — запускайте на дроне, внутри контейнера sverk_ros2.")
        print("Проверить детектор без дрона: python3 tools/webcam_demo.py")
        return 1

    drone = sverk_interfaces.init(Nodename="apple_vision_live_view")
    from apple_vision import camera_compat
    camera_compat.patch_image_api(drone)  # камера отдаёт YUV — учим to_cv2 его понимать

    config = load_config(args.config) if args.config else load_config()
    colors = tuple(c.strip() for c in args.colors.split(",") if c.strip())
    detector = None if args.raw else AppleDetector(config, colors)

    save_dir = ""
    if args.save_every > 0:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(root, "out", "live_view")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Кадры сохраняются в {save_dir}")

    if detector is not None:
        print(f"Ищем цвета: {', '.join(p.label for p in detector.profiles)}")
    else:
        print("Режим --raw: детектор выключен, публикуется сырой кадр камеры.")
    if not args.no_publish:
        print("Публикация в /out_detection — смотрите в Sverk Drone Tools или Web video server.")
    print("Ctrl+C — выход.\n")

    frames = 0
    started = time.time()
    last_report = 0.0
    last_save = 0.0

    try:
        while True:
            if args.duration > 0 and time.time() - started > args.duration:
                break

            frame = drone.image.take_picture(timeout=2.0)
            if frame is None:
                print("кадр не получен (timeout) — проверьте камеру: "
                      "ros2 topic hz /camera_1/image_raw")
                time.sleep(0.5)
                continue
            frames += 1

            canvas = frame
            detections = []
            if detector is not None:
                detections = detector.detect(frame)
                canvas = draw_detections(frame, detections, None, f"кадров: {frames}")
                if args.masks:
                    grid = stack_masks(detector.last_masks, width=canvas.shape[1])
                    if grid is not None:
                        if grid.shape[1] != canvas.shape[1]:
                            grid = cv2.resize(grid, (canvas.shape[1], grid.shape[0]))
                        canvas = cv2.vconcat([canvas, grid])

            if not args.no_publish:
                try:
                    drone.image.publish(canvas)
                except Exception as exc:
                    print(f"публикация в /out_detection не удалась: {exc}")
                    args.no_publish = True  # не спамим одной и той же ошибкой

            now = time.time()
            if detector is not None and now - last_report >= 1.0:
                last_report = now
                parts = []
                for name, mask in detector.last_masks.items():
                    hits = sum(1 for d in detections if d.color == name)
                    px = int((mask > 0).sum())
                    parts.append(f"{name}: маска {px}px, контуров прошло {hits}")
                print(" | ".join(parts) if parts else "нет включённых цветов в конфиге", flush=True)

            if save_dir and now - last_save >= args.save_every:
                last_save = now
                cv2.imwrite(os.path.join(save_dir, f"{int(now)}.jpg"), canvas)

            if args.interval > 0:
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nОстановлено оператором")
    finally:
        drone.close()
        print(f"Кадров обработано: {frames}, время: {time.time() - started:.1f} с")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
