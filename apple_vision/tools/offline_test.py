#!/usr/bin/env python3
"""Прогон детектора без дрона: по картинке, папке, видео или на синтетике.

    python3 tools/offline_test.py --demo                  # самопроверка, камера не нужна
    python3 tools/offline_test.py --image кадр.png --out разметка.png
    python3 tools/offline_test.py --dir кадры/ --out-dir результат/
    python3 tools/offline_test.py --video запись.mp4      # проверка логики зачёта
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apple_vision import AppleDetector, AppleRegistry, draw_detections, load_config  # noqa: E402
from apple_vision.overlay import stack_masks  # noqa: E402
from apple_vision.synthetic import make_flight, make_frame  # noqa: E402


def report(name: str, detections) -> None:
    if not detections:
        print(f"{name}: ничего не найдено")
        return
    for d in detections:
        print(f"{name}: {d.color:<7} центр ({d.center[0]:6.1f},{d.center[1]:6.1f})  "
              f"r={d.radius:5.1f}  S={d.area:8.0f}px²  circ={d.circularity:.2f} "
              f"solid={d.solidity:.2f} fill={d.fill:.2f} score={d.score:.2f}")


def run_images(paths: List[str], detector: AppleDetector, out_dir: str, masks: bool) -> int:
    found_any = 0
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"{path}: не читается")
            continue
        detections = detector.detect(image)
        found_any += len(detections)
        report(os.path.basename(path), detections)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            target = os.path.join(out_dir, os.path.basename(path))
            cv2.imwrite(target, draw_detections(image, detections))
            if masks:
                grid = stack_masks(detector.last_masks)
                if grid is not None:
                    cv2.imwrite(target.replace(".", "_masks."), grid)
    return found_any


def run_sequence(frames, detector: AppleDetector, registry: AppleRegistry, label: str) -> int:
    """Прогон последовательности кадров с логикой однократного зачёта."""
    for i, frame in enumerate(frames):
        detections = detector.detect(frame)
        for event in registry.update(detections):
            print(f"кадр {i:4d}: {event.describe()}")
    print(f"\n{label}: засчитано {registry.count}/{registry.config.max_apples} "
          f"({', '.join(registry.claimed_colors) or 'ничего'})")
    return 0 if registry.complete else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Офлайн-проверка детектора «яблок»")
    p.add_argument("--config", default="")
    p.add_argument("--image")
    p.add_argument("--dir")
    p.add_argument("--video")
    p.add_argument("--demo", action="store_true", help="синтетический пролёт над тремя яблоками")
    p.add_argument("--out", help="куда сохранить размеченный кадр")
    p.add_argument("--out-dir", help="куда сохранить размеченные кадры папки")
    p.add_argument("--masks", action="store_true", help="сохранять ещё и маски по цветам")
    p.add_argument("--colors", default="")
    args = p.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    colors = tuple(c.strip() for c in args.colors.split(",") if c.strip())
    detector = AppleDetector(cfg, colors)
    print(f"Профили: {', '.join(pr.name for pr in detector.profiles)}  (конфиг: {cfg.path})\n")

    if args.demo:
        sequence = make_flight()
        print(f"Синтетика: {len(sequence)} кадров, три яблока по очереди")
        code = run_sequence([f for f, _ in sequence], detector, AppleRegistry(cfg.registry), "Демо")
        if args.out:
            frame = make_frame([("red", (180, 200), 36), ("green", (400, 260), 32),
                                ("yellow", (520, 140), 30)])
            cv2.imwrite(args.out, draw_detections(frame, detector.detect(frame)))
            print(f"Пример разметки: {args.out}")
        return code

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Не читается: {args.image}")
            return 1
        detections = detector.detect(image)
        report(os.path.basename(args.image), detections)
        if args.out:
            cv2.imwrite(args.out, draw_detections(image, detections))
            print(f"Сохранено: {args.out}")
            if args.masks:
                grid = stack_masks(detector.last_masks)
                if grid is not None:
                    cv2.imwrite(args.out.replace(".", "_masks."), grid)
        return 0 if detections else 2

    if args.dir:
        paths = sorted(sum((glob.glob(os.path.join(args.dir, ext))
                            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")), []))
        if not paths:
            print(f"В {args.dir} нет изображений")
            return 1
        return 0 if run_images(paths, detector, args.out_dir or "", args.masks) else 2

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Не открывается видео: {args.video}")
            return 1
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        print(f"Кадров в видео: {len(frames)}")
        return run_sequence(frames, detector, AppleRegistry(cfg.registry), os.path.basename(args.video))

    p.error("нужен один из: --demo, --image, --dir, --video")


if __name__ == "__main__":
    raise SystemExit(main())
