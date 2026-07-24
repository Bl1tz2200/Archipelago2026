#!/usr/bin/env python3
"""Подбор HSV-порогов под освещение площадки.

Пороги в `config/apples.yaml` — стартовые. На площадке свет другой, и это
главная причина, по которой яблоко не находится. Порядок работы:

1. Снять кадр с яблоком в кадре (на дроне, GUI не нужен):
       python3 tools/calibrate_hsv.py --grab кадр.png

2. Подобрать порог по области с яблоком:
   — по координатам прямоугольника, прямо на дроне:
       python3 tools/calibrate_hsv.py --image кадр.png --roi 300,220,60,60 --color red --save
   — мышкой, на ноутбуке с GUI:
       python3 tools/calibrate_hsv.py --image кадр.png --color red --save
   — ползунками, вживую:
       python3 tools/calibrate_hsv.py --image кадр.png --trackbars

3. Проверить, что с новыми порогами яблоко находится:
       python3 tools/offline_test.py --image кадр.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apple_vision import AppleDetector, load_config, save_config  # noqa: E402
from apple_vision.config import ColorProfile  # noqa: E402


def grab_frame(path: str, timeout: float = 5.0) -> int:
    """Снимок с камеры дрона в файл."""
    try:
        import sverk_interfaces
    except ImportError:
        print("sverk_interfaces не найден — команда --grab работает только на дроне")
        return 1
    drone = sverk_interfaces.init(Nodename="hsv_grab")
    try:
        frame = drone.image.take_picture(timeout=timeout)
        if frame is None:
            print("Кадр не получен. Проверьте камеру: ros2 topic hz /camera_1/image_raw")
            return 1
        cv2.imwrite(path, frame)
        print(f"Кадр {frame.shape[1]}x{frame.shape[0]} сохранён: {path}")
    finally:
        drone.close()
    return 0


def dominant_hue(h: np.ndarray, window: int = 9) -> float:
    """Оттенок самой массивной группы пикселей.

    Среднее по области не годится: в рамку почти всегда попадает фон, и среднее
    уезжает между цветом яблока и цветом пола. Берём пик сглаженной круговой
    гистограммы — он держится за реальный цвет объекта.
    """
    hist = np.bincount(np.clip(h, 0, 179), minlength=180).astype(np.float32)
    kernel = np.ones(window, np.float32) / window
    smooth = np.convolve(np.concatenate([hist, hist, hist]), kernel, mode="same")[180:360]
    return float(np.argmax(smooth))


def hsv_bounds(patch: np.ndarray, margin_h: int, margin_sv: int, percentile: float,
               sat_min: int = 60, val_min: int = 40, hue_window: int = 18
               ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], bool]:
    """Границы HSV по области. Третье значение — True, если цвет «заворачивается» через 0 (красный).

    Серые и тёмные пиксели (пол, тень, рука) отбрасываются, дальше пороги
    считаются только по пикселям доминирующего оттенка.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    colored = (s >= sat_min) & (v >= val_min) & (v <= 252)
    if colored.sum() >= max(20, int(0.03 * h.size)):
        h, s, v = h[colored], s[colored], v[colored]

    # Оттенок — величина круговая: у красного значения лежат и около 0, и около 179,
    # поэтому работаем со смещением относительно пика, а не с абсолютным средним.
    centre = dominant_hue(h)
    delta = ((h - centre + 90.0) % 180.0) - 90.0
    near = np.abs(delta) <= hue_window
    if near.sum() >= max(20, int(0.05 * h.size)):
        delta, s, v = delta[near], s[near], v[near]

    lo_p, hi_p = percentile, 100.0 - percentile
    s_lo, s_hi = np.percentile(s, lo_p), np.percentile(s, hi_p)
    v_lo, v_hi = np.percentile(v, lo_p), np.percentile(v, hi_p)
    d_lo, d_hi = np.percentile(delta, lo_p), np.percentile(delta, hi_p)

    h_lo = centre + d_lo - margin_h
    h_hi = centre + d_hi + margin_h
    wraps = h_lo < 0 or h_hi > 179

    lower = (int(round(h_lo)) % 180, int(max(0, s_lo - margin_sv)), int(max(0, v_lo - margin_sv)))
    upper = (int(round(h_hi)) % 180, int(min(255, s_hi + margin_sv)), int(min(255, v_hi + margin_sv)))
    return lower, upper, wraps


def ranges_from_bounds(lower, upper, wraps: bool):
    """Границы → список диапазонов для inRange (красный распадается на два)."""
    if not wraps:
        return [((lower[0], lower[1], lower[2]), (upper[0], upper[1], upper[2]))]
    return [
        ((0, lower[1], lower[2]), (upper[0], upper[1], upper[2])),
        ((lower[0], lower[1], lower[2]), (179, upper[1], upper[2])),
    ]


def select_roi(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Выделение области мышкой (нужен GUI-сборка OpenCV)."""
    try:
        print("Выделите область ЯБЛОКА мышкой, затем Enter. Отмена — c.")
        roi = cv2.selectROI("Выделите яблоко", image, showCrosshair=True)
        cv2.destroyAllWindows()
    except cv2.error as exc:
        print(f"GUI OpenCV недоступен ({exc}). Задайте область числами: --roi x,y,w,h")
        return None
    return roi if roi[2] > 0 and roi[3] > 0 else None


def trackbars(image: np.ndarray) -> None:
    """Ручной подбор порогов ползунками (нужен GUI)."""
    win = "HSV"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        print(f"GUI OpenCV недоступен ({exc}). Используйте --roi.")
        return
    for name, value, maximum in (("H min", 0, 179), ("H max", 179, 179), ("S min", 100, 255),
                                 ("S max", 255, 255), ("V min", 70, 255), ("V max", 255, 255)):
        cv2.createTrackbar(name, win, value, maximum, lambda _v: None)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    print("q — выход, значения печатаются при выходе")
    while True:
        get = lambda n: cv2.getTrackbarPos(n, win)  # noqa: E731
        lower = np.array([get("H min"), get("S min"), get("V min")], np.uint8)
        upper = np.array([get("H max"), get("S max"), get("V max")], np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        cv2.imshow(win, np.hstack([image, cv2.bitwise_and(image, image, mask=mask)]))
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    print(f"lower: {list(lower)}\nupper: {list(upper)}")


def main() -> int:
    p = argparse.ArgumentParser(description="Подбор HSV-порогов «яблок»")
    p.add_argument("--grab", metavar="ФАЙЛ", help="снять кадр с камеры дрона и выйти")
    p.add_argument("--image", help="кадр для подбора порогов")
    p.add_argument("--live", action="store_true", help="взять кадр с камеры дрона прямо сейчас")
    p.add_argument("--roi", help="область яблока: x,y,w,h (без GUI)")
    p.add_argument("--color", help="в какой профиль записать результат (red/green/yellow)")
    p.add_argument("--save", action="store_true", help="записать пороги в конфиг")
    p.add_argument("--config", default="", help="путь к конфигу")
    p.add_argument("--trackbars", action="store_true", help="подбор ползунками")
    p.add_argument("--margin-h", type=int, default=6, help="запас по оттенку")
    p.add_argument("--margin-sv", type=int, default=40, help="запас по насыщенности и яркости")
    p.add_argument("--percentile", type=float, default=5.0, help="отсекаемые хвосты распределения, %%")
    args = p.parse_args()

    if args.grab:
        return grab_frame(args.grab)

    if args.live:
        try:
            import sverk_interfaces
        except ImportError:
            print("sverk_interfaces не найден — используйте --image")
            return 1
        drone = sverk_interfaces.init(Nodename="hsv_calib")
        image = drone.image.take_picture(timeout=5.0)
        drone.close()
        if image is None:
            print("Кадр не получен")
            return 1
    elif args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Не читается файл: {args.image}")
            return 1
    else:
        p.error("нужен --image, --live или --grab")

    if args.trackbars:
        trackbars(image)
        return 0

    if args.roi:
        try:
            x, y, w, h = (int(v) for v in args.roi.split(","))
        except ValueError:
            print("--roi задаётся как x,y,w,h")
            return 1
    else:
        roi = select_roi(image)
        if roi is None:
            return 1
        x, y, w, h = (int(v) for v in roi)

    patch = image[y:y + h, x:x + w]
    if patch.size == 0:
        print("Пустая область")
        return 1

    lower, upper, wraps = hsv_bounds(patch, args.margin_h, args.margin_sv, args.percentile)
    ranges = ranges_from_bounds(lower, upper, wraps)

    print(f"\nОбласть {w}x{h} px, {patch.shape[0] * patch.shape[1]} пикселей")
    print(f"Оттенок {'заворачивается через 0 (красный)' if wraps else 'сплошной'}")
    for lo, hi in ranges:
        print(f"  lower: {list(lo)}   upper: {list(hi)}")

    if not args.save:
        print("\nЧтобы записать в конфиг, добавьте --color <имя> --save")
        return 0

    if not args.color:
        print("Для --save укажите --color")
        return 1

    cfg = load_config(args.config) if args.config else load_config()
    try:
        profile = cfg.profile(args.color)
        profile.ranges = ranges
    except KeyError:
        cfg.profiles.append(ColorProfile(name=args.color, label=args.color, ranges=ranges))
        profile = cfg.profile(args.color)

    path = save_config(cfg, args.config)
    print(f"\nПрофиль '{args.color}' обновлён в {path}")

    detections = AppleDetector(cfg, colors=(args.color,)).detect(image)
    if detections:
        d = detections[0]
        print(f"Проверка на этом кадре: найдено, центр {int(d.center[0])},{int(d.center[1])}, "
              f"площадь {int(d.area)} px², оценка {d.score:.2f}")
    else:
        print("Проверка на этом кадре: НЕ найдено — ослабьте фильтры формы или увеличьте запас "
              "(--margin-sv), проверьте min_area в конфиге")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
