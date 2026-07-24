#!/usr/bin/env python3
"""Проверка распознавания на обычной вебке: показываете яблоко — оно обводится.

Тот же детектор и та же логика зачёта, что полетят на дроне, только кадры берутся
не из `/camera_1/image_raw`, а с камеры ноутбука. Удобно, чтобы подобрать пороги
под настоящее яблоко до выезда на площадку.

    python3 tools/webcam_demo.py                 # окно с разметкой
    python3 tools/webcam_demo.py --web           # без окна: http://localhost:8000
    python3 tools/webcam_demo.py --camera 1      # другая камера

Клавиши в окне:
    1 2 3   выбрать цвет для калибровки (red / green / yellow)
    c       откалибровать выбранный цвет по рамке в центре кадра
    + -     порог минимального размера: ловит лишнее — увеличить
    s       сохранить подобранные пороги в config/apples.yaml
    m       показать/скрыть маски по цветам
    r       сбросить зачёт (как новая попытка)
    q/Esc   выход

На macOS при первом запуске система спросит доступ к камере. Если запрос не
появился или камера не открылась — Системные настройки → Конфиденциальность и
безопасность → Камера → разрешить вашему терминалу, затем перезапустить терминал.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apple_vision import AppleDetector, AppleVision, draw_detections, load_config, save_config  # noqa: E402
from apple_vision.overlay import stack_masks  # noqa: E402
from calibrate_hsv import hsv_bounds, ranges_from_bounds  # noqa: E402

CALIB_COLORS = ("red", "green", "yellow")
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def open_camera(index: int, width: int, height: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Камера {index} не открылась.")
        print("macOS: Системные настройки → Конфиденциальность и безопасность → Камера —")
        print("разрешите доступ вашему терминалу и перезапустите его.")
        print("Другая камера: --camera 1")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def centre_roi(frame: np.ndarray, side_ratio: float = 0.22) -> Tuple[int, int, int, int]:
    """Квадрат в центре кадра — в него наводят яблоко при калибровке."""
    h, w = frame.shape[:2]
    side = int(min(h, w) * side_ratio)
    return (w - side) // 2, (h - side) // 2, side, side


def draw_hud(canvas: np.ndarray, vision: AppleVision, color: str, dirty: bool,
             hint: str = "") -> np.ndarray:
    x, y, w, h = centre_roi(canvas)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (180, 180, 180), 1)

    size = vision.config.profiles[0].min_size_percent if vision.config.profiles else 0.0
    lines = [
        f"калибровка: {color}   1/2/3 - цвет, c - взять из рамки" + ("  *не сохранено" if dirty else ""),
        f"min размер: {size:.2f}% кадра   +/- изменить (ловит лишнее -> больше)",
        "s - сохранить в конфиг | m - маски | r - сброс зачёта | q - выход",
    ]
    if hint:
        lines.insert(0, hint)

    height = 18 * len(lines) + 8
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, canvas.shape[0] - height), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (8, canvas.shape[0] - height + 18 + i * 18),
                    _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def calibrate_from_centre(vision: AppleVision, frame: np.ndarray, color: str,
                          margin_h: int, margin_sv: int) -> str:
    """Подстраивает пороги цвета по содержимому центральной рамки."""
    x, y, w, h = centre_roi(frame)
    patch = frame[y:y + h, x:x + w]
    lower, upper, wraps = hsv_bounds(patch, margin_h, margin_sv, percentile=5.0)
    ranges = ranges_from_bounds(lower, upper, wraps)
    try:
        vision.config.profile(color).ranges = ranges
    except KeyError:
        return f"нет профиля '{color}' в конфиге"
    # Пересобираем детектор, чтобы новые пороги действовали со следующего кадра.
    vision.detector = AppleDetector(vision.config, [p.name for p in vision.detector.profiles])
    return f"{color}: H {lower[0]}..{upper[0]}  S {lower[1]}..{upper[1]}  V {lower[2]}..{upper[2]}"


def run_window(cap: cv2.VideoCapture, vision: AppleVision, args) -> int:
    color_idx = 0
    show_masks = False
    dirty = False
    hint = ""
    hint_until = 0.0
    win = "apple_vision — вебка"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Кадр не получен, камера отвалилась")
            return 1
        if args.mirror:
            frame = cv2.flip(frame, 1)

        vision.process(frame)
        canvas = draw_detections(frame, vision.last_detections, vision.registry,
                                 f"{vision.fps:.1f} к/с")
        if time.time() > hint_until:
            hint = ""
        canvas = draw_hud(canvas, vision, CALIB_COLORS[color_idx], dirty, hint)

        if show_masks:
            grid = stack_masks(vision.detector.last_masks, width=canvas.shape[1])
            if grid is not None:
                canvas = np.vstack([canvas, grid])

        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord("1"), ord("2"), ord("3")):
            color_idx = key - ord("1")
            hint, hint_until = f"калибруем: {CALIB_COLORS[color_idx]}", time.time() + 2
        elif key == ord("c"):
            hint = calibrate_from_centre(vision, frame, CALIB_COLORS[color_idx],
                                         args.margin_h, args.margin_sv)
            hint_until = time.time() + 4
            dirty = True
            print(hint, flush=True)
        elif key == ord("s"):
            path = save_config(vision.config)
            hint, hint_until = f"сохранено: {path}", time.time() + 3
            dirty = False
            print(hint, flush=True)
        elif key in (ord("+"), ord("="), ord("-"), ord("_")):
            # Порог размера общий для всех цветов: яблоки на поле примерно одинаковы.
            step = 1.25 if key in (ord("+"), ord("=")) else 1 / 1.25
            for profile in vision.config.profiles:
                profile.min_size_percent = round(
                    min(20.0, max(0.005, profile.min_size_percent * step)), 3)
            vision.detector = AppleDetector(vision.config,
                                            [p.name for p in vision.detector.profiles])
            dirty = True
            hint, hint_until = (f"min размер: {vision.config.profiles[0].min_size_percent:.3f}% кадра",
                                time.time() + 2)
        elif key == ord("m"):
            show_masks = not show_masks
        elif key == ord("r"):
            vision.registry.reset()
            hint, hint_until = "зачёт сброшен", time.time() + 2

    cv2.destroyAllWindows()
    return 0


def run_web(cap: cv2.VideoCapture, vision: AppleVision, args) -> int:
    """Фолбэк без GUI: MJPEG-поток в браузере."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # без шума в консоли
            pass

        def do_GET(self):
            if self.path != "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<body style='margin:0;background:#111'>"
                                 b"<img src='/stream' style='width:100%'></body>")
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if args.mirror:
                        frame = cv2.flip(frame, 1)
                    vision.process(frame)
                    canvas = draw_detections(frame, vision.last_detections, vision.registry,
                                             f"{vision.fps:.1f} к/с")
                    ok, buf = cv2.imencode(".jpg", canvas)
                    if not ok:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n"
                                     + buf.tobytes() + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Откройте http://localhost:{args.port}  (Ctrl+C — выход)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Распознавание яблок с вебки")
    p.add_argument("--camera", type=int, default=0, help="индекс камеры")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--config", default="")
    p.add_argument("--colors", default="", help="ограничить цвета, например red,green")
    p.add_argument("--web", action="store_true", help="без окна: MJPEG в браузере")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mirror", action="store_true",
                   help="зеркалить картинку по горизонтали (как в зеркале)")
    p.add_argument("--margin-h", type=int, default=8, help="запас по оттенку при калибровке")
    p.add_argument("--margin-sv", type=int, default=60, help="запас по S/V при калибровке")
    p.add_argument("--snapshot", help="сохранить один кадр с разметкой и выйти")
    p.add_argument("--min-size", type=float, default=None,
                   help="минимальный размер яблока, %% площади кадра (для вебки обычно 0.5..2)")
    args = p.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    if args.min_size is not None:
        for profile in cfg.profiles:
            profile.min_size_percent = args.min_size
    colors = tuple(c.strip() for c in args.colors.split(",") if c.strip())
    vision = AppleVision(drone=None, config=cfg, colors=colors, publish=False, verbose=True)

    cap = open_camera(args.camera, args.width, args.height)
    if cap is None:
        return 1

    print(f"Ищем: {', '.join(pr.label for pr in vision.detector.profiles)}")
    print(f"Зачёт после {cfg.registry.confirm_frames} кадров подряд, всего {cfg.registry.max_apples}")

    try:
        if args.snapshot:
            for _ in range(10):     # даём камере прогреться и выставить экспозицию
                ok, frame = cap.read()
            if not ok:
                print("Кадр не получен")
                return 1
            if args.mirror:
                frame = cv2.flip(frame, 1)
            vision.process(frame)
            cv2.imwrite(args.snapshot, draw_detections(frame, vision.last_detections, vision.registry))
            print(f"Сохранено: {args.snapshot}, найдено объектов: {len(vision.last_detections)}")
            for d in vision.last_detections:
                print(f"  {d.color}: центр {int(d.center[0])},{int(d.center[1])} "
                      f"r={d.radius:.0f} score={d.score:.2f}")
            return 0

        if args.web:
            return run_web(cap, vision, args)
        return run_window(cap, vision, args)
    finally:
        cap.release()
        print()
        print(vision.summary())


if __name__ == "__main__":
    raise SystemExit(main())
