#!/usr/bin/env python3
"""Что камера видит на поле меток — проверка перед полётом.

На дроне (пропеллеры сняты), держим дрон над полем:

    python3 tools/marker_check.py --duration 30

Печатает ID найденных меток в том порядке, в каком они лежат в кадре, узлы поля,
полосу обзора в метках и стартовый (угловой) маркер. По этой картинке сразу видно:
  * тот ли словарь ArUco стоит в конфиге;
  * совпадает ли раскладка с markers.row_major / flip_x / flip_y;
  * какое разрежение проходов получит планировщик на этой высоте.

Без дрона — самопроверка на синтетическом поле:

    python3 tools/marker_check.py --synthetic
    python3 tools/marker_check.py --image кадр.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from snake_mission import field, markers  # noqa: E402
from snake_mission.config import load_config  # noqa: E402


def report(frame, detector, numbering, prefix: str = "") -> int:
    size = (frame.shape[1], frame.shape[0])
    sights = detector.detect(frame)
    if not sights:
        print(f"{prefix}меток не видно")
        return 0

    print(f"{prefix}меток в кадре: {len(sights)}")
    print(markers.layout(sights, size))

    nodes = markers.to_nodes(sights, numbering)
    print(f"{prefix}узлы поля: {sorted(nodes)}")
    pitch = markers.marker_pitch(sights, numbering)
    span = markers.visible_span(sights, numbering, size)
    print(f"{prefix}шаг сетки в кадре: "
          f"{'—' if pitch is None else format(pitch, '.0f') + ' px'}")
    print(f"{prefix}полоса обзора: {span[0]}×{span[1]} меток "
          f"(радиус проходов {max(0, (span[1] - 1) // 2)})")

    under = markers.nearest_to_center(sights, size)
    if under is not None:
        _, _, offset = markers.offset_from_center(under, size)
        print(f"{prefix}под дроном метка {under.id} → узел {numbering.node_of(under.id)}, "
              f"смещение от центра {offset:.3f} диагонали")
    corner = markers.start_marker(sights, size, numbering)
    if corner is not None:
        print(f"{prefix}угловая метка {corner.id} → фигура: {field.figure_for(corner.id) or '—'}")
    else:
        print(f"{prefix}угловых меток (0/6/42/48) в кадре нет")
    return len(sights)


def main() -> int:
    p = argparse.ArgumentParser(description="Проверка поля ArUco-меток")
    p.add_argument("--config", default="", help="путь к YAML-конфигу")
    p.add_argument("--duration", type=float, default=20.0, help="сколько секунд смотреть, с")
    p.add_argument("--period", type=float, default=2.0, help="как часто печатать отчёт, с")
    p.add_argument("--image", default="", help="разобрать картинку вместо камеры")
    p.add_argument("--synthetic", action="store_true", help="самопроверка на синтетическом поле")
    p.add_argument("--out", default="", help="куда сохранить разметку")
    p.add_argument("--no-publish", action="store_true", help="не публиковать в /out_detection")
    args = p.parse_args()

    config = load_config(args.config)
    numbering = config.markers.numbering()
    detector = markers.MarkerDetector(config.markers.dictionary,
                                      config.markers.min_marker_percent)

    if args.synthetic or args.image:
        if args.synthetic:
            from snake_mission.synthetic import make_view
            frame = make_view(center_node=(0, 0), spacing_px=150)
        else:
            frame = cv2.imread(args.image)
            if frame is None:
                print(f"{args.image}: не читается")
                return 1
        report(frame, detector, numbering)
        if args.out:
            cv2.imwrite(args.out, markers.draw(frame, detector.detect(frame)))
            print(f"разметка: {args.out}")
        return 0

    try:
        import sverk_interfaces
    except ImportError:
        print("sverk_interfaces не найден — запускайте на дроне или используйте --synthetic")
        return 1

    drone = sverk_interfaces.init(Nodename="marker_check")
    deadline = time.time() + args.duration
    seen = 0
    try:
        while time.time() < deadline:
            frame = drone.image.take_picture(timeout=2.0)
            if frame is None:
                print("кадр не получен: ros2 topic hz /camera_1/image_raw")
                time.sleep(args.period)
                continue
            print("─" * 60)
            seen += report(frame, detector, numbering)
            if not args.no_publish:
                try:
                    drone.image.publish(markers.draw(frame, detector.detect(frame)))
                except Exception as exc:
                    print(f"публикация не прошла: {exc}")
            time.sleep(args.period)
    except KeyboardInterrupt:
        print("\nостановлено оператором")
    finally:
        drone.close()

    return 0 if seen else 2


if __name__ == "__main__":
    raise SystemExit(main())
