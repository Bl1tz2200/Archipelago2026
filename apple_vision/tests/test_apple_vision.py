"""Тесты детектора и логики однократного зачёта. Дрон и камера не нужны.

    python3 -m pytest tests/ -q      (или просто: python3 tests/test_apple_vision.py)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apple_vision import AppleDetector, AppleRegistry, load_config  # noqa: E402
from apple_vision.geometry import GroundProjector, intrinsics_from_fov  # noqa: E402
from apple_vision.synthetic import make_flight, make_frame  # noqa: E402

CFG = load_config()


def test_finds_each_color():
    detector = AppleDetector(CFG)
    for color in ("red", "green", "yellow"):
        frame = make_frame([(color, (320, 240), 36)])
        detections = detector.detect(frame)
        assert detections, f"{color}: яблоко не найдено"
        assert detections[0].color == color
        assert abs(detections[0].center[0] - 320) < 12
        assert abs(detections[0].center[1] - 240) < 12


def test_finds_three_at_once():
    detector = AppleDetector(CFG)
    frame = make_frame([("red", (150, 150), 34), ("green", (400, 260), 32), ("yellow", (520, 120), 30)])
    colors = {d.color for d in detector.detect(frame)}
    assert colors == {"red", "green", "yellow"}, colors


def test_ignores_non_round_objects():
    """Цветная полоса и линия в кадре есть всегда, но яблоками считаться не должны."""
    detector = AppleDetector(CFG)
    assert detector.detect(make_frame([])) == []


def test_ignores_small_specks():
    detector = AppleDetector(CFG)
    assert detector.detect(make_frame([("red", (320, 240), 5)])) == []


def test_survives_light_changes():
    detector = AppleDetector(CFG)
    for brightness in (90, 120, 150, 190):
        frame = make_frame([("red", (300, 220), 36)], brightness=brightness)
        assert detector.detect(frame), f"яркость фона {brightness}: яблоко потеряно"


def test_registry_needs_confirmation():
    """Одного кадра мало — иначе блик поднимет дрон впустую."""
    detector = AppleDetector(CFG)
    registry = AppleRegistry(CFG.registry)
    frame = make_frame([("red", (320, 240), 36)])

    events = registry.update(detector.detect(frame))
    assert events == []
    for _ in range(CFG.registry.confirm_frames - 2):
        assert registry.update(detector.detect(frame)) == []
    events = registry.update(detector.detect(frame))
    assert len(events) == 1 and events[0].color == "red" and events[0].index == 1


def test_registry_counts_each_apple_once():
    """Регламент: повторное обнаружение засчитанного яблока дрон не поднимает."""
    detector = AppleDetector(CFG)
    registry = AppleRegistry(CFG.registry)
    frame = make_frame([("red", (320, 240), 36)])

    for _ in range(40):
        registry.update(detector.detect(frame))
    assert registry.count == 1, "красное яблоко засчитано больше одного раза"


def test_full_flight_gives_three_events():
    detector = AppleDetector(CFG)
    registry = AppleRegistry(CFG.registry)
    order = []
    for frame, _ in make_flight():
        for event in registry.update(detector.detect(frame)):
            order.append(event.color)
    assert order == ["red", "green", "yellow"], order
    assert registry.complete


def test_registry_stops_at_max():
    registry = AppleRegistry(CFG.registry)
    detector = AppleDetector(CFG)
    for frame, _ in make_flight(colors=("red", "green", "yellow", "red", "green")):
        registry.update(detector.detect(frame))
    assert registry.count == CFG.registry.max_apples


def test_ground_projection():
    """Центр кадра — прямо под дроном; смещение вверх по кадру — вперёд по курсу."""
    projector = GroundProjector(CFG.camera, intrinsics_from_fov(640, 480, 65.0))
    size = (640, 480)

    fx, fy = projector.pixel_to_body((320, 240), 1.5, size)
    assert abs(fx) < 1e-6 and abs(fy) < 1e-6

    forward, left = projector.pixel_to_body((320, 140), 1.5, size)
    assert forward > 0.2 and abs(left) < 1e-6

    world = projector.pixel_to_map((320, 240), (2.0, -1.0), 0.0, 1.5, size)
    assert abs(world[0] - 2.0) < 1e-6 and abs(world[1] + 1.0) < 1e-6

    # Высота ~0 — проекция невозможна, дедупликация по месту просто не применяется.
    assert projector.pixel_to_body((320, 240), 0.0, size) is None


def test_projection_scales_with_altitude():
    projector = GroundProjector(CFG.camera, intrinsics_from_fov(640, 480, 65.0))
    low = projector.pixel_to_body((420, 240), 1.0, (640, 480))
    high = projector.pixel_to_body((420, 240), 2.0, (640, 480))
    assert abs(high[1] - 2 * low[1]) < 1e-6


def test_calibration_survives_background_in_roi():
    """Калибровка по рамке: в неё всегда попадает фон, пороги должны остаться цветом яблока."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    from calibrate_hsv import hsv_bounds, ranges_from_bounds  # noqa: WPS433

    from copy import deepcopy

    for color in ("red", "green", "yellow"):
        frame = make_frame([(color, (320, 240), 40)])
        patch = frame[187:292, 267:372]           # рамка шире яблока — фон внутри есть
        lower, upper, wraps = hsv_bounds(patch, margin_h=8, margin_sv=60, percentile=5.0)

        cfg = deepcopy(CFG)
        cfg.profile(color).ranges = ranges_from_bounds(lower, upper, wraps)
        detections = AppleDetector(cfg, colors=(color,)).detect(frame)
        assert detections, f"{color}: после калибровки по рамке яблоко не находится"
        assert abs(detections[0].center[0] - 320) < 12


def test_saving_keeps_comments_and_values():
    """Сохранение из окна не должно превращать конфиг в нечитаемый список чисел."""
    import tempfile

    from apple_vision import save_config

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "apples.yaml")
        save_config(CFG, path)
        text = open(path, encoding="utf-8").read()
        assert text.count("#") > 20, "комментарии потерялись при сохранении"
        assert "min_size_percent" in text and "{lower: [" in text, "формат стал нечитаемым"

        again = load_config(path)
        assert [p.name for p in again.profiles] == [p.name for p in CFG.profiles]
        assert again.profile("red").ranges == CFG.profile("red").ranges
        assert again.registry.confirm_frames == CFG.registry.confirm_frames
        assert again.camera.calibration_file == CFG.camera.calibration_file


def test_old_field_names_still_load():
    """Конфиг с прежними именами (min_area в пикселях) читается без правки руками."""
    from apple_vision.config import ColorProfile

    profile = ColorProfile.from_dict({
        "name": "red",
        "ranges": [{"lower": [0, 110, 70], "upper": [8, 255, 255]}],
        "min_area": 400, "max_area": 120000,
        "min_circularity": 0.55, "min_solidity": 0.8, "min_fill": 0.45,
    })
    assert abs(profile.min_size_percent - 0.13) < 0.02   # 400 px² при 640x480
    assert profile.min_roundness == 0.55 and profile.min_filling == 0.45


def test_size_threshold_is_resolution_independent():
    """Один и тот же порог в процентах работает на кадрах разного размера."""
    percents = []
    for size in ((640, 480), (1280, 960), (1920, 1440)):     # одна пропорция, разный масштаб
        radius = int(size[0] * 0.055)
        frame = make_frame([("red", (size[0] // 2, size[1] // 2), radius)], size=size)
        detections = AppleDetector(CFG).detect(frame)
        assert detections, f"{size}: яблоко не найдено"
        percents.append(detections[0].size_percent)

    # Яблоко занимает одну и ту же долю кадра — значит и порог переносится без правки.
    assert max(percents) - min(percents) < 0.1, percents


def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"  ok   {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} пройдено")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
