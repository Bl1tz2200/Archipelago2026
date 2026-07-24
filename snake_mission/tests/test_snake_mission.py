"""Тесты поля, поискового маршрута и распознавания меток. Дрон и камера не нужны.

    python3 -m pytest tests/ -q      (или просто: python3 tests/test_snake_mission.py)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snake_mission import field, markers, search  # noqa: E402
from snake_mission.config import (  # noqa: E402
    ALTITUDE_CEILING_M,
    FlightConfig,
    MarkersConfig,
    SearchConfig,
    load_config,
)
from snake_mission.field import Numbering  # noqa: E402
from snake_mission.synthetic import make_view  # noqa: E402

CORNERS = (0, 6, 42, 48)
SPANS = ((1, 1), (2, 2), (3, 3), (4, 3), (5, 5), (7, 7))


# ─────────────────────────────────────────────────────────── поле

def test_corner_ids_match_regulation():
    """Приложение 3: ID 0, 42, 48, 6 — это углы сетки 7×7 при нумерации ID = row*7 + col."""
    assert field.node_of(0) == (0, 0)
    assert field.node_of(6) == (6, 0)
    assert field.node_of(42) == (0, 6)
    assert field.node_of(48) == (6, 6)
    assert set(field.corner_ids()) == set(CORNERS)
    for mid in CORNERS:
        assert field.is_corner(field.node_of(mid)), mid
        assert field.figure_for(mid), mid


def test_id_and_node_are_inverse():
    for mid in range(field.MARKERS):
        assert field.marker_id(field.node_of(mid)) == mid


def test_numbering_variants_stay_consistent():
    """Зеркальная и транспонированная раскладки остаются взаимно однозначными."""
    for numbering in (Numbering(row_major=False),
                      Numbering(flip_x=True),
                      Numbering(flip_y=True),
                      Numbering(row_major=False, flip_x=True, flip_y=True)):
        seen = set()
        for mid in range(field.MARKERS):
            node = numbering.node_of(mid)
            assert field.inside(node)
            assert numbering.marker_id(node) == mid
            seen.add(node)
        assert len(seen) == field.MARKERS
        # Углы поля остаются углами при любой нумерации — меняются только их ID.
        assert all(field.is_corner(numbering.node_of(mid)) for mid in numbering_corners(numbering))


def numbering_corners(numbering: Numbering):
    return [numbering.marker_id(n) for n in ((0, 0), (6, 0), (0, 6), (6, 6))]


def test_field_boundaries():
    assert field.inside((0, 0)) and field.inside((6, 6))
    for outside in ((-1, 0), (0, -1), (7, 0), (0, 7), (7, 7)):
        assert not field.inside(outside), outside


def test_legal_step_follows_snake_model():
    """Регламент: только соседние узлы, включая диагональные; поле не покидаем."""
    assert field.is_legal_step((3, 3), (4, 3))      # прямо
    assert field.is_legal_step((3, 3), (4, 4))      # диагональ, поворот 45°
    assert not field.is_legal_step((3, 3), (5, 3))  # через узел
    assert not field.is_legal_step((3, 3), (3, 3))  # на месте
    assert not field.is_legal_step((6, 6), (7, 6))  # за границу поля
    assert len(field.neighbors((0, 0))) == 3        # в углу соседей меньше
    assert len(field.neighbors((3, 3))) == 8


def test_straight_path_is_legal_and_minimal():
    for a, b in (((0, 0), (6, 6)), ((6, 0), (0, 3)), ((2, 5), (2, 0)), ((1, 1), (1, 1))):
        path = field.straight_path(a, b)
        assert len(path) == field.chebyshev(a, b), (a, b)
        search.validate([a] + path)
        if path:
            assert path[-1] == b


# ─────────────────────────────────────────────────── поисковый маршрут

def test_route_is_legal_from_every_corner():
    for mid in CORNERS:
        for span in SPANS:
            plan = search.plan(field.node_of(mid), span)
            search.validate(plan.route)          # бросит RouteError, если шаг нелегален
            assert plan.route[0] == field.node_of(mid)


def test_route_covers_whole_field():
    for mid in CORNERS:
        for span in SPANS:
            plan = search.plan(field.node_of(mid), span)
            missing = set(field.all_nodes()) - plan.coverage()
            assert not missing, f"маркер {mid}, полоса {span}: не покрыто {sorted(missing)}"


def test_wider_view_means_fewer_steps():
    """Чем шире полоса обзора, тем короче маршрут — в этом весь смысл разрежения проходов."""
    narrow = search.plan((0, 0), (1, 1)).steps
    wide = search.plan((0, 0), (3, 3)).steps
    assert wide < narrow, (wide, narrow)
    assert search.plan((0, 0), (5, 5)).steps <= wide


def test_pass_spacing_is_capped():
    """Даже если камера видит всё поле, проходы не разрежаются шире max_pass_spacing."""
    config = SearchConfig(max_pass_spacing=3)
    assert search.reach_from_span(7, 0, config.max_pass_spacing) == 1
    assert search.reach_from_span(3, 1, config.max_pass_spacing) == 0   # запас = проход по ряду
    plan = search.plan((0, 0), (7, 7), config)
    assert len(plan.coverage()) == field.MARKERS


def test_pass_lines_cover_all_rows():
    for reach in range(0, 4):
        lines = search.pass_lines(reach)
        covered = {row for line in lines for row in range(line - reach, line + reach + 1)}
        assert set(range(field.SIDE)) <= covered, (reach, lines)
        assert all(0 <= line < field.SIDE for line in lines)


def test_validate_rejects_illegal_route():
    for bad in ([(0, 0), (2, 0)], [(0, 0), (0, 0)], [(0, 0), (-1, 0)], []):
        try:
            search.validate(bad)
        except search.RouteError:
            continue
        raise AssertionError(f"маршрут {bad} должен быть отклонён")


def test_resume_keeps_remaining_nodes():
    plan = search.plan((0, 0), (3, 3))
    for index in (0, 1, 5, len(plan.route) - 1, len(plan.route)):
        tail = search.resume(plan.route, index)
        assert tail == list(plan.route[index:])
        if len(tail) > 1:
            search.validate(tail)


def test_estimate_flags_budget_overrun():
    narrow = search.estimate(search.plan((0, 0), (1, 1)), budget_s=180.0)
    wide = search.estimate(search.plan((0, 0), (3, 3)), budget_s=180.0)
    assert wide.total_s < narrow.total_s
    assert wide.fits and not narrow.fits


def test_route_ends_within_reach_of_start():
    """После сборки лидер обязан привести цепочку на стартовую позицию."""
    for mid in CORNERS:
        plan = search.plan(field.node_of(mid), (3, 3))
        assert plan.return_steps <= field.SIDE - 1
        search.validate([plan.end] + field.straight_path(plan.end, plan.start))


# ───────────────────────────────────────────────────────── метки

def test_detects_synthetic_field():
    detector = markers.MarkerDetector()
    frame = make_view(center_node=(3, 3), spacing_px=150)
    sights = detector.detect(frame)
    assert sights, "метки не найдены"
    ids = {s.id for s in sights}
    assert field.marker_id((3, 3)) in ids
    for sight in sights:
        assert 0 <= sight.id < field.MARKERS


def test_marker_under_drone_is_nearest_to_center():
    detector = markers.MarkerDetector()
    for node in ((3, 3), (0, 0), (6, 6), (2, 5)):
        frame = make_view(center_node=node, spacing_px=150)
        size = (frame.shape[1], frame.shape[0])
        under = markers.nearest_to_center(detector.detect(frame), size)
        assert under is not None and under.id == field.marker_id(node), node


def test_visible_span_grows_when_markers_are_closer():
    """Шаг меток в кадре — это высота: чем выше дрон, тем больше меток и шире полоса."""
    detector = markers.MarkerDetector()
    spans = []
    for spacing in (200, 150, 100, 70):
        sights = detector.detect(make_view(center_node=(3, 3), spacing_px=spacing))
        spans.append(markers.visible_span(sights))
    rows = [s[1] for s in spans]
    assert rows == sorted(rows), rows
    assert rows[-1] > rows[0]


def test_start_marker_prefers_corner():
    """В кадре несколько меток, но стартовой считается угловая — она задаёт фигуру."""
    detector = markers.MarkerDetector()
    for mid in CORNERS:
        frame = make_view(center_node=field.node_of(mid), spacing_px=150)
        size = (frame.shape[1], frame.shape[0])
        sights = detector.detect(frame)
        assert len(sights) > 1, "в кадре должна быть не одна метка"
        start = markers.start_marker(sights, size)
        assert start is not None and start.id == mid, (mid, start and start.id)
        assert field.figure_for(start.id)


def test_no_corner_in_the_middle_of_the_field():
    detector = markers.MarkerDetector()
    frame = make_view(center_node=(3, 3), spacing_px=150)
    size = (frame.shape[1], frame.shape[0])
    assert markers.start_marker(detector.detect(frame), size) is None


def test_offset_from_center_grows_with_shift():
    detector = markers.MarkerDetector()
    size = None
    offsets = []
    for shift in (0, 40, 120):
        frame = make_view(center_node=(3, 3), spacing_px=150, offset_px=(shift, 0))
        size = (frame.shape[1], frame.shape[0])
        sight = markers.find(detector.detect(frame), field.marker_id((3, 3)))
        assert sight is not None, shift
        offsets.append(markers.offset_from_center(sight, size)[2])
    assert offsets == sorted(offsets), offsets
    assert offsets[0] < 0.02 < offsets[-1]


def test_marker_and_apple_detectors_share_a_frame():
    """Сквозная проверка: на одном кадре видно и метки, и «яблоко»."""
    from apple_vision import AppleDetector
    from snake_mission.synthetic import make_apple_view

    frame = make_apple_view(center_node=(3, 3), apple_color="red", spacing_px=170)
    assert markers.MarkerDetector().detect(frame), "метки потерялись"
    apples = AppleDetector().detect(frame)
    assert apples and apples[0].color == "red", "яблоко не найдено на кадре с метками"


# ────────────────────────────────────────────────────────── конфиг

def test_altitude_ceiling_is_enforced():
    """Защитная сетка 4 м → выше 3 м конфиг не принимает."""
    FlightConfig(altitude=ALTITUDE_CEILING_M)
    for bad in (ALTITUDE_CEILING_M + 0.1, 4.0, 0.0, -1.0):
        try:
            FlightConfig(altitude=bad)
        except ValueError:
            continue
        raise AssertionError(f"высота {bad} должна быть отклонена")


def test_shipped_config_is_valid():
    config = load_config()
    assert config.flight.altitude <= ALTITUDE_CEILING_M
    assert config.formation_budget <= 180.0
    assert config.markers.dictionary.startswith("DICT_")
    plan = search.plan(field.node_of(0), (config.search.default_span,) * 2, config.search)
    assert len(plan.coverage()) == field.MARKERS
    assert search.estimate(plan, config.search, budget_s=config.formation_budget).fits


def test_markers_config_builds_numbering():
    numbering = MarkersConfig(row_major=False, flip_x=True).numbering()
    assert numbering.node_of(numbering.marker_id((2, 4))) == (2, 4)


# ──────────────────────────────────────────── симулятор: миссия целиком

def _sim_config(scale: float):
    """Конфиг для прогона в симуляторе: время сжато, поэтому сжаты и все паузы."""
    config = load_config()
    config.formation_budget /= scale
    config.swarm.join_wait_s = 0.2
    config.swarm.join_timeout = 2.0
    config.search.step_time_s /= scale
    config.search.stabilize_time_s /= scale
    config.search.apple_hold_s = 0.3
    return config


_FLIGHTS = {}


def _fly(start_marker: int = 0, scale: float = 25.0, altitude: float = 0.0):
    """Прогон миссии в симуляторе. Результат кэшируется: полёт не быстрый."""
    import contextlib
    import io

    key = (start_marker, scale, altitude)
    if key in _FLIGHTS:
        return _FLIGHTS[key]

    from snake_mission.mission import LeaderMission
    from snake_mission.simulator import SimDrone, SimWorld, default_apples

    config = _sim_config(scale)
    if altitude:
        config.flight.altitude = altitude
        config.flight.__post_init__()
    numbering = config.markers.numbering()
    world = SimWorld(time_scale=scale, speed=config.flight.speed)
    drone = SimDrone(world, start=numbering.node_of(start_marker),
                     apples=default_apples(), numbering=numbering)
    mission = LeaderMission(drone, config=config, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):   # вывод миссии тесту не нужен
        result = mission.run()

    _FLIGHTS[key] = (mission, drone, result)
    return _FLIGHTS[key]


def test_simulated_mission_reads_marker_finds_apples_and_lands():
    """Сквозной прогон: кадры → метки и яблоки → перелёты → зависания → возврат → посадка."""
    mission, drone, result = _fly(start_marker=42)
    assert result.start_marker == 42, result.start_marker
    assert result.figure == field.FIGURES[42]
    assert result.apples == 3, mission.vision.summary()
    assert result.returned and result.landed
    assert result.steps_done > 0
    search.validate(mission.plan.route)


def test_simulated_apple_positions_match_the_field():
    """Координаты яблок в логе совпадают с тем, где они лежат на поле."""
    from snake_mission.simulator import SimWorld

    mission, drone, result = _fly(start_marker=42)
    step = SimWorld().step_m
    for event in mission.vision.events:
        assert event.world is not None, event.describe()
        expected = drone.apples[event.color]
        error = max(abs(event.world[0] - expected[0] * step),
                    abs(event.world[1] - expected[1] * step))
        assert error < 0.35, f"{event.color}: {event.world} вместо {expected}"


def test_simulated_flight_never_leaves_the_field():
    """Регламент: полёт только в пределах полётной зоны."""
    mission, drone, result = _fly(start_marker=48)
    for col, row in drone.track:
        assert -0.4 <= col <= field.SIDE - 0.6, (col, row)
        assert -0.4 <= row <= field.SIDE - 0.6, (col, row)


def test_simulated_span_is_the_same_at_corner_and_in_the_middle():
    """Над углом видна только четверть поля, но полоса обзора та же — она меряется по шагу сетки."""
    import time

    from snake_mission.simulator import SimDrone, SimWorld

    detector = markers.MarkerDetector()
    spans = []
    for start in ((0, 0), (3, 3)):
        drone = SimDrone(SimWorld(), start=start)
        drone.command("body", 0.0, 0.0, 2.5, True)
        while abs(drone.altitude - 2.5) > 0.02:
            time.sleep(0.01)
            drone.telemetry()
        frame = drone.render()
        sights = detector.detect(frame)
        spans.append(markers.visible_span(sights, frame_size=(frame.shape[1], frame.shape[0])))
    assert spans[0] == spans[1], spans
    assert spans[0][1] >= 3, spans


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
