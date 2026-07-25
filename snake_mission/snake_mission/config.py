"""Конфигурация миссии лидера (YAML).

Сделано по образцу `apple_vision/apple_vision/config.py`: dataclass-секции, мягкое
чтение (лишние ключи игнорируются, отсутствующие берутся по умолчанию) и жёсткие
проверки там, где ошибка означает нарушение регламента — прежде всего высота.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

from .field import Numbering

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "snake.yaml"
)

# Регламент, п. 2.6.3: высота полёта не выше 4 м. Полёты идут в защитной сетке
# высотой 4 м, поэтому свой потолок держим ниже — 3 м.
ALTITUDE_CEILING_M = 3.0

# Регламент, п. 2.1.3: на сборку формации не более 3 минут с момента взлёта лидера.
FORMATION_BUDGET_S = 180.0

# Способы перелёта между узлами. Летаем по меткам: цель перелёта — метка, а не
# координата и не расстояние. `visual` — штатный (наведение по кадру), `aruco_frame` —
# через tf2-фрейм метки, `auto` — выбор между ними по доступности фреймов.
# `body_step` — аварийный: счисление на `step_m` метров, метки в управлении не
# участвуют; включать, только если меток не видно вовсе.
STRATEGIES = ("visual", "auto", "aruco_frame", "body_step")


@dataclass
class FlightConfig:
    altitude: float = 2.5          # рабочая высота поиска, м (потолок 3.0)
    speed: float = 0.6             # скорость перелёта между метками, м/с
    takeoff_timeout: float = 20.0  # сколько ждать выхода на высоту, с
    takeoff_pause: float = 8.0     # пауза после команды взлёта, если телеметрии нет, с
    yaw: float = 0.0               # курс держим постоянным весь поиск, рад
    land_at_end: bool = True       # садиться в конце (без этапа фигуры)
    auto_altitude: bool = True     # подняться ДО начала поиска, если маршрут не влезает в бюджет
    altitude_step: float = 0.5     # на сколько подниматься за раз, м

    def __post_init__(self) -> None:
        if self.altitude <= 0:
            raise ValueError("flight.altitude должна быть больше нуля")
        if self.altitude > ALTITUDE_CEILING_M:
            raise ValueError(
                f"flight.altitude = {self.altitude} м выше потолка {ALTITUDE_CEILING_M} м "
                "(защитная сетка 4 м, регламент п. 2.6.3)"
            )


@dataclass
class MarkersConfig:
    dictionary: str = "DICT_4X4_50"   # ID меток 0..48 → словарь 4x4 на 50 меток
    min_marker_percent: float = 0.02  # меньше этой доли кадра — метка слишком далеко
    row_major: bool = True            # нумерация меток: ID = row*7 + col
    flip_x: bool = False              # столбцы пронумерованы справа налево
    flip_y: bool = False              # строки пронумерованы сверху вниз

    def numbering(self) -> Numbering:
        return Numbering(row_major=self.row_major, flip_x=self.flip_x, flip_y=self.flip_y)


@dataclass
class NavigationConfig:
    strategy: str = "visual"           # visual | auto | aruco_frame | body_step (аварийный)
    step_m: float = 1.0                # шаг сетки меток на площадке, м (только body_step)
    settle_pause: float = 1.0          # запас на торможение после перелёта, с (только body_step)
    time_scale: float = 1.0            # во сколько раз сжаты паузы перелёта; >1 — только симулятор
    arrive_tolerance: float = 0.10     # попадание в узел: доля диагонали кадра
    center_tolerance: float = 0.06     # стабилизация: доля диагонали кадра
    hold_frames: int = 3               # кадров подряд в допуске до «стабилизирован»
    settle_speed: float = 0.15         # и горизонтальная скорость ниже, м/с
    goto_timeout: float = 12.0         # предел на перелёт до соседней метки, с
    stabilize_timeout: float = 3.0     # предел на стабилизацию над меткой, с
    lost_marker_timeout: float = 4.0   # метки не видно так долго → доводка вслепую
    frame_timeout: float = 1.0         # ожидание кадра с камеры, с
    poll_interval: float = 0.15        # период циклов управления, с
    frame_tolerance: float = 0.12      # попадание в узел по фрейму aruco_N, м

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"navigation.strategy = {self.strategy!r}; допустимо: {', '.join(STRATEGIES)}")
        if self.step_m <= 0:
            raise ValueError("navigation.step_m должен быть больше нуля")
        if self.time_scale <= 0:
            raise ValueError("navigation.time_scale должен быть больше нуля")


@dataclass
class SearchConfig:
    safety_markers: int = 0        # запас: вычитается из радиуса полосы обзора (1 = проход по каждому ряду)
    max_pass_spacing: int = 3      # шире не разрежаем проходы, даже если камера видит больше
    default_span: int = 3          # рядов меток в кадре, если измерить не удалось
    apple_hold_s: float = 12.0     # оценка зависания на одно яблоко, с (для расчёта времени)
    step_time_s: float = 2.2       # оценка одного перелёта, с
    stabilize_time_s: float = 1.2  # оценка одной стабилизации, с


@dataclass
class SwarmConfig:
    join_timeout: float = 30.0     # сколько ждать подтверждения входа дрона в хвост, с
    join_wait_s: float = 8.0       # заглушка: столько «занимает» вход в хвост, с


@dataclass
class MissionConfig:
    flight: FlightConfig = field(default_factory=FlightConfig)
    markers: MarkersConfig = field(default_factory=MarkersConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    formation_budget: float = FORMATION_BUDGET_S   # регламент: 3 минуты на сборку
    apples_config: str = ""        # путь к apples.yaml; пусто = штатный из apple_vision
    path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flight": asdict(self.flight),
            "markers": asdict(self.markers),
            "navigation": asdict(self.navigation),
            "search": asdict(self.search),
            "swarm": asdict(self.swarm),
            "formation_budget": self.formation_budget,
            "apples_config": self.apples_config,
        }


def _subset(cls, data: Dict[str, Any]):
    fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in (data or {}).items() if k in fields})


def load_config(path: str = DEFAULT_CONFIG_PATH) -> MissionConfig:
    """Читает YAML-конфиг миссии. Отсутствующие секции заполняются значениями по умолчанию."""
    import yaml

    path = os.path.expanduser(path or DEFAULT_CONFIG_PATH)
    if not os.path.isfile(path):
        return MissionConfig(path="")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return MissionConfig(
        flight=_subset(FlightConfig, raw.get("flight", {})),
        markers=_subset(MarkersConfig, raw.get("markers", {})),
        navigation=_subset(NavigationConfig, raw.get("navigation", {})),
        search=_subset(SearchConfig, raw.get("search", {})),
        swarm=_subset(SwarmConfig, raw.get("swarm", {})),
        formation_budget=float(raw.get("formation_budget", FORMATION_BUDGET_S)),
        apples_config=str(raw.get("apples_config", "") or ""),
        path=path,
    )
