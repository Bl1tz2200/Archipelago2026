"""Миссия лидера для соревнования «Змейка»: полёт по ArUco-меткам и поиск «яблок».

    from snake_mission import LeaderMission
    mission = LeaderMission(drone)
    mission.run()

Распознавание «яблок» берётся из соседнего пакета `apple_vision`, поэтому при импорте
в `sys.path` добавляется его каталог — чтобы модуль запускался и с борта, и из корня
репозитория без установки пакетов.
"""

from __future__ import annotations

import os
import sys

_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APPLE_VISION = os.path.join(os.path.dirname(_PACKAGE_ROOT), "apple_vision")
if os.path.isdir(_APPLE_VISION) and _APPLE_VISION not in sys.path:
    sys.path.insert(0, _APPLE_VISION)

from .config import MissionConfig, load_config  # noqa: E402
from .field import FIGURES, SIDE, Numbering, figure_for, is_legal_step, node_of, marker_id  # noqa: E402
from .markers import MarkerDetector, Sight, visible_span  # noqa: E402
from .navigator import MarkerNavigator, NavResult  # noqa: E402
from .search import RouteError, SearchPlan, estimate, plan, validate  # noqa: E402
from .swarm import ConsoleSwarmLink, SwarmLink  # noqa: E402

__all__ = [
    "MissionConfig",
    "load_config",
    "FIGURES",
    "SIDE",
    "Numbering",
    "figure_for",
    "is_legal_step",
    "node_of",
    "marker_id",
    "MarkerDetector",
    "Sight",
    "visible_span",
    "MarkerNavigator",
    "NavResult",
    "SearchPlan",
    "RouteError",
    "plan",
    "estimate",
    "validate",
    "SwarmLink",
    "ConsoleSwarmLink",
    "LeaderMission",
    "MissionResult",
    "dry_run",
]

__version__ = "1.0.0"


def __getattr__(name: str):
    # LeaderMission тянет apple_vision (numpy, OpenCV) — импортируем лениво,
    # чтобы field/search/markers работали и там, где apple_vision недоступен.
    if name in ("LeaderMission", "MissionResult", "dry_run"):
        from . import mission
        return getattr(mission, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
