"""Однократный зачёт «яблок».

Регламент, п. 2.1.2: «Каждое "яблоко" засчитывается однократно: повторное
обнаружение уже засчитанного "яблока" не приводит к взлёту следующего дрона».

Поэтому одного попадания цвета в кадр мало — нужно:
  * подтвердить цель несколькими кадрами подряд (блик или мусор так отсеивается);
  * запомнить засчитанные яблоки и больше на них не реагировать;
  * не считать одно яблоко за два, когда цепочка проходит над ним второй раз.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import RegistryConfig
from .detector import Detection


@dataclass
class AppleEvent:
    """Факт зачёта яблока — по нему поднимается следующий дрон."""

    index: int                                   # 1..3, порядок обнаружения
    color: str
    label: str
    pixel: Tuple[float, float]
    world: Optional[Tuple[float, float]]         # координаты пола в map, если известны
    altitude: Optional[float]
    timestamp: float
    frames_confirmed: int

    def describe(self) -> str:
        where = f", позиция map x={self.world[0]:.2f} y={self.world[1]:.2f}" if self.world else ""
        return f"ЯБЛОКО #{self.index}: {self.label} (цвет {self.color}){where}"


@dataclass
class _Track:
    color: str
    center: Tuple[float, float]
    world: Optional[Tuple[float, float]] = None
    hits: int = 0
    misses: int = 0
    first_seen: float = field(default_factory=time.time)


class AppleRegistry:
    """Накапливает подтверждения и выдаёт событие ровно один раз на яблоко."""

    def __init__(self, config: Optional[RegistryConfig] = None) -> None:
        self.config = config or RegistryConfig()
        self._tracks: Dict[str, _Track] = {}
        self.events: List[AppleEvent] = []

    # ------------------------------------------------------------- статус

    @property
    def count(self) -> int:
        return len(self.events)

    @property
    def complete(self) -> bool:
        return self.count >= self.config.max_apples

    @property
    def claimed_colors(self) -> List[str]:
        return [e.color for e in self.events]

    def pending(self, color: str) -> Tuple[int, int]:
        """Сколько подтверждений уже набрано по цвету и сколько нужно."""
        track = self._tracks.get(color)
        return (track.hits if track else 0), self.config.confirm_frames

    def reset(self) -> None:
        self._tracks.clear()
        self.events.clear()

    # -------------------------------------------------------------- логика

    def _already_claimed(self, color: str, world: Optional[Tuple[float, float]]) -> bool:
        for event in self.events:
            if self.config.unique_by_color and event.color == color:
                return True
            if world and event.world:
                if math.dist(world, event.world) <= self.config.merge_radius_m:
                    return True
        return False

    def update(
        self,
        detections: Sequence[Detection],
        worlds: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
        altitude: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> List[AppleEvent]:
        """Скармливает результат одного кадра. Возвращает новые события (обычно пусто)."""
        now = timestamp if timestamp is not None else time.time()
        worlds = list(worlds or [None] * len(detections))
        if len(worlds) < len(detections):
            worlds += [None] * (len(detections) - len(worlds))

        new_events: List[AppleEvent] = []
        seen_colors = set()

        for det, world in zip(detections, worlds):
            if self._already_claimed(det.color, world):
                continue
            seen_colors.add(det.color)

            track = self._tracks.get(det.color)
            if track is None or det.distance_to(track.center) > self.config.track_radius_px:
                # Новый объект или прыжок центра — начинаем подтверждение заново.
                track = _Track(color=det.color, center=det.center, world=world, hits=1)
                self._tracks[det.color] = track
            else:
                track.center = det.center
                track.world = world or track.world
                track.hits += 1
                track.misses = 0

            if track.hits >= self.config.confirm_frames and not self.complete:
                event = AppleEvent(
                    index=len(self.events) + 1,
                    color=det.color,
                    label=det.label,
                    pixel=det.center,
                    world=track.world,
                    altitude=altitude,
                    timestamp=now,
                    frames_confirmed=track.hits,
                )
                self.events.append(event)
                new_events.append(event)
                self._tracks.pop(det.color, None)

        # Цели, которых в этом кадре не было, постепенно теряют накопленные подтверждения.
        for color, track in list(self._tracks.items()):
            if color in seen_colors:
                continue
            track.misses += 1
            if track.misses > self.config.lost_frames:
                self._tracks.pop(color, None)

        return new_events
