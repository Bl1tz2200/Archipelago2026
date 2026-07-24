"""Поисковый маршрут по полю меток и проверка его на соответствие регламенту.

Задача: обойти все 49 узлов «полосой обзора» камеры за минимальное число перелётов,
не нарушив модель движения «змейки» (регламент, п. 2.1 и п. 2.1.3):

  * перелёты только между соседними узлами, включая диагональные;
  * повороты кратны 45°;
  * в пределах полётной зоны;
  * на постоянной высоте.

Ширина полосы обзора приходит из `markers.visible_span` — она измеряется **в метках**,
поэтому весь планировщик работает в индексах узлов и не знает ни одного метра.

Идея маршрута — «бустрофедон»: параллельные проходы через всё поле, между проходами
диагональный переход. Проходы разрежены настолько, насколько позволяет полоса обзора:
если камера охватывает соседний ряд меток с каждой стороны, вместо семи проходов
достаточно трёх.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from . import field
from .config import SearchConfig
from .field import SIDE, Node

Route = List[Node]


class RouteError(ValueError):
    """Маршрут нарушает модель движения «змейки» — взлетать с ним нельзя."""


# ------------------------------------------------------------------- покрытие


def reach_from_span(span: int, safety_markers: int, max_pass_spacing: int) -> int:
    """Сколько рядов меток по каждую сторону от прохода считаем просмотренными.

    `span` — сколько рядов меток видно в кадре целиком. Радиус полосы обзора — это
    половина от span; `safety_markers` вычитается из радиуса (1 = «крайнему ряду в
    кадре не доверяем»), `max_pass_spacing` ограничивает разрежение сверху.
    """
    reach = max(0, (max(1, int(span)) - 1) // 2 - max(0, int(safety_markers)))
    return min(reach, max(0, (max(1, int(max_pass_spacing)) - 1) // 2))


def pass_lines(reach: int, from_end: bool = False) -> List[int]:
    """Индексы линий проходов, покрывающие все ряды 0..6 при радиусе обзора `reach`.

    Жадная раскладка: каждый следующий проход ставится так, чтобы его полоса
    начиналась ровно там, где кончилась предыдущая. Полное покрытие — по построению.
    """
    lines: List[int] = []
    next_uncovered = 0
    while next_uncovered < SIDE:
        line = min(next_uncovered + reach, SIDE - 1)
        lines.append(line)
        next_uncovered = line + reach + 1
    if from_end:
        lines = [SIDE - 1 - line for line in reversed(lines)]
    return lines


def covered(route: Sequence[Node], reach_cols: int, reach_rows: int) -> Set[Node]:
    """Узлы, попавшие в полосу обзора хотя бы раз за маршрут."""
    seen: Set[Node] = set()
    for col, row in route:
        for dcol in range(-reach_cols, reach_cols + 1):
            for drow in range(-reach_rows, reach_rows + 1):
                node = (col + dcol, row + drow)
                if field.inside(node):
                    seen.add(node)
    return seen


# ------------------------------------------------------------------ построение


def _sweep(line: int, along_rows: bool, start_at_low: bool) -> Route:
    """Один проход через всё поле: узлы от края до края по линии `line`."""
    order = range(SIDE) if start_at_low else range(SIDE - 1, -1, -1)
    return [(pos, line) if along_rows else (line, pos) for pos in order]


def _append(route: Route, nodes: Sequence[Node]) -> None:
    """Добавляет узлы, докладывая промежуточные шаги, если узлы не соседние."""
    for node in nodes:
        if route and route[-1] == node:
            continue
        if route and not field.is_legal_step(route[-1], node):
            route.extend(field.straight_path(route[-1], node))
        else:
            route.append(node)


def _build(start: Node, reach_along: int, along_rows: bool) -> Route:
    """Бустрофедон от стартового угла: проходы по строкам либо по столбцам.

    `reach_along` — радиус обзора поперёк проходов (в рядах меток).
    """
    start_line = start[1] if along_rows else start[0]
    start_pos = start[0] if along_rows else start[1]

    lines = pass_lines(reach_along, from_end=start_line > (SIDE - 1) / 2)
    # Первый проход — ближайший к стартовому углу, дальше по порядку раскладки.
    lines.sort(key=lambda line: abs(line - start_line))
    first = lines[0]
    rest = sorted((line for line in lines if line != first),
                  key=lambda line: abs(line - first))
    lines = [first] + rest

    route: Route = [start]
    start_at_low = start_pos == 0
    for i, line in enumerate(lines):
        sweep = _sweep(line, along_rows, start_at_low if i % 2 == 0 else not start_at_low)
        _append(route, sweep)
    return route


@dataclass
class SearchPlan:
    """Готовый маршрут поиска вместе с тем, как он был получен."""

    route: Route
    start: Node
    reach_cols: int
    reach_rows: int
    along_rows: bool
    span: Tuple[int, int]

    @property
    def steps(self) -> int:
        return max(0, len(self.route) - 1)

    @property
    def end(self) -> Node:
        return self.route[-1]

    @property
    def return_steps(self) -> int:
        """Сколько перелётов от конца маршрута до стартового узла."""
        return field.chebyshev(self.end, self.start)

    def coverage(self) -> Set[Node]:
        return covered(self.route, self.reach_cols, self.reach_rows)

    def describe(self) -> str:
        direction = "по строкам" if self.along_rows else "по столбцам"
        full = len(self.coverage()) == field.MARKERS
        return (
            f"Маршрут поиска: {self.steps} перелётов, проходы {direction}, "
            f"полоса обзора {self.span[0]}×{self.span[1]} меток "
            f"(радиус {self.reach_cols}×{self.reach_rows}), "
            f"покрытие {len(self.coverage())}/{field.MARKERS}"
            f"{'' if full else ' — НЕПОЛНОЕ'}, "
            f"возврат на старт {self.return_steps} перелётов"
        )


def plan(
    start: Node,
    span: Tuple[int, int] = (3, 3),
    config: Optional[SearchConfig] = None,
) -> SearchPlan:
    """Строит маршрут поиска от стартового угла при измеренной полосе обзора.

    `span` — (столбцов, строк) меток в кадре. Из двух ориентаций проходов берётся
    та, что даёт меньше перелётов; при равенстве — та, после которой ближе
    возвращаться на старт.
    """
    config = config or SearchConfig()
    if not field.inside(start):
        raise RouteError(f"стартовый узел {start} вне поля")

    span_cols = max(1, int(span[0]))
    span_rows = max(1, int(span[1]))
    reach_cols = reach_from_span(span_cols, config.safety_markers, config.max_pass_spacing)
    reach_rows = reach_from_span(span_rows, config.safety_markers, config.max_pass_spacing)

    candidates = []
    for along_rows in (True, False):
        # Проходы вдоль строк разрежаются радиусом по строкам, вдоль столбцов — по столбцам.
        reach_along = reach_rows if along_rows else reach_cols
        route = _build(start, reach_along, along_rows)
        candidate = SearchPlan(route=route, start=start, reach_cols=reach_cols,
                               reach_rows=reach_rows, along_rows=along_rows,
                               span=(span_cols, span_rows))
        if len(candidate.coverage()) < field.MARKERS:
            continue
        candidates.append(candidate)

    if not candidates:
        # Полоса обзора не покрывает поле ни при какой раскладке — идём по каждому ряду.
        route = _build(start, 0, True)
        candidates = [SearchPlan(route=route, start=start, reach_cols=0, reach_rows=0,
                                 along_rows=True, span=(span_cols, span_rows))]

    best = min(candidates, key=lambda p: (p.steps, p.return_steps))
    validate(best.route)
    return best


# ------------------------------------------------------------------- проверки


def validate(route: Sequence[Node]) -> None:
    """Проверяет маршрут на соответствие модели движения «змейки».

    Вызывается до взлёта: невалидный маршрут — это отказ от старта, а не нарушение
    в воздухе. Регламент, п. 2.1.3: «перелёты только между соседними узлами
    координатной сетки, включая диагональные, в пределах полётной зоны».
    """
    if not route:
        raise RouteError("маршрут пуст")
    for i, node in enumerate(route):
        if not field.inside(node):
            raise RouteError(f"узел {node} (шаг {i}) вне поля {SIDE}×{SIDE}")
    for i in range(1, len(route)):
        if not field.is_legal_step(route[i - 1], route[i]):
            raise RouteError(
                f"шаг {i}: перелёт {route[i - 1]} → {route[i]} не между соседними узлами "
                "(разрешены только соседние, включая диагональные)"
            )


def resume(route: Sequence[Node], index: int) -> Route:
    """Остаток маршрута начиная с узла `index` — продолжение обхода после зависания."""
    index = max(0, min(int(index), len(route)))
    return list(route[index:])


# --------------------------------------------------------------- оценка времени


@dataclass
class TimeEstimate:
    steps: int
    flight_s: float
    stabilize_s: float
    apples_s: float
    return_s: float
    total_s: float
    budget_s: float

    @property
    def fits(self) -> bool:
        return self.total_s <= self.budget_s

    def describe(self) -> str:
        verdict = "укладываемся" if self.fits else "НЕ УКЛАДЫВАЕМСЯ"
        return (
            f"Оценка времени сборки: {self.total_s:.0f} с из {self.budget_s:.0f} с — {verdict}\n"
            f"  перелёты {self.steps} × ~{self.flight_s / max(1, self.steps):.1f} с = {self.flight_s:.0f} с\n"
            f"  стабилизация на метках = {self.stabilize_s:.0f} с\n"
            f"  зависания на яблоках = {self.apples_s:.0f} с\n"
            f"  возврат на старт = {self.return_s:.0f} с"
        )


def estimate(
    plan_or_route: SearchPlan | Sequence[Node],
    config: Optional[SearchConfig] = None,
    apples: int = 3,
    budget_s: float = 180.0,
) -> TimeEstimate:
    """Оценка времени сборки формации: перелёты + стабилизации + зависания + возврат.

    Считается до взлёта, чтобы заранее видеть, влезает ли маршрут в 3 минуты
    регламента, и при необходимости поднять высоту или разредить проходы.
    """
    config = config or SearchConfig()
    if isinstance(plan_or_route, SearchPlan):
        route, return_steps = plan_or_route.route, plan_or_route.return_steps
    else:
        route = list(plan_or_route)
        return_steps = field.chebyshev(route[-1], route[0]) if route else 0

    steps = max(0, len(route) - 1)
    flight_s = steps * config.step_time_s
    stabilize_s = len(route) * config.stabilize_time_s
    apples_s = apples * config.apple_hold_s
    return_s = return_steps * (config.step_time_s + config.stabilize_time_s)
    return TimeEstimate(
        steps=steps,
        flight_s=flight_s,
        stabilize_s=stabilize_s,
        apples_s=apples_s,
        return_s=return_s,
        total_s=flight_s + stabilize_s + apples_s + return_s,
        budget_s=budget_s,
    )
