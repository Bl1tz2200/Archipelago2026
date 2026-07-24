"""Полётное поле как сетка ArUco-меток 7×7.

Метража здесь нет и быть не должно: поле задано метками, узел — это пара индексов
`(col, row)`, а перелёт — «на соседнюю метку». Регламент, п. 2.1.3 и модель движения:
«перелёты только между соседними узлами координатной сетки, включая диагональные,
в пределах полётной зоны».

Соответствие ID ↔ узел выведено из Приложения 3 регламента: ID углов 0, 42, 48, 6 —
это ровно четыре угла сетки 7×7 при нумерации `ID = row * 7 + col`:

        row 6 |  42 43 44 45 46 47 48
        row 5 |  35 ...           41
        ...   |
        row 0 |   0  1  2  3  4  5  6
                 col 0 ............ 6

Физическая раскладка на площадке может оказаться зеркальной или транспонированной —
для этого есть `Numbering`. Проверяется одним кадром: `tools/marker_check.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

SIDE = 7                      # меток по стороне поля
MARKERS = SIDE * SIDE         # 49 меток в квадрате

Node = Tuple[int, int]        # (col, row), обе координаты 0..6

# Приложение 3 регламента: угол, из которого стартует команда, задаёт фигуру.
FIGURES: Dict[int, str] = {
    0: "Квадрат",
    42: "Прямоугольник",
    48: "Трапеция",
    6: "Ромб",
}

# Те же четыре ID как углы сетки — при нумерации по умолчанию.
CORNER_NODES: Dict[int, Node] = {
    0: (0, 0),
    6: (SIDE - 1, 0),
    42: (0, SIDE - 1),
    48: (SIDE - 1, SIDE - 1),
}


@dataclass(frozen=True)
class Numbering:
    """Как физическая раскладка меток ложится на индексы узлов.

    По умолчанию — `ID = row * 7 + col`, нумерация от узла (0, 0). Если на площадке
    метки разложены иначе, правится не код, а `field.numbering` в конфиге.
    """

    row_major: bool = True     # False — ID растёт по столбцам: ID = col * 7 + row
    flip_x: bool = False       # True — столбцы пронумерованы справа налево
    flip_y: bool = False       # True — строки пронумерованы сверху вниз

    def marker_id(self, node: Node) -> int:
        col, row = node
        if not inside(node):
            raise ValueError(f"узел {node} вне поля {SIDE}×{SIDE}")
        if self.flip_x:
            col = SIDE - 1 - col
        if self.flip_y:
            row = SIDE - 1 - row
        return row * SIDE + col if self.row_major else col * SIDE + row

    def node_of(self, marker_id: int) -> Node:
        if not 0 <= marker_id < MARKERS:
            raise ValueError(f"ID {marker_id} вне поля: ожидается 0..{MARKERS - 1}")
        if self.row_major:
            row, col = divmod(marker_id, SIDE)
        else:
            col, row = divmod(marker_id, SIDE)
        if self.flip_x:
            col = SIDE - 1 - col
        if self.flip_y:
            row = SIDE - 1 - row
        return col, row


DEFAULT_NUMBERING = Numbering()


def marker_id(node: Node, numbering: Numbering = DEFAULT_NUMBERING) -> int:
    """Узел → ID метки, которая в нём лежит."""
    return numbering.marker_id(node)


def node_of(mid: int, numbering: Numbering = DEFAULT_NUMBERING) -> Node:
    """ID метки → узел сетки."""
    return numbering.node_of(mid)


def inside(node: Node) -> bool:
    """Узел внутри полётной зоны. Граница поля — это и есть граница сетки меток."""
    col, row = node
    return 0 <= col < SIDE and 0 <= row < SIDE


def is_corner(node: Node) -> bool:
    col, row = node
    return col in (0, SIDE - 1) and row in (0, SIDE - 1)


def corner_ids(numbering: Numbering = DEFAULT_NUMBERING) -> List[int]:
    """ID четырёх угловых меток при заданной нумерации."""
    return [numbering.marker_id(n) for n in
            ((0, 0), (SIDE - 1, 0), (0, SIDE - 1), (SIDE - 1, SIDE - 1))]


def figure_for(mid: int) -> Optional[str]:
    """Назначенная фигура по ID стартового маркера (Приложение 3). None — не угол."""
    return FIGURES.get(mid)


def neighbors(node: Node) -> List[Node]:
    """Восемь соседних узлов, включая диагональные; за границу не выходим."""
    col, row = node
    result = []
    for dcol in (-1, 0, 1):
        for drow in (-1, 0, 1):
            if dcol == 0 and drow == 0:
                continue
            candidate = (col + dcol, row + drow)
            if inside(candidate):
                result.append(candidate)
    return result


def is_legal_step(a: Node, b: Node) -> bool:
    """Разрешён ли перелёт a → b по модели движения «змейки».

    Разрешены только соседние узлы, включая диагональ (повороты кратны 45°),
    и обе точки должны лежать в пределах поля.
    """
    if not inside(a) or not inside(b):
        return False
    dcol, drow = b[0] - a[0], b[1] - a[1]
    if dcol == 0 and drow == 0:
        return False
    return abs(dcol) <= 1 and abs(drow) <= 1


def chebyshev(a: Node, b: Node) -> int:
    """Сколько перелётов между узлами минимум (диагональ считается за один)."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def straight_path(a: Node, b: Node) -> List[Node]:
    """Кратчайший путь узел→узел по правилам «змейки», без начальной точки.

    Сначала идут диагональные шаги, потом прямые — так получается минимальное
    число перелётов, и каждый шаг легален.
    """
    path: List[Node] = []
    col, row = a
    while (col, row) != b:
        col += (b[0] > col) - (b[0] < col)
        row += (b[1] > row) - (b[1] < row)
        path.append((col, row))
    return path


def all_nodes() -> Iterator[Node]:
    """Все 49 узлов поля, по строкам снизу вверх."""
    for row in range(SIDE):
        for col in range(SIDE):
            yield col, row


def render(route: List[Node] = (), start: Optional[Node] = None) -> str:
    """Поле 7×7 в ASCII с порядковыми номерами посещения — для проверки маршрута глазами."""
    order: Dict[Node, int] = {}
    for i, node in enumerate(route):
        order.setdefault(node, i)

    lines = []
    for row in range(SIDE - 1, -1, -1):
        cells = []
        for col in range(SIDE):
            node = (col, row)
            if node == start:
                cells.append("  S")
            elif node in order:
                cells.append(f"{order[node]:3d}")
            else:
                cells.append("  .")
        lines.append(f"row {row} |" + "".join(cells))
    lines.append("        " + "".join(f"{c:3d}" for c in range(SIDE)))
    lines.append("         " + "  col")
    return "\n".join(lines)
