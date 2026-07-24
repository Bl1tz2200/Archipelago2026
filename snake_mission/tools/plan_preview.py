#!/usr/bin/env python3
"""Маршрут поиска, покрытие и время — без дрона и без камеры.

    python3 tools/plan_preview.py                      # все четыре угла, полоса 3×3
    python3 tools/plan_preview.py --start 42 --span 3  # один угол подробно
    python3 tools/plan_preview.py --all-spans          # как меняется маршрут от высоты

Показывает, что маршрут законен (проверка регламента), что он покрывает все 49 меток
и что сборка укладывается в отведённые 3 минуты.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snake_mission import field, search  # noqa: E402
from snake_mission.config import load_config  # noqa: E402


def show(start_marker: int, span: int, config) -> bool:
    numbering = config.markers.numbering()
    start = numbering.node_of(start_marker)
    figure = field.figure_for(start_marker)

    print("─" * 68)
    print(f"Стартовый маркер {start_marker} → узел {start} → фигура: {figure or '—'}")
    plan = search.plan(start, (span, span), config.search)
    estimate = search.estimate(plan, config.search, apples=3, budget_s=config.formation_budget)

    print(plan.describe())
    print(estimate.describe())
    print(field.render(plan.route, plan.start))

    missing = set(field.all_nodes()) - plan.coverage()
    if missing:
        print(f"!!! НЕ ПОКРЫТО {len(missing)} узлов: {sorted(missing)[:10]}")
    try:
        search.validate(plan.route)
        print("Проверка регламента: все перелёты между соседними узлами, поле не покидаем.")
    except search.RouteError as exc:
        print(f"!!! МАРШРУТ НАРУШАЕТ РЕГЛАМЕНТ: {exc}")
        return False
    return not missing and estimate.fits


def main() -> int:
    p = argparse.ArgumentParser(description="Предпросмотр поискового маршрута")
    p.add_argument("--config", default="", help="путь к YAML-конфигу")
    p.add_argument("--start", type=int, default=-1, help="ID стартового маркера (0, 6, 42, 48)")
    p.add_argument("--span", type=int, default=0, help="сколько меток видно в кадре по стороне")
    p.add_argument("--all-spans", action="store_true", help="сравнить полосы обзора 1..5")
    args = p.parse_args()

    config = load_config(args.config)
    span = args.span or config.search.default_span
    starts = [args.start] if args.start >= 0 else sorted(field.FIGURES)

    ok = True
    if args.all_spans:
        print(f"{'полоса':>8} {'перелётов':>10} {'покрытие':>9} {'время, с':>9} {'бюджет':>8}")
        for test_span in range(1, 6):
            plan = search.plan(field.node_of(starts[0]), (test_span, test_span), config.search)
            est = search.estimate(plan, config.search, budget_s=config.formation_budget)
            print(f"{test_span:>8} {plan.steps:>10} {len(plan.coverage()):>9} "
                  f"{est.total_s:>9.0f} {'ok' if est.fits else 'НЕ ВЛЕЗАЕТ':>8}")
        return 0

    for start_marker in starts:
        ok = show(start_marker, span, config) and ok
    print("─" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
