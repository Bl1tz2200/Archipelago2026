#!/usr/bin/env python3
"""Точка входа: миссия лидера «Змейка».

    python3 main.py                      # зачётный запуск на дроне
    python3 main.py --dry-run --start 42 # расчёт маршрута без дрона
    python3 main.py --alt 2.0 --no-land  # своя высота, без посадки в конце

Все ключи запуска и подробности — в snake_mission/README.md.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "snake_mission"))

from run_leader import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
