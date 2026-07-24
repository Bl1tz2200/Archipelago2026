"""Связь с роем: поднять следующий дрон и дождаться, пока он встанет в хвост.

Регламент, п. 2.1.2: при распознавании «яблока» цепочка зависает, с земли взлетает
следующий дрон, и лидер не возобновляет движение, пока новый дрон не займёт узел
пройденного следа, а дистанция по UWB не подтвердит устойчивость строя.

Сам канал связи и подтверждение по UWB — за пределами этой задачи. Здесь только
интерфейс, в который команда подставит свою реализацию, и заглушка для отладки.
"""

from __future__ import annotations

import time
from typing import Optional, Protocol


class SwarmLink(Protocol):
    """Что миссия лидера требует от связи с роем."""

    def launch_next(self, index: int) -> None:
        """Команда «подними следующий дрон». `index` — номер яблока, 1..3."""

    def wait_tail_joined(self, index: int, timeout: float) -> bool:
        """Ждёт подтверждения, что новый дрон встал в хвост. False — не дождались."""


class ConsoleSwarmLink:
    """Заглушка: печатает команды в консоль и выдерживает паузу вместо подтверждения.

    Вывод в терминал фиксируют эксперты (регламент, п. 2.1.1), поэтому строки
    сформулированы так, чтобы по ним читался ход сборки формации.
    """

    def __init__(self, join_wait_s: float = 8.0) -> None:
        self.join_wait_s = float(join_wait_s)
        self.launched = 0
        self._launched_at: Optional[float] = None

    def launch_next(self, index: int) -> None:
        self.launched += 1
        self._launched_at = time.monotonic()
        print(f">>> КОМАНДА РОЮ: взлёт дрона #{index + 1} и вход в хвост формации", flush=True)

    def wait_tail_joined(self, index: int, timeout: float) -> bool:
        wait = min(self.join_wait_s, timeout)
        time.sleep(max(0.0, wait))
        print(f">>> ДРОН #{index + 1} В ХВОСТЕ (заглушка, {wait:.0f} с)", flush=True)
        return True
