#!/usr/bin/env python3
"""Главная точка входа Archipelago2026 — одна команда на все системы.

Запускается только Python-ом, и тем самым, каким вы захотите:

    python3 run.py flight --alt 0.5 --speed 0.3
    ~/venv/bin/python3 run.py sim
    ./run.py check

Каким интерпретатором запущен сам `run.py`, тем же пойдут и все дочерние
скрипты (`sys.executable`) — отдельного «системного python3» тут нет.

Команды
───────
  check             наземная проверка ВСЕХ систем подряд (пропеллеры сняты):
                    поле меток → распознавание «яблок» → расчёт маршрута
  mission [ключи]   ЗАЧЁТНЫЙ ПОЛЁТ: навигация по меткам + поиск «яблок»
                    в одном процессе (это и есть «все системы» в воздухе)
  flight  [ключи]   ПРОСТОЙ ПОЛЁТ: арм → взлёт → змейка по полю → посадка,
                    без «яблок» и роя. Ни один вызов не виснет молча —
                    у каждого свой таймаут, поэтому дрон всегда сядет.

  markers [ключи]   только проверка поля меток   (snake_mission/tools/marker_check.py)
  apples  [ключи]   только распознавание «яблок» (apple_vision/run_apple_detect.py)
  plan    [ключи]   расчёт маршрута без дрона     (snake_mission/tools/plan_preview.py)
  sim     [ключи]   прогон миссии в симуляторе    (snake_mission/tools/simulate_mission.py)
  help              эта справка

Ключи пробрасываются в дочерний скрипт как есть:
  python3 run.py mission --alt 2.0 --speed 0.6 --no-land
  python3 run.py flight --config snake_mission/config/test_0_5m.yaml

Длительности наземной проверки — переменными окружения:
  MARKER_SECONDS=30 APPLE_SECONDS=60 START=42 python3 run.py check

Окружение ROS 2
───────────────
На дроне ROS 2 и ~/sverk_ws подхватываются сами — руками source-ить не нужно.
Единственное место, где всё ещё участвует оболочка: сами файлы `setup.bash`
написаны на shell, и переменные (LD_LIBRARY_PATH, AMENT_PREFIX_PATH, PYTHONPATH)
обязаны стоять ДО загрузки нативных библиотек ROS. Поэтому `run.py` один раз
перезапускает сам себя через `bash -c 'source …; exec python3 run.py …'`.
Отключается ключом `--no-ros` (или переменной SNAKE_NO_ROS=1), если окружение
вы уже подняли сами.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from glob import glob
from typing import List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))

# Метка «ROS уже подняли, второй раз не перезапускаться» — иначе получится петля.
REEXEC_GUARD = "SNAKE_ROS_READY"

# Команда → скрипт, который она запускает.
TOOLS = {
    "mission": "snake_mission/run_leader.py",
    "flight": "snake_mission/run_flight.py",
    "markers": "snake_mission/tools/marker_check.py",
    "apples": "apple_vision/run_apple_detect.py",
    "plan": "snake_mission/tools/plan_preview.py",
    "sim": "snake_mission/tools/simulate_mission.py",
}
ALIASES = {"fly": "mission", "run": "mission", "preflight": "check"}

# Расчёт маршрута и симулятор ROS не требуют — для них не поднимаем окружение.
NEEDS_ROS = {"check", "mission", "flight", "markers", "apples"}

# Сколько ждать сверх заказанной длительности, прежде чем убить шаг проверки.
CHECK_GRACE_S = 15


# ------------------------------------------------------------- окружение ROS


def find_ros_setups() -> List[str]:
    """Файлы окружения, которые нужно вычитать: сам ROS 2 и рабочее пространство."""
    setups: List[str] = []
    for candidate in sorted(glob("/opt/ros/*/setup.bash")):
        if os.path.isfile(candidate):
            setups.append(candidate)
            break                      # дистрибутив ROS нужен ровно один
    workspace = os.path.expanduser("~/sverk_ws/install/setup.bash")
    if os.path.isfile(workspace):
        setups.append(workspace)
    return setups


def reexec_with_ros(argv: Sequence[str]) -> None:
    """Перезапускает себя с поднятым окружением ROS 2. Возвращается — значит не понадобилось.

    Вычитать `setup.bash` внутрь уже запущенного процесса нельзя: переменные вроде
    LD_LIBRARY_PATH читает динамический загрузчик при старте, и правка `os.environ`
    задним числом на него не влияет. Поэтому — единственный перезапуск.
    """
    if os.environ.get(REEXEC_GUARD) or os.environ.get("SNAKE_NO_ROS"):
        return
    setups = find_ros_setups()
    if not setups:
        return                         # не дрон: симулятор и расчёты ROS не требуют

    shell = "/bin/bash"
    if not os.path.isfile(shell):
        print(f"[run] {shell} не найден — окружение ROS 2 поднимите сами "
              "(source /opt/ros/*/setup.bash)", file=sys.stderr, flush=True)
        return

    script = " && ".join(f". {shlex.quote(path)}" for path in setups) + ' && exec "$@"'
    child = [sys.executable, os.path.join(HERE, "run.py"), *argv]
    os.environ[REEXEC_GUARD] = "1"
    try:
        os.execv(shell, [shell, "-c", script, "run.py", *child])
    except OSError as exc:             # не смогли — не повод не работать вовсе
        print(f"[run] окружение ROS 2 поднять не удалось ({exc}), продолжаем как есть",
              file=sys.stderr, flush=True)
        os.environ.pop(REEXEC_GUARD, None)


# ----------------------------------------------------------------- запуск


def tool_path(name: str) -> str:
    return os.path.join(HERE, TOOLS[name])


def exec_tool(name: str, args: Sequence[str]) -> int:
    """Отдаёт процесс дочернему скрипту целиком — как `exec` в оболочке.

    Замена процесса, а не подпроцесс: Ctrl+C и сигналы приходят прямо в миссию,
    между терминалом и дроном не остаётся лишнего звена.
    """
    command = [sys.executable, tool_path(name), *args]
    try:
        os.execv(sys.executable, command)
    except OSError as exc:
        print(f"[run] не удалось запустить {TOOLS[name]}: {exc}", file=sys.stderr)
        return 1
    return 0                            # сюда не доходим


def run_step(title: str, args: Sequence[str], timeout: Optional[int] = None,
             hint: str = "") -> int:
    """Один шаг наземной проверки: под жёстким таймером, с понятным словом на выходе.

    Без телеметрии FCU некоторые вызовы не возвращаются и не реагируют на SIGTERM,
    поэтому по истечении времени шаг убивается насмерть (`kill`), а не «просится выйти».
    """
    print(f"########## {title} ##########", flush=True)
    process = subprocess.Popen([sys.executable, *args], cwd=HERE)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        print(f"(шаг завис и снят по таймеру{' — ' + hint if hint else ''})", flush=True)
        return 137
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise


def cmd_check() -> int:
    """Наземная проверка всех систем подряд. Пропеллеры сняты, дрон не взлетает."""
    marker_s = int(os.environ.get("MARKER_SECONDS", "20"))
    apple_s = int(os.environ.get("APPLE_SECONDS", "30"))
    start = os.environ.get("START", "42")
    camera_hint = "проверьте камеру: ros2 topic hz /camera_1/image_raw"

    run_step("1/3 · ПОЛЕ МЕТОК",
             ["snake_mission/tools/marker_check.py", "--duration", str(marker_s)],
             timeout=marker_s + CHECK_GRACE_S, hint=camera_hint)

    # --no-telemetry: проверка идёт на земле, без взлёта, телеметрия FCU для неё
    # не нужна, а get_telemetry() без неё виснет вечно.
    run_step("2/3 · РАСПОЗНАВАНИЕ «ЯБЛОК»",
             ["apple_vision/run_apple_detect.py", "--duration", str(apple_s),
              "--no-telemetry"],
             timeout=apple_s + CHECK_GRACE_S, hint=camera_hint)

    run_step("3/3 · РАСЧЁТ МАРШРУТА (без полёта)",
             ["snake_mission/run_leader.py", "--dry-run", "--start", start])

    print("########## наземная проверка завершена ##########", flush=True)
    return 0


def usage() -> str:
    return (__doc__ or "").strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # --no-ros снимаем до разбора команды: он адресован лаунчеру, не дочернему скрипту.
    if "--no-ros" in argv:
        argv.remove("--no-ros")
        os.environ["SNAKE_NO_ROS"] = "1"

    command = argv[0] if argv else "help"
    args = argv[1:]
    command = ALIASES.get(command, command)

    if command in ("help", "-h", "--help"):
        print(usage())
        return 0

    if command not in TOOLS and command != "check":
        print(f"run.py: неизвестная команда «{command}»\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        return 2

    if command in NEEDS_ROS:
        reexec_with_ros([command, *args])   # обычно не возвращается

    if command == "check":
        try:
            return cmd_check()
        except KeyboardInterrupt:
            print("\n[run] наземная проверка остановлена оператором", flush=True)
            return 130

    return exec_tool(command, args)


if __name__ == "__main__":
    raise SystemExit(main())
