#!/usr/bin/env bash
# Главная точка входа Archipelago2026 — одна команда на все системы.
#
# На дроне (внутри контейнера sverk_ros2) сам подхватывает окружение ROS 2 и
# рабочее пространство ~/sverk_ws, поэтому руками source-ить ничего не нужно.
#
#   ./run.sh check            наземная проверка ВСЕХ систем подряд (пропеллеры сняты):
#                             поле меток → распознавание «яблок» → расчёт маршрута
#   ./run.sh mission [ключи]  ЗАЧЁТНЫЙ ПОЛЁТ: навигация по меткам + поиск «яблок»
#                             в одном процессе (это и есть «все системы» в воздухе)
#
#   ./run.sh markers [ключи]  только проверка поля меток   (snake_mission/tools/marker_check.py)
#   ./run.sh apples  [ключи]  только распознавание «яблок» (apple_vision/run_apple_detect.py)
#   ./run.sh plan    [ключи]  расчёт маршрута без дрона     (snake_mission/tools/plan_preview.py)
#   ./run.sh sim     [ключи]  прогон миссии в симуляторе    (snake_mission/tools/simulate_mission.py)
#
# Ключи полёта пробрасываются как есть, например:
#   ./run.sh mission --alt 2.0 --speed 0.6 --no-land
#
# Длительности наземной проверки можно переопределить переменными окружения:
#   MARKER_SECONDS=30 APPLE_SECONDS=60 START=42 ./run.sh check

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

# --- окружение ROS 2 и рабочее пространство (только на дроне) ----------------
# Симулятор и расчёт маршрута ROS не требуют, поэтому отсутствие setup.bash — не ошибка.
source_ros() {
    local s
    for s in /opt/ros/*/setup.bash; do
        [ -f "$s" ] && source "$s" && break
    done
    if [ -f "$HOME/sverk_ws/install/setup.bash" ]; then
        source "$HOME/sverk_ws/install/setup.bash"
    fi
}

usage() {
    # печатает шапку-комментарий со 2-й строки до первой не-«#» строки
    awk 'NR>=2 && /^#/ { sub(/^# ?/, ""); print; next } NR>=2 { exit }' "$HERE/run.sh"
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift

case "$cmd" in
    check|preflight)
        source_ros
        ms="${MARKER_SECONDS:-20}"
        as="${APPLE_SECONDS:-30}"
        # Жёсткий таймер на шаг с метками — на случай, если камера не отдаёт кадры.
        # Шаг с яблоками идёт с --no-telemetry: наземная проверка выполняется
        # с пропеллерами, без взлёта, поэтому телеметрия FCU для неё не нужна, а
        # get_telemetry() без неё виснет вечно и не реагирует на обычный сигнал.
        echo "########## 1/3 · ПОЛЕ МЕТОК ##########"
        timeout -s KILL "$((ms + 15))" python3 snake_mission/tools/marker_check.py --duration "$ms"
        [ $? -eq 137 ] && echo "(проверка меток зависла и снята по таймеру — проверьте камеру: ros2 topic hz /camera_1/image_raw)"
        echo "########## 2/3 · РАСПОЗНАВАНИЕ «ЯБЛОК» ##########"
        timeout -s KILL "$((as + 15))" python3 apple_vision/run_apple_detect.py --duration "$as" --no-telemetry
        [ $? -eq 137 ] && echo "(распознавание зависло и снято по таймеру — проверьте камеру: ros2 topic hz /camera_1/image_raw)"
        echo "########## 3/3 · РАСЧЁТ МАРШРУТА (без полёта) ##########"
        python3 snake_mission/run_leader.py --dry-run --start "${START:-42}"
        echo "########## наземная проверка завершена ##########"
        ;;
    mission|fly|run)
        source_ros
        exec python3 snake_mission/run_leader.py "$@"
        ;;
    markers)
        source_ros
        exec python3 snake_mission/tools/marker_check.py "$@"
        ;;
    apples)
        source_ros
        exec python3 apple_vision/run_apple_detect.py "$@"
        ;;
    plan)
        exec python3 snake_mission/tools/plan_preview.py "$@"
        ;;
    sim)
        exec python3 snake_mission/tools/simulate_mission.py "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "run.sh: неизвестная команда «$cmd»"
        echo
        usage
        exit 2
        ;;
esac
