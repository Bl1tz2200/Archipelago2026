ЗАПУСК ИЗ БРАУЗЕРНОГО VSCODE ДРОНА

1. Откройте браузерный VSCode дрона по адресу 192.168.1.126.
2. Загрузите и распакуйте репозиторий Archipelago2026 без изменения структуры каталогов.
3. Откройте встроенный терминал VSCode и перейдите в каталог проекта:

   cd ~/Archipelago2026


ПРОВЕРКИ БЕЗ ПОЛЁТА (пропеллеры сняты)

4. Что видит камера над полем меток — ID, раскладка, полоса обзора:

   python3 snake_mission/tools/marker_check.py --duration 30

5. Распознаются ли «яблоки» под камерой:

   python3 apple_vision/run_apple_detect.py --duration 60

   Размеченный поток публикуется в /out_detection.

6. Расчёт маршрута без полёта — покрытие поля и оценка времени:

   python3 snake_mission/run_leader.py --dry-run --start 42


ПОЛЁТ

7. Зачётный запуск:

   python3 snake_mission/run_leader.py

Основные ключи:

   --alt 2.5        высота поиска, м (потолок 3.0)
   --speed 0.6      скорость перелёта между метками, м/с
   --no-land        не садиться: зависнуть на старте для этапа фигуры
   --strategy       auto | aruco_frame | visual — как лететь к метке

Пример:

   python3 snake_mission/run_leader.py --alt 2.0 --no-land


СТРУКТУРА

   snake_mission/    миссия лидера: поле, маршрут, навигация по меткам
   apple_vision/     распознавание «яблок»

Пороги цвета «яблок» лежат в apple_vision/config/apples.yaml. Под свет зала их надо
переснять на месте:

   python3 apple_vision/tools/calibrate_hsv.py --grab кадр.png

Подробности — в README.md, snake_mission/README.md и apple_vision/README.md.
