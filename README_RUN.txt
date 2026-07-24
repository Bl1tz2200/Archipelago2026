ЗАПУСК ИЗ БРАУЗЕРНОГО VSCODE ДРОНА

1. Откройте браузерный VSCode дрона по адресу 192.168.1.126.
2. Загрузите и распакуйте репозиторий Archipelago2026 без изменения структуры каталогов.
3. Откройте встроенный терминал VSCode и перейдите в каталог проекта:

   cd ~/Archipelago2026


ВАРИАНТ 1: head_drone.py — миссия лидера одним файлом

4. Проверка камеры и распознавания БЕЗ полёта:

   python3 head_drone.py

   Размеченный поток публикуется в /out_detection.

5. Полётный запуск только после проверки камеры и направления коррекции:

   python3 head_drone.py --fly

Основные параметры:

   --alt 1.5              высота полёта, м
   --marker-step 1.0      расстояние между центрами соседних ArUco, м
   --search-speed 0.35    скорость перелёта между узлами, м/с
   --apple-hold 2.0       зависание над «яблоком», с

Пример:

   python3 head_drone.py --fly --alt 1.5 --marker-step 1.0

Если дрон корректируется от цели, а не к ней:

   --image-x-sign 1 --image-y-sign 1


ВАРИАНТ 2: snake_mission — миссия лидера пакетом

4. Что видит камера над полем меток (пропеллеры сняты):

   python3 snake_mission/tools/marker_check.py --duration 30

5. Расчёт маршрута без полёта — покрытие поля и оценка времени:

   python3 snake_mission/run_leader.py --dry-run --start 42

6. Полётный запуск:

   python3 snake_mission/run_leader.py
   python3 snake_mission/run_leader.py --alt 2.0 --no-land

Отличия двух вариантов и выбор между ними — в README.md.


СТРУКТУРА

   apple_vision/     распознавание «яблок», общая основа для обеих программ
   snake_mission/    миссия лидера пакетом: поле, маршрут, навигация по меткам
   head_drone.py     миссия лидера одним файлом

Пороги цвета «яблок» лежат в apple_vision/config/apples.yaml и используются обеими
программами. Под свет зала их надо переснять на месте:

   python3 apple_vision/tools/calibrate_hsv.py --grab кадр.png
