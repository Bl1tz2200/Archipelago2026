# apple_vision — распознавание «яблок» с камеры дрона

Модуль для соревнования «Взаимное позиционирование в рое: "Змейка"», п. 2.1.2 регламента:
на полу три «яблока», различимые между собой цветом; при распознавании каждого
цепочка зависает и с земли взлетает следующий дрон; **каждое яблоко засчитывается
однократно**. За этап дают 15 баллов (3 × 5).

Платформа — Обрик ROS 2, работа через `sverk_interfaces` (`drone.image`), топик
камеры `/camera_1/image_raw`, разметка публикуется в `/out_detection`.

## Что делает

1. Берёт кадры с камеры (одиночные или потоком).
2. Ищет яблоки по HSV-цвету + форме (круглость, выпуклость, заполнение окружности) —
   цветная разметка и блики отсекаются.
3. Требует несколько подтверждений подряд, прежде чем засчитать яблоко (защита от
   ложного срабатывания, из-за которого впустую поднялся бы дрон).
4. Считает каждое яблоко один раз: по цвету и, если известна высота, по координате
   на полу в системе `map`.
5. Дёргает ваш коллбек `on_apple(event)` — ровно один раз на яблоко.
6. Публикует размеченный кадр в `/out_detection` и печатает событие в консоль
   (вывод в терминал фиксируют эксперты).

## Быстрый старт

```bash
# на любой машине, без дрона — самопроверка всей цепочки на синтетике
python3 tools/offline_test.py --demo

# тесты
python3 tests/test_apple_vision.py        # или: python3 -m pytest tests/ -q

# на дроне (внутри контейнера sverk_ros2), пропеллеры сняты, яблоко под камерой
python3 run_apple_detect.py --duration 60

# полёт: взлёт, обход сетки змейкой, поиск яблок
python3 run_apple_detect.py --fly --alt 1.5 --step 1.0 --grid 3
```

Копировать на дрон целиком:

```bash
scp -r apple_vision sverk@<IP>:~/
ssh sverk@<IP>
cd ~/apple_vision && python3 tools/offline_test.py --demo
```

## Встраивание в полётную программу лидера

```python
import sverk_interfaces
from apple_vision import AppleVision

drone = sverk_interfaces.init(Nodename="leader")

def on_apple(event):
    # Регламент: цепочка останавливается и зависает, взлетает следующий дрон.
    print(event.describe())            # вывод в терминал для экспертов
    hold_position()                    # ваша команда «зависнуть»
    launch_next_drone(event.index)     # ваша команда рою
    wait_until_tail_in_place()         # подтверждение по UWB

vision = AppleVision(drone, on_apple=on_apple)
vision.start()                         # фоновый разбор потока кадров

try:
    for x, y in snake_route():          # ваш обход узлов сетки
        if vision.complete:             # все 3 яблока найдены — сборка окончена
            break
        drone.control.navigate_wait(x=x, y=y, z=1.5, yaw=0.0, frame_id="map",
                                    tolerance=0.2, timeout=15)
finally:
    vision.stop()
    print(vision.summary())
```

Если удобнее без коллбека и без фонового потока — опрашивайте сами:

```python
vision = AppleVision(drone)
while not vision.complete:
    vision.process_once()              # один кадр
    event = vision.poll()              # None или новое засчитанное яблоко
    if event:
        hover_and_launch_next(event)
```

### API

| Что | Зачем |
|---|---|
| `AppleVision(drone, on_apple=..., colors=(), publish=True)` | основной объект |
| `.start(duration=None)` / `.stop()` | фоновый разбор потока кадров |
| `.process(frame)` | разобрать свой кадр (numpy BGR), вернёт новые события |
| `.process_once(timeout)` | взять один кадр с камеры и разобрать |
| `.poll()` / `.wait_for_apple(timeout)` | забрать событие без коллбека |
| `.count` / `.complete` / `.events` / `.summary()` | состояние сборки |
| `.registry.reset()` | сброс между попытками (в зачётном слоте запусков несколько) |

`AppleEvent`: `index` (1..3), `color`, `label`, `pixel`, `world` (x, y в `map` или `None`),
`altitude`, `timestamp`, `describe()`.

## Настройка на площадке — главное

Пороги в `config/apples.yaml` подобраны «в среднем». **Свет в зале другой, поэтому
пороги надо переснять на месте** — это причина №1, по которой яблоко не находится.

```bash
# 1. снять кадр с яблоком в поле зрения (GUI не нужен)
python3 tools/calibrate_hsv.py --grab кадр.png

# 2а. подобрать по прямоугольнику с яблоком (координаты глазами по картинке)
python3 tools/calibrate_hsv.py --image кадр.png --roi 300,220,60,60 --color red --save

# 2б. или мышкой, если гоняете на ноутбуке
python3 tools/calibrate_hsv.py --image кадр.png --color green --save

# 2в. или ползунками вживую
python3 tools/calibrate_hsv.py --image кадр.png --trackbars

# 3. проверить
python3 tools/offline_test.py --image кадр.png --masks --out проверка.png
```

Повторить для каждого из трёх цветов. Инструмент сам определяет, что красный
«заворачивается» через ноль по оттенку, и пишет для него два диапазона.

### Что крутить, если не находит

| Симптом | Что менять в `config/apples.yaml` |
|---|---|
| Яблоко не видно совсем | пороги профиля (пересобрать калибровкой), `min_area` вниз |
| Видно только вблизи | `min_area` вниз (площадь падает квадратично с высотой) |
| Ловит разметку/тени | `min_circularity`, `min_fill` вверх; сузить `S`/`V` снизу |
| Срабатывает раньше времени | `registry.confirm_frames` вверх |
| Реагирует слишком поздно | `confirm_frames` вниз, `resize_width: 480` для скорости |
| Одно яблоко засчиталось дважды | `merge_radius_m` вверх; проверить `unique_by_color: true` |
| Низкий FPS на плате | `resize_width: 320`, `morph_iterations: 1`, `blur_ksize: 3` |

Диагностика масок: `--masks` сохраняет рядом картинку с маской каждого цвета —
сразу видно, попал ли цвет в диапазон.

## Проверка перед зачётом

```bash
ros2 topic hz /camera_1/image_raw        # камера публикует кадры
python3 tools/offline_test.py --demo     # логика цела
python3 run_apple_detect.py --duration 30  # яблоки на полу под камерой
```

Смотреть разметку: веб-интерфейс дрона (`http://<IP>`) или
`ros2 run rqt_image_view rqt_image_view`, топик `/out_detection`.

## Структура

```
apple_vision/
├── apple_vision/
│   ├── config.py      профили цветов и параметры (YAML)
│   ├── detector.py    поиск по HSV + фильтры формы
│   ├── registry.py    подтверждение и однократный зачёт
│   ├── geometry.py    пиксель → координаты пола (map)
│   ├── overlay.py     разметка для /out_detection
│   ├── synthetic.py   генератор тестовых кадров (без камеры)
│   └── vision.py      связка всего + работа с drone
├── config/apples.yaml
├── run_apple_detect.py        боевой запуск на дроне
├── tools/calibrate_hsv.py     подбор порогов на площадке
├── tools/offline_test.py      прогон по кадрам/видео/синтетике
└── tests/test_apple_vision.py
```

## Границы применимости

- Цвета яблок в регламенте не названы — заданы red / green / yellow. Если на площадке
  цвета другие, профили в `config/apples.yaml` переименовываются и калибруются тем же
  инструментом; количество цветов не ограничено тремя.
- Проекция в `map` считает камеру смотрящей строго вниз и не компенсирует крен —
  ошибка на высоте 1.5 м и наклоне 5° около 13 см, что заметно меньше
  `merge_radius_m`. Для дедупликации этого достаточно, для точной привязки — нет.
- Разворот камеры относительно корпуса задаётся `camera.yaw_offset_deg`; если
  координаты яблок в логе уезжают в другую сторону — поправьте его (0/90/180/270).
- Взаимодействием с роем (взлёт следующего дрона, UWB-подтверждение строя) модуль не
  занимается: он только даёт событие в `on_apple`.
