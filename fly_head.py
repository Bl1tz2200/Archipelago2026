#!/usr/bin/env python3
"""Головной дрон (лидер) на зачётной попытке «Змейки».

Взлёт → чтение стартового углового маркера (он же назначает фигуру) → поисковый
облёт поля → распознавание трёх «яблок» с зависанием над каждым → возврат на
стартовую метку → посадка.

Яблоки различаются по МЕСТУ на поле, а не по цвету: на зачёте несколько яблок
могут оказаться одного цвета. Место считается от видимой метки в узлах сетки, и
два пятна ближе MERGE_STEPS друг к другу считаются одним яблоком.

Хвостовых дронов здесь нет: там, где по п. 2.1.2 регламента с земли взлетает
следующий дрон, лидер просто зависает HOVER_S секунд и печатает событие.
Выполнение назначенной фигуры (п. 2.1.3) в этот файл не входит.

Весь полёт — теми же командами, что в базовом примере из документации:

    drone.control.navigate(x, y, z, yaw, speed, frame_id="body", auto_arm=...)
    time.sleep(...)
    drone.control.land(timeout=10.0)

Никакого offboard-управления setpoint'ами, никакой телеметрии, никаких координат.
Дрон ориентируется только по меткам: цель перелёта — метка, признак прилёта —
метка в центре кадра. Метры появляются лишь как масштаб кадра, и тот считается
по самой метке (её сторона известна), а не по высоте.

Перед вылетом править только блок НАСТРОЙКИ ниже.

    python3 fly_head.py
"""

import math
import time

import cv2
import numpy as np
import sverk_interfaces

# ═══════════════════════════════════════════════════════════════════════
#  НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════════════

ALT = 1.5           # высота полёта, м (потолок по регламенту — 4.0)
SPEED = 0.5         # скорость перелёта, м/с
CLIMB_SPEED = 0.3   # скорость набора высоты — медленнее горизонтальной, взлёт мягче
YAW = 0.0           # курс держим постоянным весь полёт, рад

# Взлёт: пока дрон качает после набора высоты, командовать ему нельзя — наведение
# по прыгающей в кадре метке только раскачивает сильнее. Поэтому сначала ждём,
# пока картинка не устоится, и только потом трогаемся с места.
SETTLE_S = 3.0      # запас на успокоение сверх времени набора высоты, с
SETTLE_FRAMES = 3   # столько кадров подряд метка должна стоять на месте
SETTLE_DRIFT = 0.15 # «стоит на месте»: сдвиг между кадрами меньше этой доли метки
SETTLE_TRIES = 20   # предел ожидания успокоения (кадров), дальше летим как есть
GAIN = 0.7          # какую долю рассчитанной поправки отрабатывать за раз

GRID = 7            # поле 7×7, ID = строка*7 + столбец
FLIP_X = True       # столбцы пронумерованы справа налево (проверено на поле: метка 0 справа)
STEP_M = 1.0        # расстояние между соседними метками на площадке, м
MARKER_M = 0.3      # сторона метки, м — по ней кадр переводится в метры

TOL = 0.08          # «над целью»: смещение меньше этой доли диагонали кадра
TRIES = 8           # попыток довести дрон до одной метки, дальше — следующий узел
ALT_FIX = 0.3       # предел поправки высоты за одну команду, м
LOOK_UP = 0.3       # на столько подняться, если меток не видно вовсе, м
BLIND_FRAMES = 2    # столько кадров подряд без единой метки — и поднимаемся

# Приложение 3: фигура назначается по ID маркера того угла, откуда стартуем.
# Угол заранее не сообщается — читаем метку под дроном сразу после взлёта.
FIGURES = {0: "Квадрат", 42: "Прямоугольник", 48: "Трапеция", 6: "Ромб"}

SEARCH_ROW_STEP = 2     # облёт через строку: на все 49 узлов трёх минут не хватит
SEARCH_LIMIT_S = 180.0  # п. 2.1.3: на сборку формации — не более 3 минут от взлёта лидера
APPLES_TOTAL = 3        # п. 2.1.2: на полигоне три «яблока»
HOVER_S = 2.0           # зависание над яблоком — время взлёта хвостового дрона
MERGE_STEPS = 0.6       # два пятна ближе этой доли шага сетки — одно и то же яблоко

# Цвета «яблок» в HSV: H 0..179, S 0..255, V 0..255.
# Замерено по снимкам с полёта: яблоко на кадре ТЁМНОЕ (V 26..75) и очень насыщенное
# (S 226..255), а пол и жёлтая полоса на нём — S не выше 100. Поэтому яблоко от фона
# отделяет насыщенность, а не яркость: по яркости они почти не отличаются, и верхнего
# предела по V нет вовсе — он ничего не отсекал, зато мешал бы при ярком свете.
# У красного два диапазона — его оттенок лежит по обе стороны нуля.
APPLES = {
    "красное": [((0, 130, 12), (9, 255, 255)), ((168, 130, 12), (179, 255, 255))],
    "жёлтое": [((10, 130, 12), (27, 255, 255))],
    "зелёное": [((28, 130, 12), (46, 255, 255))],
}
APPLE_MIN_PERCENT = 0.04   # с рабочей высоты яблоко занимает 0.06..0.20% кадра
APPLE_MIN_ROUND = 0.55     # круглость: 1.00 — идеальный круг (у яблок выходит 0.75..0.91)

# ═══════════════════════════════════════════════════════════════════════
#  КАМЕРА
# ═══════════════════════════════════════════════════════════════════════

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

_ARUCO_DICT = (cv2.aruco.getPredefinedDictionary if hasattr(cv2.aruco, "getPredefinedDictionary")
               else cv2.aruco.Dictionary_get)(cv2.aruco.DICT_4X4_50)
_ARUCO_PARAMS = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters")
                 else cv2.aruco.DetectorParameters_create())
_ARUCO = (cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)
          if hasattr(cv2.aruco, "ArucoDetector") else None)   # None — OpenCV 4.5/4.6


def patch_yuv(drone):
    """Научить камеру отдавать BGR: бортовая публикует yuv422_yuy2, а to_cv2 его не знает."""
    image = drone.image
    original = getattr(image, "to_cv2", None)
    if original is None:
        return

    def to_cv2(msg):
        if (getattr(msg, "encoding", "") or "").lower() in ("yuv422_yuy2", "yuyv", "yuv422"):
            yuv = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width, 2))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUY2)
        return original(msg)

    image.to_cv2 = to_cv2


def look(drone):
    """Один кадр с камеры (BGR) или None."""
    try:
        return drone.image.take_picture(timeout=2.0)
    except Exception as exc:
        print(f"кадр не получен: {exc}", flush=True)
        return None


# ═══════════════════════════════════════════════════════════════════════
#  ЗРЕНИЕ
# ═══════════════════════════════════════════════════════════════════════


def markers(img):
    """Метки на кадре: {ID: (x, y, сторона в пикселях, поворот в радианах)}.

    Поворот — угол верхней стороны квадрата в кадре. Метки на поле уложены
    одинаково, поэтому их поворот показывает, как повёрнут сам дрон.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if _ARUCO is not None:
        corners, ids, _ = _ARUCO.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, _ARUCO_DICT, parameters=_ARUCO_PARAMS)
    if ids is None or len(ids) == 0:
        return {}

    seen = {}
    for quad, mid in zip(corners, ids.flatten()):
        pts = np.asarray(quad, np.float32).reshape(-1, 2)
        x, y = pts.mean(axis=0)
        # Сторона метки — среднее четырёх рёбер квадрата. Это и есть масштаб кадра.
        side = float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))
        edge = pts[1] - pts[0]
        if side > 1.0:
            seen[int(mid)] = (float(x), float(y), side, math.atan2(float(edge[1]), float(edge[0])))
    return seen


def apples(img):
    """«Яблоки» на кадре: [(цвет, x, y)] — самое крупное годное пятно каждого цвета."""
    hsv = cv2.cvtColor(cv2.medianBlur(img, 5), cv2.COLOR_BGR2HSV)
    min_area = APPLE_MIN_PERCENT / 100.0 * img.shape[0] * img.shape[1]

    found = []
    for name, ranges in APPLES.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        # CLOSE — заштопать дырки от блика на тёмном яблоке (иначе контур рвётся и
        # круглость проваливается), OPEN — убрать точечный шум.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area < min_area or perimeter <= 0:
                continue
            if 4.0 * math.pi * area / (perimeter * perimeter) < APPLE_MIN_ROUND:
                continue
            if best is None or area > best[0]:
                moments = cv2.moments(contour)
                best = (area,
                        moments["m10"] / moments["m00"],
                        moments["m01"] / moments["m00"])
        if best is not None:
            found.append((name, best[1], best[2]))
    return found


def report(target, seen, fruit, side=0.0):
    """Обстановка в консоль: что вокруг видно, есть ли яблоки и на какой мы высоте."""
    ids = " ".join(str(i) for i in sorted(seen)) if seen else "не видно"
    names = ", ".join(name for name, _, _ in fruit)
    now = alt_by_side(side)
    print(f"цель {target:2d} | метки: {ids} | "
          f"{'яблоки: ' + names if names else 'яблок нет'}"
          f"{f' | h≈{now:.1f} м' if now else ''}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  КАРТА ПОЛЯ
# ═══════════════════════════════════════════════════════════════════════


def node(mid):
    """ID метки → (столбец, строка). Это вся карта поля."""
    col, row = mid % GRID, mid // GRID
    return (GRID - 1 - col if FLIP_X else col), row


def mark(col, row):
    """(столбец, строка) → ID метки. Обратная к node."""
    return row * GRID + (GRID - 1 - col if FLIP_X else col)


def nearest(seen, center):
    """Ближайшая к центру кадра метка — та, над которой висим: (ID, (x, y, сторона))."""
    if not seen:
        return None
    mid = min(seen, key=lambda i: math.hypot(seen[i][0] - center[0], seen[i][1] - center[1]))
    return mid, seen[mid]


# ═══════════════════════════════════════════════════════════════════════
#  ВЫСОТА — тоже по метке
# ═══════════════════════════════════════════════════════════════════════
#
# `navigate(frame_id="body", z=0)` держит не высоту, а «сколько было в момент
# команды»: цель считается от текущего положения. При разгоне дрон наклоняется и
# слегка всплывает, следующая команда принимает эту высоту за норму — и ошибка
# копится только вверх, тем быстрее, чем чаще команды (то есть как раз когда метки
# видны и идёт наведение).
#
# Лечится тем же, чем и всё остальное здесь, — меткой: её сторона в пикселях
# обратно пропорциональна высоте. Запоминаем сторону на рабочей высоте сразу после
# взлёта и дальше каждой командой возвращаемся к ней. Телеметрия не нужна.

SIDE_REF = 0.0      # сторона метки на рабочей высоте, пиксели (замер после взлёта)
ANGLE_REF = None    # поворот меток в кадре на взлёте — «нос смотрит туда же, что и тогда»


def alt_by_side(side):
    """Оценка высоты по стороне метки в кадре, м. 0.0 — эталон ещё не замерен."""
    if SIDE_REF <= 1.0 or side <= 1.0:
        return 0.0
    return ALT * SIDE_REF / side


def hold_alt(side):
    """На сколько подняться (+) или опуститься (−), чтобы вернуться на рабочую высоту."""
    now = alt_by_side(side)
    if not now:
        return 0.0
    correction = ALT - now
    if abs(correction) < 0.05:          # мёртвая зона: не дёргаем дрон по мелочи
        return 0.0
    return max(-ALT_FIX, min(ALT_FIX, correction))


# ═══════════════════════════════════════════════════════════════════════
#  КУРС — тоже по метке
# ═══════════════════════════════════════════════════════════════════════
#
# Смещения, посчитанные ПО КАДРУ (метка левее/выше центра), в поправке не
# нуждаются: кадр поворачивается вместе с дроном. А вот смещения ПО КАРТЕ
# («цель на два узла вперёд») заданы в осях поля — если дрон отвернуло, их надо
# развернуть в оси корпуса. Насколько отвернуло, показывает поворот метки в
# кадре: на поле они уложены одинаково.


def turn_error(angle):
    """На сколько дрон отвернулся от курса взлёта, радианы (−π…π)."""
    if ANGLE_REF is None:
        return 0.0
    return (angle - ANGLE_REF + math.pi) % (2 * math.pi) - math.pi


def to_body(forward, left, angle):
    """Вектор из осей поля в оси корпуса с учётом того, что дрон отвернуло."""
    error = turn_error(angle)
    if abs(error) < math.radians(3):        # мелочь, не крутим
        return forward, left
    c, s = math.cos(error), math.sin(error)
    return forward * c + left * s, -forward * s + left * c


def grid_step(seen, side):
    """Шаг сетки в метрах, измеренный по двум соседним меткам в кадре.

    Избавляет от ручной подгонки STEP_M: расстояние между соседями в пикселях,
    переведённое масштабом метки, и есть шаг площадки. Нет пары соседей — 0.0.
    """
    for a in seen:
        for b in seen:
            if a >= b:
                continue
            (acol, arow), (bcol, brow) = node(a), node(b)
            if abs(acol - bcol) + abs(arow - brow) != 1:
                continue                    # не соседи по сетке
            gap = math.hypot(seen[a][0] - seen[b][0], seen[a][1] - seen[b][1])
            return gap * MARKER_M / side
    return 0.0


def place(base, x, y):
    """Где точка кадра лежит на поле: (столбец, строка) в узлах сетки, дробные.

    Отсчёт от видимой метки: сторона метки задаёт масштаб, её ID — узел. Столбцы
    растут вправо по кадру, строки — вверх, ровно как в перелётах (см. `goto`).
    """
    base_id, (bx, by, side, _) = base
    scale = MARKER_M / side / STEP_M          # пиксель → доля шага сетки
    col, row = node(base_id)
    return col + (x - bx) * scale, row - (y - by) * scale


def at(spot):
    """Место на поле для печати: «у метки 23» (ближайший узел, не вылезая за поле)."""
    col = min(max(int(round(spot[0])), 0), GRID - 1)
    row = min(max(int(round(spot[1])), 0), GRID - 1)
    return f"у метки {mark(col, row)}"


def counted(found, colour, spot):
    """Это яблоко уже засчитано?

    Опознаём по МЕСТУ на поле, а не по цвету: на зачёте несколько яблок могут быть
    одного цвета, и зачёт по цвету склеил бы их в одно. Цвет при этом всё равно
    учитывается — два разных цвета рядом это заведомо два разных яблока.
    """
    return any(c == colour and math.hypot(col - spot[0], row - spot[1]) < MERGE_STEPS
               for c, col, row in found)


def search_route(start_id):
    """Маршрут поиска яблок: змейка по полю от стартового угла.

    Соседние узлы маршрута — соседние узлы сетки (п. 2.1.3: перелёты только между
    соседними узлами). Строки берутся через SEARCH_ROW_STEP, а промежуточные —
    проходятся по одному узлу на развороте, чтобы не было прыжка через клетку.
    """
    col0, row0 = node(start_id)
    cols = list(range(GRID)) if col0 == 0 else list(range(GRID - 1, -1, -1))
    step = SEARCH_ROW_STEP if row0 == 0 else -SEARCH_ROW_STEP
    rows = list(range(row0, GRID if row0 == 0 else -1, step))

    route = []
    for i, row in enumerate(rows):
        line = cols if i % 2 == 0 else cols[::-1]
        route += [mark(col, row) for col in line]
        if i + 1 < len(rows):
            # разворот: спускаемся на следующую полосу по одному узлу
            down = 1 if rows[i + 1] > row else -1
            route += [mark(line[-1], r) for r in range(row + down, rows[i + 1], down)]
    return route


# ═══════════════════════════════════════════════════════════════════════
#  ПОЛЁТ
# ═══════════════════════════════════════════════════════════════════════


def fly(drone, forward, left, up=0.0):
    """Смещение по корпусу: x вперёд, y влево, z вверх. Команда и пауза — как в примере."""
    distance = math.hypot(forward, left)
    if distance < 0.05 and abs(up) < 0.05:
        return
    drone.control.navigate(x=float(forward), y=float(left), z=float(up), yaw=YAW,
                           speed=SPEED, frame_id="body", auto_arm=False)
    time.sleep(distance / SPEED + 0.5)


def settle(drone):
    """Дождаться спокойного висения. Возвращает сторону метки на спокойном кадре.

    Признак того, что дрон перестал качать, — метка под ним стоит в кадре: между
    соседними кадрами её центр и размер почти не меняются. Пока этого нет, команд
    не отправляем вовсе: доводка по прыгающей метке раскачивает дрон ещё сильнее.
    """
    previous = None
    calm = 0
    side = 0.0
    for _ in range(SETTLE_TRIES):
        img = look(drone)
        if img is None:
            time.sleep(0.3)
            continue

        base = nearest(markers(img), (img.shape[1] / 2.0, img.shape[0] / 2.0))
        if base is None:
            previous, calm = None, 0
            time.sleep(0.3)
            continue

        mid, (x, y, side, _) = base
        if previous is not None and previous[0] == mid:
            px, py, pside = previous[1]
            drift = math.hypot(x - px, y - py) / side      # сдвиг в долях метки
            zoom = abs(side - pside) / side                # и «дыхание» размера
            if drift < SETTLE_DRIFT and zoom < SETTLE_DRIFT:
                calm += 1
                if calm >= SETTLE_FRAMES:
                    print(f"          висим спокойно (метка {mid}, {side:.0f} px)", flush=True)
                    return side
            else:
                calm = 0
        previous = (mid, (x, y, side))
        time.sleep(0.3)

    print("          успокоиться не вышло — летим как есть", flush=True)
    return side


def approach(drone, colour):
    """Подвестись над яблоком нужного цвета.

    Возвращает место яблока на поле — (столбец, строка) в узлах — или None, если
    яблоко или метки пропали из кадра. Место считается по последнему кадру, когда
    дрон уже над яблоком: так оно точнее всего.
    """
    where = None
    for _ in range(TRIES):
        img = look(drone)
        if img is None:
            time.sleep(0.5)
            continue

        spot = next((s for s in apples(img) if s[0] == colour), None)
        if spot is None:
            return where

        height, width = img.shape[:2]
        cx, cy = width / 2.0, height / 2.0
        _, x, y = spot

        # Масштаб кадра — по стороне видимой метки; без метки метров не посчитать.
        base = nearest(markers(img), (cx, cy))
        if base is None:
            return where
        where = place(base, x, y)

        if math.hypot(x - cx, y - cy) <= TOL * math.hypot(width, height):
            fly(drone, 0.0, 0.0, hold_alt(base[1][2]))   # над яблоком — вернуть высоту
            return where
        # GAIN < 1: не отрабатываем весь промах разом, иначе дрон проскакивает цель.
        scale = MARKER_M / base[1][2] * GAIN
        fly(drone, -(y - cy) * scale, -(x - cx) * scale, hold_alt(base[1][2]))
    return where


def watch(drone, img, found):
    """Новое яблоко в кадре → подлететь, зависнуть, засчитать. True — засчитали.

    Каждое яблоко засчитывается однократно (п. 2.1.2). Опознаётся по месту на поле:
    цвета на зачёте могут повторяться, поэтому «уже видели такой цвет» — не признак
    того же самого яблока.
    """
    base = nearest(markers(img), (img.shape[1] / 2.0, img.shape[0] / 2.0))
    if base is None:
        return False        # без метки места не посчитать — разберёмся на следующем кадре

    for colour, x, y in apples(img):
        if counted(found, colour, place(base, x, y)):
            continue

        # Место с края кадра прикидочное, поэтому подлетаем и уточняем его над яблоком.
        where = approach(drone, colour)
        if where is None or counted(found, colour, where):
            continue

        found.append((colour, where[0], where[1]))
        print(f">>> ЯБЛОКО {len(found)}/{APPLES_TOTAL}: {colour} {at(where)} — "
              f"зависаем {HOVER_S} с (здесь взлетает дрон {len(found) + 1})", flush=True)
        time.sleep(HOVER_S)
        return True
    return False


def goto(drone, target, found):
    """Долететь до метки `target`, попутно высматривая яблоки. True — встали над ней."""
    blind = 0
    for _ in range(TRIES):
        img = look(drone)
        if img is None:
            time.sleep(0.5)
            continue

        seen = markers(img)
        height, width = img.shape[:2]
        cx, cy = width / 2.0, height / 2.0
        base = nearest(seen, (cx, cy))
        report(target, seen, apples(img), base[1][2] if base else 0.0)
        if watch(drone, img, found):
            # Дрон сошёл с узла ради яблока — следующий заход доведёт его до цели.
            continue

        if target in seen:
            x, y, side, _ = seen[target]
            if math.hypot(x - cx, y - cy) <= TOL * math.hypot(width, height):
                print(f"          над меткой {target}", flush=True)
                # Встали над меткой — заодно вернём высоту, если её увело.
                fly(drone, 0.0, 0.0, hold_alt(side))
                return True
            # Пиксели в метры — по стороне самой метки. Камера смотрит вниз:
            # верх кадра — это «вперёд», левый край — «влево».
            # GAIN < 1: отрабатываем не весь промах разом, иначе дрон проскакивает
            # цель и начинает качаться от команды к команде.
            scale = MARKER_M / side * GAIN
            fly(drone, -(y - cy) * scale, -(x - cx) * scale, hold_alt(side))
            continue

        # Цели в кадре нет — идём к ней по карте от той метки, что видно.
        if base is None:
            # Не видно вообще ничего: поднимаемся, чтобы расширить обзор. Как только
            # метка найдётся, hold_alt сам вернёт дрон на рабочую высоту.
            blind += 1
            if blind >= BLIND_FRAMES:
                print("          меток не видно — поднимаемся осмотреться", flush=True)
                fly(drone, 0.0, 0.0, LOOK_UP)
                blind = 0
            else:
                time.sleep(0.5)
            continue

        blind = 0
        base_id, (x, y, side, angle) = base
        scale = MARKER_M / side
        dcol = node(target)[0] - node(base_id)[0]
        drow = node(target)[1] - node(base_id)[1]
        # Строки поля идут вперёд по корпусу, столбцы — влево; к смещению по сетке
        # добавляем то, насколько сама опорная метка сдвинута от центра кадра.
        forward, left = to_body(drow * STEP_M, -dcol * STEP_M, angle)
        fly(drone,
            forward - (y - cy) * scale,
            left - (x - cx) * scale,
            hold_alt(side))

    print(f"          узел {target} пропущен", flush=True)
    return False


def scan(drone):
    """Стартовый угол со взлёта: какая метка под дроном и какую фигуру она назначает.

    Здесь же замеряется эталон высоты: сторона метки в кадре сразу после взлёта —
    это «как выглядит поле с рабочей высоты ALT». Замер делается по спокойному
    кадру. Заодно снимаются ещё два эталона: ANGLE_REF — поворот меток в кадре,
    то есть курс, от которого дальше считается отклонение, и STEP_M — шаг сетки
    площадки, измеренный по двум соседним меткам.
    """
    global SIDE_REF, ANGLE_REF, STEP_M
    calm_side = settle(drone)
    for _ in range(TRIES):
        img = look(drone)
        if img is None:
            time.sleep(0.5)
            continue
        seen = markers(img)
        if not seen:
            print("меток не видно, ищем…", flush=True)
            time.sleep(0.5)
            continue
        under_sight = nearest(seen, (img.shape[1] / 2.0, img.shape[0] / 2.0))
        under = under_sight[0]
        SIDE_REF = calm_side or under_sight[1][2]
        ANGLE_REF = under_sight[1][3]
        measured = grid_step(seen, SIDE_REF)
        fruit = ", ".join(name for name, _, _ in apples(img))
        print("=" * 54, flush=True)
        print(f">>> СТАРТОВЫЙ МАРКЕР: {under}  (узел {node(under)})", flush=True)
        print(f">>> ВЫСОТА {ALT} м = метка {SIDE_REF:.0f} px в кадре (эталон)", flush=True)
        if measured:
            print(f">>> ШАГ СЕТКИ: {measured:.2f} м (замер по соседним меткам, "
                  f"в настройках было {STEP_M:.2f})", flush=True)
            STEP_M = measured
        else:
            print(f">>> ШАГ СЕТКИ: соседей в кадре нет, оставляем {STEP_M:.2f} м", flush=True)
        if under in FIGURES:
            print(f">>> НАЗНАЧЕННАЯ ФИГУРА: {FIGURES[under]}", flush=True)
        else:
            print(">>> ФИГУРА НЕ ОПРЕДЕЛЕНА: метка не угловая", flush=True)
        print(f">>> ВИДНО МЕТОК: {' '.join(str(i) for i in sorted(seen))}", flush=True)
        print(f">>> ЯБЛОКИ: {fruit or 'не видно'}", flush=True)
        print("=" * 54, flush=True)
        return under
    print("стартовый угол не опознан", flush=True)
    return None


def main():
    drone = sverk_interfaces.init(Nodename="fly_head")
    patch_yuv(drone)
    try:
        print(f"ВЗЛЁТ на {ALT} м", flush=True)
        # Три минуты на сборку формации идут с момента взлёта, а не с набора высоты.
        deadline = time.time() + SEARCH_LIMIT_S
        # Набор высоты на своей, пониженной скорости: чем мягче взлёт, тем меньше
        # раскачка наверху. Пауза — время самого набора плюс запас на успокоение.
        drone.control.navigate(x=0.0, y=0.0, z=ALT, yaw=YAW, speed=CLIMB_SPEED,
                               frame_id="body", auto_arm=True)
        time.sleep(ALT / CLIMB_SPEED + SETTLE_S)

        start = scan(drone)
        found = []
        if start is None:
            print("без стартовой метки поиск невозможен", flush=True)
        else:
            print(f"ПОИСК ЯБЛОК: {SEARCH_LIMIT_S:.0f} с или {APPLES_TOTAL} шт.", flush=True)
            for target in search_route(start):
                if len(found) >= APPLES_TOTAL:
                    print("все яблоки найдены", flush=True)
                    break
                if time.time() > deadline:
                    print("время на поиск вышло", flush=True)
                    break
                goto(drone, target, found)

            listing = ", ".join(f"{c} {at((col, row))}" for c, col, row in found)
            print("=" * 54, flush=True)
            print(f">>> НАЙДЕНО ЯБЛОК: {len(found)}/{APPLES_TOTAL} "
                  f"({listing or 'ни одного'})", flush=True)
            print("=" * 54, flush=True)
            print(f"ВОЗВРАТ на стартовую метку {start}", flush=True)
            goto(drone, start, found)
    finally:
        # Взлетели — обязаны сесть, чем бы ни кончился полёт.
        try:
            print("ПОСАДКА", flush=True)
            resp = drone.control.land(timeout=10.0)
            print("land:", resp.success, resp.message, flush=True)
        finally:
            drone.close()


if __name__ == "__main__":
    main()
