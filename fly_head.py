#!/usr/bin/env python3
"""Головной дрон (лидер) на зачётной попытке «Змейки».

Взлёт → чтение стартового углового маркера (он же назначает фигуру) → поисковый
облёт поля → распознавание трёх «яблок» с зависанием над каждым → возврат на
стартовую метку → посадка.

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
YAW = 0.0           # курс держим постоянным весь полёт, рад

GRID = 7            # поле 7×7, ID = строка*7 + столбец
FLIP_X = True       # столбцы пронумерованы справа налево (проверено на поле: метка 0 справа)
STEP_M = 1.0        # расстояние между соседними метками на площадке, м
MARKER_M = 0.3      # сторона метки, м — по ней кадр переводится в метры

TOL = 0.08          # «над целью»: смещение меньше этой доли диагонали кадра
TRIES = 8           # попыток довести дрон до одной метки, дальше — следующий узел

# Приложение 3: фигура назначается по ID маркера того угла, откуда стартуем.
# Угол заранее не сообщается — читаем метку под дроном сразу после взлёта.
FIGURES = {0: "Квадрат", 42: "Прямоугольник", 48: "Трапеция", 6: "Ромб"}

SEARCH_ROW_STEP = 2     # облёт через строку: на все 49 узлов трёх минут не хватит
SEARCH_LIMIT_S = 180.0  # п. 2.1.3: на сборку формации — не более 3 минут от взлёта лидера
APPLES_TOTAL = 3        # п. 2.1.2: на полигоне три «яблока»
HOVER_S = 2.0           # зависание над яблоком — время взлёта хвостового дрона

# Цвета «яблок» в HSV: H 0..179, S 0..255, V 0..255.
# Замерено по снимкам с полёта: яблоко на кадре ТЁМНОЕ (V 26..75) и очень насыщенное
# (S 226..255), а пол и жёлтая полоса на нём — S не выше 100. Поэтому яблоко от фона
# отделяет насыщенность, а не яркость: по яркости они почти не отличаются.
# У красного два диапазона — его оттенок лежит по обе стороны нуля.
APPLES = {
    "красное": [((0, 130, 12), (9, 255, 200)), ((168, 130, 12), (179, 255, 200))],
    "жёлтое": [((10, 130, 12), (27, 255, 200))],
    "зелёное": [((28, 130, 12), (46, 255, 200))],
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
    """Метки на кадре: {ID: (x, y, сторона в пикселях)}."""
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
        if side > 1.0:
            seen[int(mid)] = (float(x), float(y), side)
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


def report(target, seen, fruit):
    """Обстановка в консоль: что вокруг видно и есть ли яблоки."""
    ids = " ".join(str(i) for i in sorted(seen)) if seen else "не видно"
    names = ", ".join(name for name, _, _ in fruit)
    print(f"цель {target:2d} | метки: {ids} | "
          f"{'яблоки: ' + names if names else 'яблок нет'}", flush=True)


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


def fly(drone, forward, left):
    """Смещение по корпусу: x вперёд, y влево. Команда и пауза на перелёт — как в примере."""
    distance = math.hypot(forward, left)
    if distance < 0.05:
        return
    drone.control.navigate(x=float(forward), y=float(left), z=0.0, yaw=YAW,
                           speed=SPEED, frame_id="body", auto_arm=False)
    time.sleep(distance / SPEED + 0.5)


def approach(drone, colour):
    """Подвестись над яблоком нужного цвета: True — встали над ним."""
    for _ in range(TRIES):
        img = look(drone)
        if img is None:
            time.sleep(0.5)
            continue

        spot = next((s for s in apples(img) if s[0] == colour), None)
        if spot is None:
            return False

        height, width = img.shape[:2]
        cx, cy = width / 2.0, height / 2.0
        _, x, y = spot
        if math.hypot(x - cx, y - cy) <= TOL * math.hypot(width, height):
            return True

        # Масштаб кадра — по стороне видимой метки; без метки метров не посчитать.
        base = nearest(markers(img), (cx, cy))
        if base is None:
            return False
        scale = MARKER_M / base[1][2]
        fly(drone, -(y - cy) * scale, -(x - cx) * scale)
    return False


def watch(drone, img, found):
    """Новое яблоко в кадре → подлететь, зависнуть, засчитать. True — засчитали.

    Каждое яблоко засчитывается однократно (п. 2.1.2), опознаётся по цвету —
    «яблоки визуально различимы между собой цветом», других похожих объектов нет.
    """
    for colour, _, _ in apples(img):
        if colour in found:
            continue
        approach(drone, colour)
        found.append(colour)
        print(f">>> ЯБЛОКО {len(found)}/{APPLES_TOTAL}: {colour} — "
              f"зависаем {HOVER_S} с (здесь взлетает дрон {len(found) + 1})", flush=True)
        time.sleep(HOVER_S)
        return True
    return False


def goto(drone, target, found):
    """Долететь до метки `target`, попутно высматривая яблоки. True — встали над ней."""
    for _ in range(TRIES):
        img = look(drone)
        if img is None:
            time.sleep(0.5)
            continue

        seen = markers(img)
        report(target, seen, apples(img))
        if watch(drone, img, found):
            # Дрон сошёл с узла ради яблока — следующий заход доведёт его до цели.
            continue

        height, width = img.shape[:2]
        cx, cy = width / 2.0, height / 2.0

        if target in seen:
            x, y, side = seen[target]
            if math.hypot(x - cx, y - cy) <= TOL * math.hypot(width, height):
                print(f"          над меткой {target}", flush=True)
                return True
            # Пиксели в метры — по стороне самой метки. Камера смотрит вниз:
            # верх кадра — это «вперёд», левый край — «влево».
            scale = MARKER_M / side
            fly(drone, -(y - cy) * scale, -(x - cx) * scale)
            continue

        # Цели в кадре нет — идём к ней по карте от той метки, что видно.
        base = nearest(seen, (cx, cy))
        if base is None:
            time.sleep(0.5)
            continue
        base_id, (x, y, side) = base
        scale = MARKER_M / side
        dcol = node(target)[0] - node(base_id)[0]
        drow = node(target)[1] - node(base_id)[1]
        # Строки поля идут вперёд по корпусу, столбцы — влево; к смещению по сетке
        # добавляем то, насколько сама опорная метка сдвинута от центра кадра.
        fly(drone,
            drow * STEP_M - (y - cy) * scale,
            -dcol * STEP_M - (x - cx) * scale)

    print(f"          узел {target} пропущен", flush=True)
    return False


def scan(drone):
    """Стартовый угол со взлёта: какая метка под дроном и какую фигуру она назначает."""
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
        under = nearest(seen, (img.shape[1] / 2.0, img.shape[0] / 2.0))[0]
        fruit = ", ".join(name for name, _, _ in apples(img))
        print("=" * 54, flush=True)
        print(f">>> СТАРТОВЫЙ МАРКЕР: {under}  (узел {node(under)})", flush=True)
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
        drone.control.navigate(x=0.0, y=0.0, z=ALT, yaw=YAW, speed=SPEED,
                               frame_id="body", auto_arm=True)
        time.sleep(10.0)

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

            print("=" * 54, flush=True)
            print(f">>> НАЙДЕНО ЯБЛОК: {len(found)}/{APPLES_TOTAL} "
                  f"({', '.join(found) or 'ни одного'})", flush=True)
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
