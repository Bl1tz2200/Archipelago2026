#!/usr/bin/env python3
"""Змейка по ArUco-меткам: взлёт → чтение меток → облёт маршрута → посадка.

Весь полёт — теми же командами, что в базовом примере из документации:

    drone.control.navigate(x, y, z, yaw, speed, frame_id="body", auto_arm=...)
    time.sleep(...)
    drone.control.land(timeout=10.0)

Никакого offboard-управления setpoint'ами, никакой телеметрии, никаких координат.
Дрон ориентируется только по меткам: цель перелёта — метка, признак прилёта —
метка в центре кадра. Метры появляются лишь как масштаб кадра, и тот считается
по самой метке (её сторона известна), а не по высоте.

Перед вылетом править только блок НАСТРОЙКИ ниже.

    python3 fly_snake.py
"""

import math
import time

import cv2
import numpy as np
import sverk_interfaces

# ═══════════════════════════════════════════════════════════════════════
#  НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════════════

# Маршрут: ID меток в порядке облёта. Змейка по всему полю 7×7 — правьте под задачу,
# для первых проверок оставьте 2-3 соседних ID.
ROUTE = [
    0,  7,  14,  15, 16,  23, 24, 25
]

ALT = 1.5           # высота полёта, м (потолок по задаче — 2.0)
SPEED = 0.5         # скорость перелёта, м/с
YAW = 0.0           # курс держим постоянным весь полёт, рад

GRID = 7            # поле 7×7, ID = строка*7 + столбец
FLIP_X = True       # столбцы пронумерованы справа налево (проверено на поле: метка 0 справа)
STEP_M = 1.0        # расстояние между соседними метками на площадке, м
MARKER_M = 0.3      # сторона метки, м — по ней кадр переводится в метры

TOL = 0.08          # «над меткой»: смещение меньше этой доли диагонали кадра
TRIES = 8           # попыток довести дрон до одной метки, дальше — следующий узел

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
    """Какие «яблоки» видно на кадре: список названий цветов."""
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
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area < min_area or perimeter <= 0:
                continue
            if 4.0 * math.pi * area / (perimeter * perimeter) < APPLE_MIN_ROUND:
                continue
            found.append(name)
            break
    return found


def report(target, seen, fruit):
    """Обстановка в консоль: что вокруг видно и есть ли яблоки."""
    ids = " ".join(str(i) for i in sorted(seen)) if seen else "не видно"
    print(f"цель {target:2d} | метки: {ids} | "
          f"{'яблоки: ' + ', '.join(fruit) if fruit else 'яблок нет'}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  КАРТА ПОЛЯ
# ═══════════════════════════════════════════════════════════════════════


def node(mid):
    """ID метки → (столбец, строка). Это вся карта поля."""
    col, row = mid % GRID, mid // GRID
    return (GRID - 1 - col if FLIP_X else col), row


def nearest(seen, center):
    """Ближайшая к центру кадра метка — та, над которой висим: (ID, (x, y, сторона))."""
    if not seen:
        return None
    mid = min(seen, key=lambda i: math.hypot(seen[i][0] - center[0], seen[i][1] - center[1]))
    return mid, seen[mid]


# ═══════════════════════════════════════════════════════════════════════
#  ПОЛЁТ
# ═══════════════════════════════════════════════════════════════════════


def fly(drone, forward, left):
    """Смещение по корпусу: x вперёд, y влево. Команда и пауза на перелёт — как в примере."""
    distance = math.hypot(forward, left)
    if distance < 0.05:
        return


    
    drone.control.navigate(x=float(forward), y=float(left), z=-0.5, yaw=YAW,
                           speed=SPEED, frame_id="body", auto_arm=False)

    
    time.sleep(distance / SPEED + 0.5)


def goto(drone, target):
    """Долететь до метки `target`. True — встали над ней."""
    for _ in range(TRIES):
        img = look(drone)
        if img is None:
            time.sleep(0.5)
            continue

        seen = markers(img)
        report(target, seen, apples(img))

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
    """Сканирование поля со взлёта: какая метка под дроном и что видно вокруг."""
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
        under = nearest(seen, (img.shape[1] / 2.0, img.shape[0] / 2.0))
        print("=" * 54, flush=True)
        print(f">>> МЕТКА ПОД ДРОНОМ: {under[0]}  (узел {node(under[0])})", flush=True)
        print(f">>> ВИДНО МЕТОК: {' '.join(str(i) for i in sorted(seen))}", flush=True)
        print(f">>> ЯБЛОКИ: {', '.join(apples(img)) or 'не видно'}", flush=True)
        print("=" * 54, flush=True)
        return under[0]
    print("поле не опознано — летим по маршруту вслепую", flush=True)
    return None


def main():
    drone = sverk_interfaces.init(Nodename="fly_snake")
    patch_yuv(drone)
    try:
        print(f"ВЗЛЁТ на {ALT} м", flush=True)

        drone.control.navigate(x=0.0, y=0.0, z=ALT, yaw=YAW, speed=SPEED,
                               frame_id="body", auto_arm=True)

        time.sleep(5.0)

        scan(drone)
        for target in ROUTE:
            goto(drone, target)
    finally:
        # Взлетели — обязаны сесть, чем бы ни кончился маршрут.
        try:
            print("ПОСАДКА", flush=True)
            resp = drone.control.land(timeout=10.0)
            print("land:", resp.success, resp.message, flush=True)
        finally:
            drone.close()


if __name__ == "__main__":
    main()
