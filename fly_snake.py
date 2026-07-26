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
STEP_M = 1.0        # расстояние между центрами соседних меток, м (замер на поле: 0.99)
MARKER_M = 0.325    # сторона метки, м — по ней кадр переводится в метры (замер рулеткой)

TOL = 0.08          # «над меткой»: смещение меньше этой доли диагонали кадра
TRIES = 8           # попыток довести дрон до одной метки, дальше — следующий узел
ALT_FIX = 0.3       # предел поправки высоты за одну команду, м
LOOK_UP = 0.3       # на столько подняться, если меток не видно вовсе, м
BLIND_FRAMES = 2    # столько кадров подряд без единой метки — и поднимаемся

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


def report(target, seen, fruit, side=0.0):
    """Обстановка в консоль: что вокруг видно, есть ли яблоки и на какой мы высоте."""
    ids = " ".join(str(i) for i in sorted(seen)) if seen else "не видно"
    now = alt_by_side(side)
    print(f"цель {target:2d} | метки: {ids} | "
          f"{'яблоки: ' + ', '.join(fruit) if fruit else 'яблок нет'}"
          f"{f' | h≈{now:.1f} м' if now else ''}", flush=True)


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


# ═══════════════════════════════════════════════════════════════════════
#  ПОЛЁТ
# ═══════════════════════════════════════════════════════════════════════


def fly(drone, forward, left, up=0.0):
    """Смещение по корпусу: x вперёд, y влево, z вверх. Команда и пауза — как в примере."""
    distance = math.hypot(forward, left)
    if distance < 0.05 and abs(up) < 0.05:
        return
    # Отказ команды печатается: молча продолжать наведение по не улетевшему дрону
    # значит потом гадать, почему он стоял на месте. Пауза считает и вертикаль.
    resp = drone.control.navigate(x=float(forward), y=float(left), z=float(up), yaw=YAW,
                                  speed=SPEED, frame_id="body", auto_arm=False)
    if resp is not None and not getattr(resp, "success", True):
        print(f"          navigate отказал: {getattr(resp, 'message', '')}", flush=True)
    time.sleep(math.hypot(distance, up) / SPEED + 0.5)


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


def goto(drone, target):
    """Долететь до метки `target`. True — встали над ней."""
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

        if target in seen:
            x, y, side, _ = seen[target]
            if math.hypot(x - cx, y - cy) <= TOL * math.hypot(width, height):
                print(f"          над меткой {target}", flush=True)
                # Встали над меткой — заодно вернём высоту, если её увело.
                fly(drone, 0.0, 0.0, hold_alt(side))
                return True
            # Пиксели в метры — по стороне самой метки. Камера смотрит вниз:
            # верх кадра — это «вперёд», левый край — «влево». Поправки по кадру
            # разворота не требуют: кадр поворачивается вместе с дроном.
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
    """Сканирование поля со взлёта: какая метка под дроном и что видно вокруг.

    Здесь же снимаются три эталона на весь полёт — все по спокойному кадру, потому
    что замер на раскачке увёл бы их на весь оставшийся полёт:

      * SIDE_REF  — сторона метки на рабочей высоте («как выглядит поле с ALT»);
      * ANGLE_REF — поворот меток в кадре, то есть курс, от которого потом считаем
        отклонение;
      * STEP_M    — шаг сетки площадки, измеренный по двум соседним меткам.
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
        under = nearest(seen, (img.shape[1] / 2.0, img.shape[0] / 2.0))
        SIDE_REF = calm_side or under[1][2]
        ANGLE_REF = under[1][3]
        measured = grid_step(seen, SIDE_REF)
        print("=" * 54, flush=True)
        print(f">>> МЕТКА ПОД ДРОНОМ: {under[0]}  (узел {node(under[0])})", flush=True)
        print(f">>> ВЫСОТА {ALT} м = метка {SIDE_REF:.0f} px в кадре (эталон)", flush=True)
        if measured:
            print(f">>> ШАГ СЕТКИ: {measured:.2f} м (замер по соседним меткам, "
                  f"в настройках было {STEP_M:.2f})", flush=True)
            STEP_M = measured
        else:
            print(f">>> ШАГ СЕТКИ: соседей в кадре нет, оставляем {STEP_M:.2f} м", flush=True)
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
        # Набор высоты на своей, пониженной скорости: чем мягче взлёт, тем меньше
        # раскачка наверху. Пауза — время самого набора плюс запас на успокоение.
        resp = drone.control.navigate(x=0.0, y=0.0, z=ALT, yaw=YAW, speed=CLIMB_SPEED,
                                      frame_id="body", auto_arm=True)
        print(f"взлёт: {getattr(resp, 'success', '?')} {getattr(resp, 'message', '')}",
              flush=True)
        if resp is not None and not getattr(resp, "success", True):
            # Не встал в OFFBOARD или не заармился — облетать маршрут нечем.
            raise RuntimeError("взлёт не принят — маршрут отменён")
        time.sleep(ALT / CLIMB_SPEED + SETTLE_S)

        scan(drone)
        for target in ROUTE:
            goto(drone, target)
    finally:
        # Взлетели — обязаны сесть, чем бы ни кончился маршрут.
        try:
            print("ПОСАДКА", flush=True)
            try:
                resp = drone.control.land()
            except TypeError:               # сборка со старой сигнатурой
                resp = drone.control.land(timeout=10.0)
            print("land:", getattr(resp, "success", "?"), getattr(resp, "message", ""),
                  flush=True)
            time.sleep(8.0)                 # дать сесть, прежде чем гасить ноду
        finally:
            drone.close()


if __name__ == "__main__":
    main()
