#!/usr/bin/env python3
"""Головной дрон: ArUco + поиск и стабилизация над яблоками.

Запускать непосредственно в терминале браузерного VSCode на дроне:
  python3 head_drone.py          # камера и отладка без полёта
  python3 head_drone.py --fly    # полёт

Детектор яблок, реестр подтверждений, геометрия и HSV-конфиг взяты из
репозитория Archipelago2026/apple_vision. Код ведомых дронов отсутствует.
"""
from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apple_vision import (  # noqa: E402
    AppleDetector,
    AppleEvent,
    AppleRegistry,
    GroundProjector,
    draw_detections,
    load_config,
)

FIGURES = {0: "квадрат", 42: "прямоугольник", 48: "трапеция", 6: "ромб"}
GRID_SIZE = 7
MIN_GRID_ID = 0
MAX_GRID_ID = GRID_SIZE * GRID_SIZE - 1


def marker_to_grid(marker_id: int) -> Tuple[int, int]:
    """Преобразует ID 0..48 в (строка, столбец) сетки 7x7."""
    if not MIN_GRID_ID <= marker_id <= MAX_GRID_ID:
        raise ValueError(f"ArUco ID {marker_id} вне поля 7x7")
    return marker_id // GRID_SIZE, marker_id % GRID_SIZE


def grid_to_marker(row: int, col: int) -> int:
    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
        raise ValueError(f"Узел ({row}, {col}) вне поля 7x7")
    return row * GRID_SIZE + col


class State(Enum):
    WAIT_CAMERA = auto()
    TAKEOFF = auto()
    SEARCH_START_ARUCO = auto()
    CENTER_START_ARUCO = auto()
    SEARCH_APPLES = auto()
    CENTER_APPLE = auto()
    HOLD_APPLE = auto()
    COMPLETE = auto()
    FAILSAFE = auto()


@dataclass
class ArucoDetection:
    marker_id: int
    center: Tuple[float, float]
    corners: np.ndarray
    area: float


@dataclass
class GridWaypoint:
    marker_id: int
    row: int
    col: int
    x: float
    y: float


@dataclass
class SharedVision:
    frame: Optional[np.ndarray] = None
    apples: Optional[list] = None
    aruco: Optional[ArucoDetection] = None
    last_frame_time: float = 0.0
    fps: float = 0.0
    frame_count: int = 0


class LeaderMission:
    def __init__(self, drone, args: argparse.Namespace) -> None:
        self.drone = drone
        self.args = args
        self.cfg = load_config(args.config)
        # В текущем этапе нужны только красное и зелёное яблоки.
        self.detector = AppleDetector(self.cfg, colors=("red", "green"))
        self.registry = AppleRegistry(self.cfg.registry)
        self.projector = GroundProjector(self.cfg.camera)
        self.shared = SharedVision(apples=[])
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.state = State.WAIT_CAMERA
        self.state_since = time.time()
        self.start_marker_id: Optional[int] = None
        self.target_apple = None
        self.target_apple_world: Optional[Tuple[float, float]] = None
        self.last_event: Optional[AppleEvent] = None
        self.centered_since: Optional[float] = None
        self.route_index = 0
        self.route: List[GridWaypoint] = []
        self.home = None
        self._fps_t0 = time.time()
        self._fps_n = 0
        self._last_log = 0.0
        self._aruco_detector = self._make_aruco_detector()

    def log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] [{self.state.name}] {text}", flush=True)

    def set_state(self, state: State, reason: str = "") -> None:
        if self.state == state:
            return
        old = self.state
        self.state = state
        self.state_since = time.time()
        self.centered_since = None
        self.log(f"STATE {old.name} -> {state.name}" + (f": {reason}" if reason else ""))

    @staticmethod
    def _make_aruco_detector():
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 4
        params.minMarkerPerimeterRate = 0.012
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return cv2.aruco.ArucoDetector(dictionary, params)

    def detect_aruco(self, frame: np.ndarray) -> Optional[ArucoDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        if ids is None:
            return None
        candidates = []
        for c, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            pts = c.reshape(4, 2)
            area = abs(float(cv2.contourArea(pts.astype(np.float32))))
            if MIN_GRID_ID <= marker_id <= MAX_GRID_ID and area >= self.args.aruco_min_area:
                candidates.append(ArucoDetection(marker_id, tuple(pts.mean(axis=0)), pts, area))
        return max(candidates, key=lambda x: x.area) if candidates else None

    def telemetry(self):
        try:
            return self.drone.control.get_telemetry(frame_id="map")
        except Exception as exc:
            self.log(f"ОШИБКА телеметрии: {exc}")
            return None

    def project_apples(self, frame, detections):
        telemetry = self.telemetry()
        worlds = []
        if telemetry is None:
            return [None] * len(detections), None
        position = (float(telemetry.x), float(telemetry.y))
        yaw = float(getattr(telemetry, "yaw", 0.0) or 0.0)
        altitude = float(telemetry.z)
        image_size = (frame.shape[1], frame.shape[0])
        for det in detections:
            worlds.append(self.projector.pixel_to_map(det.center, position, yaw, altitude, image_size))
        return worlds, altitude

    def frame_callback(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            self.log("Получен пустой кадр")
            return
        try:
            apples = self.detector.detect(frame)
            aruco = self.detect_aruco(frame)
            worlds, altitude = self.project_apples(frame, apples)
            # Засчитываем яблоко только в фазе поиска/наведения, а не на старте.
            events = []
            if self.state in (State.SEARCH_APPLES, State.CENTER_APPLE, State.HOLD_APPLE):
                events = self.registry.update(apples, worlds, altitude=altitude)
            for event in events:
                self.last_event = event
                self.log(event.describe())

            self._fps_n += 1
            now = time.time()
            elapsed = now - self._fps_t0
            fps = self.shared.fps
            if elapsed >= 1.0:
                fps = self._fps_n / elapsed
                self._fps_n = 0
                self._fps_t0 = now

            with self.lock:
                self.shared.frame = frame.copy()
                self.shared.apples = apples
                self.shared.aruco = aruco
                self.shared.last_frame_time = now
                self.shared.frame_count += 1
                self.shared.fps = fps

            debug = self.make_debug_frame(frame, apples, aruco, fps)
            self.drone.image.publish(debug)
        except Exception as exc:
            self.log(f"ОШИБКА обработки кадра: {type(exc).__name__}: {exc}")

    def make_debug_frame(self, frame, apples, aruco, fps):
        canvas = draw_detections(
            frame, apples, self.registry,
            f"state={self.state.name} fps={fps:.1f} frame={self.shared.frame_count}",
        )
        h, w = canvas.shape[:2]
        # Дополнительные направляющие для визуального регулятора.
        cv2.line(canvas, (w // 2 - 25, h // 2), (w // 2 + 25, h // 2), (255, 255, 255), 1)
        cv2.line(canvas, (w // 2, h // 2 - 25), (w // 2, h // 2 + 25), (255, 255, 255), 1)
        cv2.circle(canvas, (w // 2, h // 2), self.args.center_tolerance_px, (255, 255, 255), 1)

        if aruco is not None:
            pts = aruco.corners.astype(np.int32)
            cv2.polylines(canvas, [pts], True, (255, 0, 255), 3, cv2.LINE_AA)
            cx, cy = map(int, aruco.center)
            cv2.drawMarker(canvas, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.line(canvas, (w // 2, h // 2), (cx, cy), (255, 0, 255), 2)
            row, col = marker_to_grid(aruco.marker_id)
            figure = FIGURES.get(aruco.marker_id, "-")
            text = (f"ARUCO id={aruco.marker_id} grid=({row},{col}) "
                    f"figure={figure} area={aruco.area:.0f}")
            cv2.putText(canvas, text, (max(5, cx - 100), max(45, cy - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)

        status = f"aruco={self.start_marker_id} | apples={self.registry.count}/3"
        cv2.rectangle(canvas, (0, 26), (w, 52), (0, 0, 0), -1)
        cv2.putText(canvas, status, (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def get_snapshot(self):
        with self.lock:
            return (self.shared.frame, list(self.shared.apples or []), self.shared.aruco,
                    self.shared.last_frame_time, self.shared.fps)

    def start_camera(self) -> None:
        self.log("Запуск потока /camera_1/image_raw -> /out_detection")
        self.drone.image.stream(self.frame_callback)

    def stop_camera(self) -> None:
        try:
            self.drone.image.stop_stream()
        except Exception:
            pass

    def build_route(self) -> None:
        """Строит путь по ArUco-сетке 7x7, начиная с стартового угла.

        Принята нумерация по строкам:
          0..6, 7..13, ..., 42..48.
        Стартовыми допустимыми углами являются 0, 6, 42 и 48.
        """
        if self.start_marker_id not in FIGURES:
            raise RuntimeError(
                f"стартовая метка должна быть угловой {sorted(FIGURES)}, "
                f"получено {self.start_marker_id}"
            )

        t = self.telemetry()
        if t is None:
            raise RuntimeError("нет телеметрии для построения маршрута")
        self.home = (float(t.x), float(t.y))
        start_row, start_col = marker_to_grid(self.start_marker_id)

        row_order = list(range(GRID_SIZE)) if start_row == 0 else list(range(GRID_SIZE - 1, -1, -1))
        first_left_to_right = start_col == 0
        grid_nodes: List[Tuple[int, int]] = []
        for lane_index, row in enumerate(row_order):
            left_to_right = first_left_to_right if lane_index % 2 == 0 else not first_left_to_right
            cols = range(GRID_SIZE) if left_to_right else range(GRID_SIZE - 1, -1, -1)
            for col in cols:
                grid_nodes.append((row, col))

        # Координаты map считаются относительно стартовой угловой метки.
        # marker_step — расстояние между центрами соседних ArUco.
        route: List[GridWaypoint] = []
        for row, col in grid_nodes:
            dx = (col - start_col) * self.args.marker_step
            dy = (row - start_row) * self.args.marker_step
            route.append(GridWaypoint(
                marker_id=grid_to_marker(row, col),
                row=row,
                col=col,
                x=self.home[0] + dx,
                y=self.home[1] + dy,
            ))

        self.route = route
        self.route_index = 0
        self.log(
            f"Маршрут ArUco 7x7 построен: {len(route)} меток; "
            f"старт ID={self.start_marker_id} grid=({start_row},{start_col}); "
            f"шаг={self.args.marker_step:.3f} м"
        )

    def wait_for_expected_marker(self, expected_id: int, timeout: float) -> bool:
        """Ждёт ожидаемую метку и стабилизируется над ней."""
        deadline = time.time() + timeout
        last_seen = None
        while time.time() < deadline and not self.stop_event.is_set():
            frame, _, aruco, frame_time, _ = self.get_snapshot()
            if frame is None or time.time() - frame_time > self.args.camera_timeout:
                time.sleep(0.08)
                continue
            if aruco is None:
                time.sleep(0.08)
                continue
            if aruco.marker_id != expected_id:
                if aruco.marker_id != last_seen:
                    row, col = marker_to_grid(aruco.marker_id)
                    self.log(
                        f"Ожидалась ArUco {expected_id}, вижу {aruco.marker_id} "
                        f"grid=({row},{col})"
                    )
                    last_seen = aruco.marker_id
                time.sleep(0.08)
                continue
            self.log(f"Подтверждена ожидаемая ArUco ID={expected_id}")
            return self.center_over("aruco", self.args.center_timeout)
        return False

    def image_error(self, center, frame):
        h, w = frame.shape[:2]
        return center[0] - w / 2.0, center[1] - h / 2.0

    def correction_body(self, center, frame):
        ex, ey = self.image_error(center, frame)
        # Для нижней камеры: вертикальная ошибка изображения управляет body X,
        # горизонтальная — body Y. Знаки доступны из CLI для стендовой проверки.
        dx = np.clip(self.args.image_y_sign * ey * self.args.vision_gain,
                     -self.args.max_correction, self.args.max_correction)
        dy = np.clip(self.args.image_x_sign * ex * self.args.vision_gain,
                     -self.args.max_correction, self.args.max_correction)
        return float(dx), float(dy), math.hypot(ex, ey)

    def nudge(self, center, frame, label: str) -> float:
        dx, dy, err = self.correction_body(center, frame)
        self.log(f"{label}: pixel_error={err:.1f}px; correction body dx={dx:+.3f} dy={dy:+.3f}")
        # Небольшая относительная команда; следующая итерация заново измеряет ошибку.
        self.drone.control.navigate_wait(
            x=dx, y=dy, z=0.0, yaw=0.0, speed=self.args.center_speed,
            frame_id="body", tolerance=0.04, timeout=2.0,
        )
        return err

    def center_over(self, kind: str, timeout: float) -> bool:
        deadline = time.time() + timeout
        stable_since = None
        while time.time() < deadline and not self.stop_event.is_set():
            frame, apples, aruco, frame_time, _ = self.get_snapshot()
            if frame is None or time.time() - frame_time > self.args.camera_timeout:
                self.log(f"{kind}: нет свежего кадра")
                time.sleep(0.1)
                continue
            if kind == "aruco":
                target = aruco
                center = target.center if target else None
            else:
                # Выбираем незасчитанный объект, ближайший к центру кадра.
                h, w = frame.shape[:2]
                fresh = [d for d in apples if d.color not in self.registry.claimed_colors]
                target = min(fresh, key=lambda d: math.hypot(d.center[0]-w/2, d.center[1]-h/2)) if fresh else None
                center = target.center if target else None
            if center is None:
                stable_since = None
                self.log(f"{kind}: цель потеряна")
                time.sleep(0.12)
                continue
            err = self.nudge(center, frame, kind)
            if err <= self.args.center_tolerance_px:
                stable_since = stable_since or time.time()
                if time.time() - stable_since >= self.args.center_hold:
                    self.log(f"{kind}: центрирование подтверждено {err:.1f}px")
                    return True
            else:
                stable_since = None
            time.sleep(0.08)
        return False

    def run(self) -> None:
        camera_thread = threading.Thread(target=self.start_camera, daemon=True)
        camera_thread.start()
        try:
            self.log(f"Целевой адрес дрона: {self.args.drone_ip}")
            deadline = time.time() + self.args.camera_wait
            while time.time() < deadline:
                _, _, _, frame_time, _ = self.get_snapshot()
                if frame_time > 0:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("камера не дала ни одного кадра")

            if self.args.fly:
                self.set_state(State.TAKEOFF, "взлёт")
                self.drone.control.navigate_wait(
                    x=0, y=0, z=self.args.alt, yaw=0.0, speed=0.45,
                    frame_id="body", auto_arm=True, tolerance=0.12, timeout=15,
                )
            else:
                self.log("Режим без полёта: только камера/детекция/разметка")

            self.set_state(State.SEARCH_START_ARUCO)
            aruco_deadline = time.time() + self.args.aruco_search_timeout
            while time.time() < aruco_deadline:
                frame, _, aruco, _, _ = self.get_snapshot()
                if aruco is not None:
                    self.start_marker_id = aruco.marker_id
                    if aruco.marker_id not in FIGURES:
                        row, col = marker_to_grid(aruco.marker_id)
                        self.log(
                            f"Вижу внутреннюю ArUco ID={aruco.marker_id} grid=({row},{col}); "
                            "ищу угловую стартовую метку"
                        )
                        time.sleep(0.1)
                        continue
                    self.start_marker_id = aruco.marker_id
                    row, col = marker_to_grid(aruco.marker_id)
                    self.log(
                        f"Найдена стартовая ArUco: ID={aruco.marker_id}, "
                        f"grid=({row},{col}), фигура={FIGURES[aruco.marker_id]}"
                    )
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("стартовая ArUco не найдена")

            if self.args.fly:
                self.set_state(State.CENTER_START_ARUCO)
                if not self.center_over("aruco", self.args.center_timeout):
                    raise RuntimeError("не удалось стабилизироваться над ArUco")

            self.build_route()
            self.set_state(State.SEARCH_APPLES)
            while not self.registry.complete and not self.stop_event.is_set():
                if self.route_index >= len(self.route):
                    self.route_index = 0
                    self.log("Маршрут завершён; повтор поиска с первого узла")
                waypoint = self.route[self.route_index]
                self.log(
                    f"Узел {self.route_index+1}/{len(self.route)}: "
                    f"ID={waypoint.marker_id} grid=({waypoint.row},{waypoint.col}) "
                    f"map=({waypoint.x:.2f},{waypoint.y:.2f})"
                )
                if self.args.fly:
                    self.drone.control.navigate_wait(
                        x=waypoint.x, y=waypoint.y, z=self.args.alt, yaw=0.0,
                        speed=self.args.search_speed, frame_id="map",
                        tolerance=0.18, timeout=15,
                    )
                    if not self.wait_for_expected_marker(
                        waypoint.marker_id, self.args.marker_confirm_timeout
                    ):
                        raise RuntimeError(
                            f"не удалось подтвердить ArUco ID={waypoint.marker_id} "
                            f"в узле ({waypoint.row},{waypoint.col})"
                        )
                else:
                    time.sleep(0.5)

                frame, apples, _, _, _ = self.get_snapshot()
                unclaimed = [d for d in apples if d.color not in self.registry.claimed_colors]
                if unclaimed:
                    self.set_state(State.CENTER_APPLE, f"кандидатов={len(unclaimed)}")
                    if self.args.fly and not self.center_over("apple", self.args.center_timeout):
                        self.log("Не удалось удержать яблоко; продолжаю поиск")
                    else:
                        self.set_state(State.HOLD_APPLE)
                        hold_deadline = time.time() + self.args.apple_hold
                        initial_count = self.registry.count
                        while time.time() < hold_deadline and self.registry.count == initial_count:
                            time.sleep(0.05)
                        if self.registry.count > initial_count:
                            event = self.registry.events[-1]
                            self.log(f"ЗАФИКСИРОВАНО: {event.describe()}")
                            self.log("Точка интеграции ведомого дрона пропущена по требованию")
                        else:
                            self.log("Яблоко не набрало подтверждения реестра")
                    self.set_state(State.SEARCH_APPLES)
                self.route_index += 1

            self.set_state(State.COMPLETE, self.registry.events[-1].describe() if self.registry.events else "")
            self.log("Все три яблока найдены. Лидер остаётся в зависании; посадка не запускается автоматически.")
            while self.args.keep_alive and not self.stop_event.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.log("Остановлено оператором")
        except Exception as exc:
            self.set_state(State.FAILSAFE, f"{type(exc).__name__}: {exc}")
            self.log("Команды движения прекращены; автоматическая посадка не выполняется без явного флага")
            if self.args.land_on_error and self.args.fly:
                try:
                    self.drone.control.land()
                except Exception as land_exc:
                    self.log(f"Ошибка посадки: {land_exc}")
            raise
        finally:
            self.stop_camera()
            self.log(f"ИТОГ: {self.registry.count}/{self.cfg.registry.max_apples}")
            for event in self.registry.events:
                self.log(event.describe())


def check_ip(ip: str, timeout: float = 1.0) -> bool:
    try:
        socket.create_connection((ip, 22), timeout=timeout).close()
        return True
    except OSError:
        return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config", "apples.yaml"))
    p.add_argument("--fly", action="store_true")
    p.add_argument("--alt", type=float, default=1.5)
    p.add_argument(
        "--marker-step", type=float, default=1.0,
        help="расстояние между центрами соседних ArUco, м",
    )
    p.add_argument(
        "--marker-confirm-timeout", type=float, default=6.0,
        help="время подтверждения ожидаемой ArUco после перехода, с",
    )
    p.add_argument("--search-speed", type=float, default=0.35)
    p.add_argument("--center-speed", type=float, default=0.18)
    p.add_argument("--vision-gain", type=float, default=0.0018)
    p.add_argument("--max-correction", type=float, default=0.16)
    p.add_argument("--image-x-sign", type=float, default=-1.0)
    p.add_argument("--image-y-sign", type=float, default=-1.0)
    p.add_argument("--center-tolerance-px", type=int, default=14)
    p.add_argument("--center-hold", type=float, default=1.0)
    p.add_argument("--center-timeout", type=float, default=12.0)
    p.add_argument("--apple-hold", type=float, default=2.0)
    p.add_argument("--aruco-min-area", type=float, default=120.0)
    p.add_argument("--aruco-search-timeout", type=float, default=30.0)
    p.add_argument("--camera-wait", type=float, default=8.0)
    p.add_argument("--camera-timeout", type=float, default=1.0)
    p.add_argument("--keep-alive", action="store_true")
    p.add_argument("--land-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("[BOOT] Запуск непосредственно на дроне через браузерный VSCode", flush=True)
    try:
        import sverk_interfaces
    except ImportError:
        print("sverk_interfaces не найден. Запускайте файл внутри контейнера sverk_ros2 на дроне.", flush=True)
        return 2
    # sverk_interfaces работает с локальным ROS 2-графом борта. IP используется
    # для подключения/копирования по SSH, а не как аргумент init().
    drone = sverk_interfaces.init(Nodename="archipelago_leader")
    LeaderMission(drone, args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())