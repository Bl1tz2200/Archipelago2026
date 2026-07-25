"""Связка «камера → детектор → однократный зачёт → событие».

Это то, что подключается к полётной программе лидера:

    from apple_vision import AppleVision

    vision = AppleVision(drone, on_apple=lambda e: launch_next_drone(e.index))
    vision.start()          # фоновый разбор кадров с /camera_1/image_raw
    ...  полёт по узлам сетки ...
    vision.stop()

Либо синхронно, кадр за кадром, внутри собственного цикла миссии:

    if vision.poll():       # True, если только что засчитано новое яблоко
        hover_and_launch_next()
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .config import AppleVisionConfig, load_config
from .detector import AppleDetector, Detection
from .geometry import GroundProjector
from .overlay import draw_detections
from .registry import AppleEvent, AppleRegistry

AppleCallback = Callable[[AppleEvent], None]


class AppleVision:
    """Распознавание «яблок» с камеры дрона с однократным зачётом каждого.

    Аргументы:
        drone:      объект `sverk_interfaces.init(...)`; можно None — тогда
                    класс работает по кадрам, которые передают в `process()`.
        on_apple:   вызывается один раз на каждое засчитанное яблоко.
        config:     путь к YAML или готовый AppleVisionConfig.
        colors:     ограничить набор цветов (по умолчанию все из конфига).
        publish:    публиковать размеченный кадр в `/out_detection`.
        verbose:    печатать события и прогресс в консоль.
    """

    def __init__(
        self,
        drone=None,
        on_apple: Optional[AppleCallback] = None,
        config: AppleVisionConfig | str | None = None,
        colors: Sequence[str] = (),
        publish: bool = True,
        verbose: bool = True,
        use_telemetry: bool = True,
    ) -> None:
        if config is None or isinstance(config, str):
            config = load_config(config) if config else load_config()
        self.config: AppleVisionConfig = config
        self.drone = drone
        self.on_apple = on_apple
        self.publish = publish and drone is not None
        self.verbose = verbose
        # get_telemetry() виснет вечно, если FCU не публикует телеметрию (стенд без
        # него) — обычный timeout/SIGTERM это не берёт. При use_telemetry=False
        # регистрация идёт по цвету, без проекции в мировые координаты.
        self.use_telemetry = use_telemetry and drone is not None

        self.detector = AppleDetector(config, colors)
        self.registry = AppleRegistry(config.registry)
        self.projector = GroundProjector(config.camera)

        self.last_detections: List[Detection] = []
        self.last_frame: Optional[np.ndarray] = None
        self.frames_processed = 0
        self.fps = 0.0

        self._new_events: List[AppleEvent] = []
        self._lock = threading.Lock()
        self._streaming = False
        self._last_fps_time = time.time()
        self._fps_counter = 0

    # ------------------------------------------------------------- телеметрия

    def _telemetry(self) -> Tuple[Optional[Tuple[float, float]], float, Optional[float]]:
        """Позиция в map, курс и высота. При недоступной телеметрии — (None, 0, None)."""
        if not self.use_telemetry:
            return None, 0.0, None
        try:
            t = self.drone.control.get_telemetry(frame_id="map")
        except Exception:
            return None, 0.0, None
        if t is None:
            return None, 0.0, None
        try:
            xy = (float(t.x), float(t.y))
            return xy, float(getattr(t, "yaw", 0.0) or 0.0), float(t.z)
        except (TypeError, ValueError):
            return None, 0.0, None

    # ------------------------------------------------------------- обработка

    def process(self, frame: np.ndarray) -> List[AppleEvent]:
        """Разбирает один кадр. Возвращает события, засчитанные именно на нём."""
        if frame is None:
            return []

        detections = self.detector.detect(frame)
        position, yaw, altitude = self._telemetry()

        worlds: List[Optional[Tuple[float, float]]] = []
        size = (frame.shape[1], frame.shape[0])
        for det in detections:
            if position is None or altitude is None:
                worlds.append(None)
            else:
                worlds.append(self.projector.pixel_to_map(det.center, position, yaw, altitude, size))

        events = self.registry.update(detections, worlds, altitude=altitude)

        self.last_detections = detections
        self.last_frame = frame
        self.frames_processed += 1
        self._tick_fps()

        if self.publish:
            self._publish(frame, detections)

        for event in events:
            if self.verbose:
                print(f"[apple_vision] {event.describe()}", flush=True)
            with self._lock:
                self._new_events.append(event)
            if self.on_apple is not None:
                try:
                    self.on_apple(event)
                except Exception as exc:  # коллбек не должен ронять поток стрима
                    print(f"[apple_vision] ошибка в on_apple: {exc}", flush=True)

        return events

    def _tick_fps(self) -> None:
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self.fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._last_fps_time = now

    def _publish(self, frame: np.ndarray, detections: Sequence[Detection]) -> None:
        status = f"fps {self.fps:.1f} | frames {self.frames_processed}"
        try:
            self.drone.image.publish(draw_detections(frame, detections, self.registry, status))
        except Exception as exc:
            self.publish = False  # не спамим ошибкой на каждом кадре
            print(f"[apple_vision] публикация в /out_detection отключена: {exc}", flush=True)

    # ------------------------------------------------------------ режимы работы

    def process_once(self, timeout: float = 2.0) -> List[AppleEvent]:
        """Берёт один кадр с камеры и разбирает его."""
        if self.drone is None:
            raise RuntimeError("нет объекта drone: передавайте кадры в process()")
        frame = self.drone.image.take_picture(timeout=timeout)
        if frame is None:
            return []
        return self.process(frame)

    def start(self, duration: Optional[float] = None) -> None:
        """Запускает фоновый разбор потока кадров (не блокирует полётную логику)."""
        if self.drone is None:
            raise RuntimeError("нет объекта drone: передавайте кадры в process()")
        if self._streaming:
            return
        self._streaming = True

        def worker() -> None:
            try:
                if duration is None:
                    self.drone.image.stream(self._on_stream_frame)
                else:
                    self.drone.image.stream(self._on_stream_frame, duration=duration)
            except Exception as exc:
                print(f"[apple_vision] поток кадров остановлен: {exc}", flush=True)
            finally:
                self._streaming = False

        self._thread = threading.Thread(target=worker, name="apple_vision", daemon=True)
        self._thread.start()

    def _on_stream_frame(self, frame: np.ndarray) -> None:
        if self.registry.complete:
            return  # все яблоки найдены — кадры больше не разбираем
        self.process(frame)

    def stop(self) -> None:
        """Останавливает фоновый разбор потока."""
        if self.drone is not None:
            try:
                self.drone.image.stop_stream()
            except Exception:
                pass
        self._streaming = False

    def poll(self) -> Optional[AppleEvent]:
        """Забирает следующее необработанное событие (для собственного цикла миссии)."""
        with self._lock:
            return self._new_events.pop(0) if self._new_events else None

    def wait_for_apple(self, timeout: float = 60.0, poll_interval: float = 0.05) -> Optional[AppleEvent]:
        """Ждёт очередное засчитанное яблоко. None — если не дождались за timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            event = self.poll()
            if event is not None:
                return event
            if not self._streaming and self.drone is not None:
                self.process_once()
            else:
                time.sleep(poll_interval)
        return None

    # ------------------------------------------------------------------ прочее

    @property
    def count(self) -> int:
        """Сколько яблок засчитано."""
        return self.registry.count

    @property
    def complete(self) -> bool:
        """Найдены ли все яблоки (по умолчанию 3)."""
        return self.registry.complete

    @property
    def events(self) -> List[AppleEvent]:
        return list(self.registry.events)

    def summary(self) -> str:
        lines = [f"Засчитано яблок: {self.registry.count}/{self.registry.config.max_apples}"]
        for e in self.registry.events:
            lines.append("  " + e.describe())
        return "\n".join(lines)
