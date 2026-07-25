"""Совместимость с камерой Обрика, публикующей кадры в YUV.

Бортовая камера (`/usb_cam`) отдаёт топик `/camera_1/image_raw` в формате
`yuv422_yuy2` — 2 байта на пиксель. Штатный `sverk_interfaces.to_cv2` знает
только `rgb`/`bgr` (3 канала) и `mono` (1 канал), а всё остальное молча
трактует как 1-канальное. Для YUYV это даёт

    ValueError: cannot reshape array of size 614400 into shape (480,640,1)

и роняет `take_picture()` и `stream()` ещё до нашей логики.

`patch_image_api(drone)` оборачивает `drone.image.to_cv2` так, чтобы YUV-кадры
конвертировались в BGR через OpenCV, а все известные библиотеке форматы шли
прежним путём. Обёртка на объекте, поэтому чинит сразу оба пути получения кадра
(`take_picture` и `stream` внутри зовут `self.to_cv2`). Вызывать один раз сразу
после `sverk_interfaces.init(...)`.

Идемпотентно и безопасно: если у `drone.image` нет `to_cv2` (симулятор) или
патч уже стоит — ничего не делает.
"""

from __future__ import annotations

import subprocess

import cv2
import numpy as np

# Двухбайтовые YUV 4:2:2 упаковки → код cv2 для перевода в BGR.
_YUV422_TO_BGR = {
    "yuv422_yuy2": cv2.COLOR_YUV2BGR_YUY2,
    "yuyv": cv2.COLOR_YUV2BGR_YUY2,
    "yuv422": cv2.COLOR_YUV2BGR_YUY2,
    "yuv422_uyvy": cv2.COLOR_YUV2BGR_UYVY,
    "uyvy": cv2.COLOR_YUV2BGR_UYVY,
}


def _decode(image_msg, fallback):
    """sensor_msgs/Image → BGR numpy; YUV сами, остальное — штатным to_cv2."""
    enc = (getattr(image_msg, "encoding", "") or "").lower()
    code = _YUV422_TO_BGR.get(enc)
    if code is not None:
        yuv = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            (image_msg.height, image_msg.width, 2)
        )
        return cv2.cvtColor(yuv, code)
    if enc == "nv12":
        yuv = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            (image_msg.height * 3 // 2, image_msg.width)
        )
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    return fallback(image_msg)


def patch_image_api(drone):
    """Научить `drone.image` понимать YUV-кадры камеры. Возвращает drone."""
    image = getattr(drone, "image", None)
    original = getattr(image, "to_cv2", None)
    if original is None or getattr(image, "_yuv_patched", False):
        return drone

    def to_cv2(image_msg, _original=original):
        frame = _decode(image_msg, _original)
        # publish() берёт заголовок последнего кадра — сохраняем, как штатный to_cv2.
        image._last_header = getattr(image_msg, "header", None)
        return frame

    image.to_cv2 = to_cv2
    image._yuv_patched = True
    return drone


# Название ручки авто-баланса белого разнится между версиями UVC-драйвера.
_WB_AUTO_CTRL_NAMES = ("white_balance_automatic", "white_balance_temperature_auto")


def set_white_balance(device: str, auto: bool = True, temperature: int = None) -> bool:
    """Настроить баланс белого камеры через v4l2-ctl (`/usb_cam` — обычная V4L2/UVC-камера).

    Пороги HSV в `config/apples.yaml` калибруются под конкретный баланс белого — если он
    "плавает" в auto-режиме между калибровкой и полётом, цвет яблока на кадре смещается и
    пороги перестают совпадать. Поэтому имеет смысл фиксировать баланс белого (auto=False,
    temperature=<K>) на время калибровки и полёта, а не полагаться на авто-режим камеры.

    Best-effort: если `v4l2-ctl` не установлен или устройство/ручка недоступны — печатает
    предупреждение и возвращает False, не бросает исключение (настройка камеры не должна
    ронять калибровку или миссию).
    """
    for name in _WB_AUTO_CTRL_NAMES:
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", device, "--set-ctrl", f"{name}={1 if auto else 0}"],
                capture_output=True, text=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            print(f"Баланс белого не настроен: {exc}")
            return False
        if result.returncode == 0:
            break
    else:
        print(f"Баланс белого: на {device} не нашлась ручка авто-режима. "
              f"Смотрите доступные: v4l2-ctl -d {device} --list-ctrls")
        return False

    if not auto and temperature is not None:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--set-ctrl", f"white_balance_temperature={int(temperature)}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            print(f"Баланс белого: не удалось задать температуру на {device}: {result.stderr.strip()}")
            return False
    return True
