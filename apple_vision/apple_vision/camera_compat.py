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
