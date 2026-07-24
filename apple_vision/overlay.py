"""Отрисовка результата — кадр уходит в `/out_detection`, его видно в веб-интерфейсе."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .detector import Detection
from .registry import AppleRegistry

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    registry: Optional[AppleRegistry] = None,
    status: str = "",
) -> np.ndarray:
    """Рисует найденные «яблоки» и счётчик. Возвращает копию кадра."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    for det in detections:
        color = tuple(int(c) for c in det.draw_bgr)
        cx, cy = int(det.center[0]), int(det.center[1])
        cv2.circle(canvas, (cx, cy), int(det.radius), color, 2)
        cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)

        claimed = registry is not None and det.color in registry.claimed_colors
        # Размер показан в тех же процентах, в которых задан порог min_size_percent,
        # чтобы значение из кадра можно было прямо перенести в конфиг.
        text = f"{det.color} {det.score:.2f} {det.size_percent:.2f}%"
        if claimed:
            text += " [OK]"
        elif registry is not None:
            hits, need = registry.pending(det.color)
            if hits:
                text += f" {hits}/{need}"
        cv2.putText(canvas, text, (cx - int(det.radius), cy - int(det.radius) - 8),
                    _FONT, 0.5, color, 2, cv2.LINE_AA)

    # Перекрестие центра кадра — ориентир при наведении на цель.
    cv2.drawMarker(canvas, (width // 2, height // 2), (200, 200, 200), cv2.MARKER_TILTED_CROSS, 12, 1)

    if registry is not None:
        header = f"apples: {registry.count}/{registry.config.max_apples}"
        if registry.claimed_colors:
            header += " [" + ",".join(registry.claimed_colors) + "]"
        cv2.rectangle(canvas, (0, 0), (width, 26), (0, 0, 0), -1)
        cv2.putText(canvas, header, (8, 18), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if status:
        cv2.rectangle(canvas, (0, height - 24), (width, height), (0, 0, 0), -1)
        cv2.putText(canvas, status, (8, height - 7), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def stack_masks(masks: dict[str, np.ndarray], width: int = 640) -> Optional[np.ndarray]:
    """Склеивает маски профилей в одну картинку — удобно при отладке порогов."""
    if not masks:
        return None
    tiles = []
    for name, mask in masks.items():
        tile = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, name, (6, 18), _FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    grid = np.hstack(tiles)
    if grid.shape[1] > width:
        scale = width / float(grid.shape[1])
        grid = cv2.resize(grid, (width, max(1, int(grid.shape[0] * scale))))
    return grid
