"""Распознавание «яблок» с камеры дрона для соревнования «Змейка».

    from apple_vision import AppleVision
    vision = AppleVision(drone, on_apple=on_apple)
    vision.start()
"""

from .config import (
    AppleVisionConfig,
    CameraConfig,
    ColorProfile,
    DetectorConfig,
    RegistryConfig,
    load_config,
    save_config,
)
from .detector import AppleDetector, Detection
from .geometry import CameraIntrinsics, GroundProjector
from .overlay import draw_detections, stack_masks
from .registry import AppleEvent, AppleRegistry
from .vision import AppleVision

__all__ = [
    "AppleVision",
    "AppleDetector",
    "AppleRegistry",
    "AppleEvent",
    "Detection",
    "GroundProjector",
    "CameraIntrinsics",
    "AppleVisionConfig",
    "ColorProfile",
    "DetectorConfig",
    "RegistryConfig",
    "CameraConfig",
    "load_config",
    "save_config",
    "draw_detections",
    "stack_masks",
]

__version__ = "1.0.0"
