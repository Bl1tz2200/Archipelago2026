"""Загрузка и сохранение конфигурации распознавания «яблок»."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Sequence, Tuple

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "apples.yaml"
)

HsvRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


@dataclass
class ColorProfile:
    """Цветовой профиль одного «яблока»."""

    name: str
    ranges: List[HsvRange]
    label: str = ""
    draw_bgr: Tuple[int, int, int] = (255, 255, 255)
    min_area: float = 400.0
    max_area: float = 120000.0
    min_circularity: float = 0.55
    min_solidity: float = 0.80
    min_fill: float = 0.45
    enabled: bool = True

    def __post_init__(self) -> None:
        self.label = self.label or self.name
        self.ranges = [
            ((int(lo[0]), int(lo[1]), int(lo[2])), (int(hi[0]), int(hi[1]), int(hi[2])))
            for lo, hi in self.ranges
        ]
        self.draw_bgr = tuple(int(c) for c in self.draw_bgr)  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColorProfile":
        ranges = [(tuple(r["lower"]), tuple(r["upper"])) for r in data.get("ranges", [])]
        known = {f for f in cls.__dataclass_fields__ if f != "ranges"}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(ranges=ranges, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["ranges"] = [{"lower": list(lo), "upper": list(hi)} for lo, hi in self.ranges]
        data["draw_bgr"] = list(self.draw_bgr)
        return data


@dataclass
class DetectorConfig:
    blur_ksize: int = 5
    resize_width: int = 0
    morph_ksize: int = 5
    morph_iterations: int = 1
    max_per_color: int = 1
    min_score: float = 0.35


@dataclass
class RegistryConfig:
    confirm_frames: int = 4
    lost_frames: int = 3
    track_radius_px: float = 120.0
    merge_radius_m: float = 0.6
    unique_by_color: bool = True
    max_apples: int = 3


@dataclass
class CameraConfig:
    calibration_file: str = "~/camera_calibrations/latest_calibration.yaml"
    fallback_hfov_deg: float = 65.0
    yaw_offset_deg: float = 0.0
    mirror_x: bool = False


@dataclass
class AppleVisionConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    profiles: List[ColorProfile] = field(default_factory=list)
    path: str = ""

    def profile(self, name: str) -> ColorProfile:
        for p in self.profiles:
            if p.name == name:
                return p
        raise KeyError(f"профиль '{name}' не найден в {self.path or 'конфиге'}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": asdict(self.detector),
            "profiles": [p.to_dict() for p in self.profiles],
            "registry": asdict(self.registry),
            "camera": asdict(self.camera),
        }


def _subset(cls, data: Dict[str, Any]):
    fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in (data or {}).items() if k in fields})


def load_config(path: str = DEFAULT_CONFIG_PATH) -> AppleVisionConfig:
    """Читает YAML-конфиг. Отсутствующие секции заполняются значениями по умолчанию."""
    import yaml

    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = AppleVisionConfig(
        detector=_subset(DetectorConfig, raw.get("detector", {})),
        registry=_subset(RegistryConfig, raw.get("registry", {})),
        camera=_subset(CameraConfig, raw.get("camera", {})),
        profiles=[ColorProfile.from_dict(p) for p in raw.get("profiles", [])],
        path=path,
    )
    if not cfg.profiles:
        raise ValueError(f"в {path} не задан ни один цветовой профиль")
    return cfg


def save_config(cfg: AppleVisionConfig, path: str = "") -> str:
    """Сохраняет конфиг обратно в YAML (используется калибровкой порогов)."""
    import yaml

    target = os.path.expanduser(path or cfg.path or DEFAULT_CONFIG_PATH)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False)
    return target


def enabled_profiles(cfg: AppleVisionConfig, names: Sequence[str] = ()) -> List[ColorProfile]:
    """Активные профили; при заданных `names` — только перечисленные."""
    profiles = [p for p in cfg.profiles if p.enabled]
    if names:
        wanted = set(names)
        profiles = [p for p in profiles if p.name in wanted]
    return profiles
