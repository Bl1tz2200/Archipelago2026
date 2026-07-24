"""Загрузка и сохранение конфигурации распознавания «яблок»."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Sequence, Tuple

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "apples.yaml"
)

HsvRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


# Старые имена полей → новые. Конфиг, написанный до переименования, читается как есть.
LEGACY_KEYS = {
    "min_area": "min_size_percent",
    "max_area": "max_size_percent",
    "min_circularity": "min_roundness",
    "min_solidity": "min_smoothness",
    "min_fill": "min_filling",
}


@dataclass
class ColorProfile:
    """Цветовой профиль одного «яблока».

    Размер задаётся долей площади кадра, а не пикселями: одна и та же настройка
    работает и на камере дрона 640x480, и на вебке 1920x1080.
    """

    name: str
    ranges: List[HsvRange]
    label: str = ""
    draw_bgr: Tuple[int, int, int] = (255, 255, 255)
    min_size_percent: float = 0.05     # меньше этой доли кадра — шум
    max_size_percent: float = 25.0     # больше — уже не яблоко, а фон в кадре
    min_roundness: float = 0.55        # 1.0 = идеальный круг
    min_smoothness: float = 0.80       # 1.0 = ровный контур без вырезов
    min_filling: float = 0.45          # насколько плотно пятно заполняет свой круг
    enabled: bool = True

    def __post_init__(self) -> None:
        self.label = self.label or self.name
        self.ranges = [
            ((int(lo[0]), int(lo[1]), int(lo[2])), (int(hi[0]), int(hi[1]), int(hi[2])))
            for lo, hi in self.ranges
        ]
        self.draw_bgr = tuple(int(c) for c in self.draw_bgr)  # type: ignore[assignment]

    def area_limits(self, frame_area: float) -> Tuple[float, float]:
        """Границы площади пятна в пикселях для кадра заданной площади."""
        return (self.min_size_percent / 100.0 * frame_area,
                self.max_size_percent / 100.0 * frame_area)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColorProfile":
        ranges = [(tuple(r["lower"]), tuple(r["upper"])) for r in data.get("ranges", [])]
        known = {f for f in cls.__dataclass_fields__ if f != "ranges"}  # type: ignore[attr-defined]
        kwargs = {}
        for key, value in data.items():
            if key in LEGACY_KEYS:
                # Старые пороги были в пикселях при 640x480 — переводим в проценты.
                kwargs.setdefault(LEGACY_KEYS[key], value / 3072.0
                                  if key in ("min_area", "max_area") else value)
            elif key in known:
                kwargs[key] = value
        return cls(ranges=ranges, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["ranges"] = [{"lower": list(lo), "upper": list(hi)} for lo, hi in self.ranges]
        data["draw_bgr"] = list(self.draw_bgr)
        return data


@dataclass
class DetectorConfig:
    blur_ksize: int = 5            # сглаживание перед разбором цвета, 0 = выключено
    resize_width: int = 0          # 0 = кадр как есть; меньше = быстрее, но грубее
    morph_ksize: int = 5           # чистка маски от точечного шума
    morph_iterations: int = 1
    max_per_color: int = 1         # яблок одного цвета на поле
    min_score: float = 0.35        # общая оценка «похоже на яблоко», 0..1


@dataclass
class RegistryConfig:
    confirm_frames: int = 4        # кадров подряд до зачёта: больше = надёжнее, но позже
    lost_frames: int = 3           # кадров без цели, после которых счётчик обнуляется
    track_radius_px: float = 120.0  # насколько центр может прыгнуть между кадрами
    merge_radius_m: float = 0.6    # ближе этого на полу — то же самое яблоко
    unique_by_color: bool = True   # цвета яблок различны → цвет и есть их номер
    max_apples: int = 3            # сколько всего яблок на поле


@dataclass
class CameraConfig:
    calibration_file: str = "~/camera_calibrations/latest_calibration.yaml"
    fallback_hfov_deg: float = 65.0  # угол обзора, если файла калибровки нет
    yaw_offset_deg: float = 0.0      # поворот камеры относительно носа дрона
    mirror_x: bool = False           # true, если картинка зеркальна по горизонтали


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


HEADER = """\
# ═══════════════════════════════════════════════════════════════════════
#  Настройки распознавания «яблок»
# ═══════════════════════════════════════════════════════════════════════
#
#  ЧТО ДЕЛАТЬ, ЕСЛИ...
#
#  яблоко не находится вообще ...... переснять цвет: tools/calibrate_hsv.py
#                                    (или в окне вебки: навести в рамку и нажать c)
#  находится только вблизи ......... min_size_percent → меньше
#  ловит лишнее на фоне ............ min_size_percent → больше,
#                                    min_roundness / min_filling → больше
#  срабатывает раньше времени ...... confirm_frames → больше
#  реагирует слишком поздно ........ confirm_frames → меньше
#  тормозит на плате дрона ......... resize_width: 480, blur_ksize: 3
#  одно яблоко засчиталось дважды .. merge_radius_m → больше
#
#  Проверить, что получилось:
#      python3 tools/offline_test.py --image кадр.png --masks --out проверка.png
#
#  Размеры заданы в ПРОЦЕНТАХ ПЛОЩАДИ КАДРА, а не в пикселях, поэтому одни и те
#  же значения работают и на камере дрона, и на вебке с другим разрешением.
# ═══════════════════════════════════════════════════════════════════════

"""

_DETECTOR_COMMENTS = {
    "blur_ksize": "сглаживание перед разбором цвета (нечётное), 0 = выключить",
    "resize_width": "0 = кадр как есть; 480 или 320 = быстрее, но грубее",
    "morph_ksize": "чистка маски от точечного шума",
    "morph_iterations": "сколько раз чистить",
    "max_per_color": "сколько яблок одного цвета может быть на поле",
    "min_score": "общая оценка «похоже на яблоко», 0..1",
}

_PROFILE_COMMENTS = {
    "min_size_percent": "меньше этой доли кадра — шум, не яблоко",
    "max_size_percent": "больше — это уже не яблоко",
    "min_roundness": "круглость: 1.00 = идеальный круг",
    "min_smoothness": "ровность контура: 1.00 = без вырезов и заусенцев",
    "min_filling": "плотность: насколько пятно заполняет свой круг",
    "enabled": "false = не искать этот цвет",
}

_REGISTRY_COMMENTS = {
    "confirm_frames": "кадров подряд до зачёта: больше = надёжнее, но позже",
    "lost_frames": "кадров без цели, после которых счётчик подтверждений обнуляется",
    "track_radius_px": "насколько центр может прыгнуть между кадрами, пиксели",
    "merge_radius_m": "ближе этого на полу — считаем тем же яблоком, метры",
    "unique_by_color": "цвета яблок различны → цвет и есть номер яблока",
    "max_apples": "сколько всего яблок на поле (по регламенту 3)",
}

_CAMERA_COMMENTS = {
    "calibration_file": "результат калибровки камеры; нет файла — берётся угол ниже",
    "fallback_hfov_deg": "горизонтальный угол обзора камеры, градусы",
    "yaw_offset_deg": "поворот камеры относительно носа дрона: 0 / 90 / 180 / 270",
    "mirror_x": "true, если картинка зеркальна по горизонтали",
}


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_scalar(v) for v in value) + "]"
    if isinstance(value, str) and (value.startswith("~") or ":" in value):
        return f'"{value}"'
    return str(value)


def _section(title: str, data: Dict[str, Any], comments: Dict[str, str], indent: str = "  ") -> str:
    width = max((len(f"{k}: {_scalar(v)}") for k, v in data.items()), default=0)
    lines = [f"{title}:"]
    for key, value in data.items():
        pair = f"{key}: {_scalar(value)}"
        comment = comments.get(key)
        lines.append(f"{indent}{pair.ljust(width) + '   # ' + comment if comment else pair}")
    return "\n".join(lines)


def dump_config_yaml(cfg: AppleVisionConfig) -> str:
    """YAML с пояснениями. Используется при записи, чтобы файл не терял комментарии."""
    parts = [HEADER, _section("detector", asdict(cfg.detector), _DETECTOR_COMMENTS), ""]

    parts.append("# ── ЦВЕТА ЯБЛОК ───────────────────────────────────────────────────────")
    parts.append("# HSV: H — оттенок 0..179, S — насыщенность 0..255, V — яркость 0..255.")
    parts.append("# Проще всего не править руками, а переснять: tools/calibrate_hsv.py")
    parts.append("profiles:")
    for profile in cfg.profiles:
        data = profile.to_dict()
        parts.append("")
        parts.append(f"  - name: {data['name']}")
        parts.append(f"    label: \"{data['label']}\"")
        parts.append(f"    draw_bgr: {_scalar(data['draw_bgr'])}   # цвет обводки на картинке (B, G, R)")
        if len(data["ranges"]) > 1:
            parts.append("    ranges:            # два диапазона: оттенок лежит по обе стороны от нуля")
        else:
            parts.append("    ranges:")
        for rng in data["ranges"]:
            parts.append(f"      - {{lower: {_scalar(rng['lower'])}, upper: {_scalar(rng['upper'])}}}")
        shape = {k: v for k, v in data.items()
                 if k not in ("name", "label", "draw_bgr", "ranges")}
        body = _section("", shape, _PROFILE_COMMENTS, indent="    ").split("\n")[1:]
        parts.extend(body)

    parts.append("")
    parts.append("# ── КОГДА СЧИТАТЬ ЯБЛОКО НАЙДЕННЫМ ────────────────────────────────────")
    parts.append(_section("registry", asdict(cfg.registry), _REGISTRY_COMMENTS))
    parts.append("")
    parts.append("# ── КАМЕРА (нужна только для координат яблок в логе) ───────────────────")
    parts.append(_section("camera", asdict(cfg.camera), _CAMERA_COMMENTS))
    parts.append("")
    return "\n".join(parts)


def save_config(cfg: AppleVisionConfig, path: str = "") -> str:
    """Сохраняет конфиг в YAML вместе с пояснениями (вызывается из калибровки)."""
    target = os.path.expanduser(path or cfg.path or DEFAULT_CONFIG_PATH)
    with open(target, "w", encoding="utf-8") as f:
        f.write(dump_config_yaml(cfg))
    return target


def enabled_profiles(cfg: AppleVisionConfig, names: Sequence[str] = ()) -> List[ColorProfile]:
    """Активные профили; при заданных `names` — только перечисленные."""
    profiles = [p for p in cfg.profiles if p.enabled]
    if names:
        wanted = set(names)
        profiles = [p for p in profiles if p.name in wanted]
    return profiles
