"""Save and load annotation configurations (v2 schema).

Supports v1 backward compatibility: loads old configs and fills defaults.
"""

import json
import logging
from datetime import datetime

logger = logging.getLogger("sam2studio.annotation_config")

CURRENT_VERSION = "2.0"


def save_annotation_config(
    filepath: str,
    input_points: list,
    input_labels: list,
    model_name: str = "",
    device: str = "",
    threshold: float = 0.0,
    start_frame: int = 1,
    end_frame: int = 0,
    intermediate_format: str = "JPEG (fast)",
    correction_points: list | None = None,
    smoothing_spatial: dict | None = None,
    smoothing_temporal: dict | None = None,
    shapes: list | None = None,
) -> None:
    """Save annotation points and experiment config to JSON file (v2)."""
    # Serialize ShapeOverlay objects to plain dicts
    shape_dicts = []
    if shapes:
        for s in shapes:
            shape_dicts.append({
                "mode": s.mode,
                "shape_type": s.shape_type,
                "points": s.points,
            })

    config = {
        "version": CURRENT_VERSION,
        "saved_at": datetime.now().isoformat(),
        "annotation": {
            "points": input_points,
            "labels": input_labels,
            "correction_points": correction_points or [],
        },
        "parameters": {
            "model": model_name,
            "device": device,
            "threshold": threshold,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "intermediate_format": intermediate_format,
        },
        "smoothing": {
            "spatial": smoothing_spatial or {
                "iterations": 50, "dt": 0.1, "kappa": 30.0, "option": 1
            },
            "temporal": smoothing_temporal or {
                "sigma": 2.0, "neighbors": 2
            },
        },
        "shapes": shape_dicts,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {filepath}")


def load_annotation_config(filepath: str) -> dict:
    """Load annotation config from JSON file.

    Supports both v1 and v2 formats.
    Returns normalized v2 dict.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "annotation" not in config:
        raise ValueError("Invalid config file: missing 'annotation' section")
    if "points" not in config["annotation"] or "labels" not in config["annotation"]:
        raise ValueError("Invalid config file: missing points or labels")

    # v1 -> v2 migration
    version = config.get("version", "1.0")
    if version == "1.0":
        logger.info("Migrating config from v1 to v2")
        config["annotation"].setdefault("correction_points", [])
        config.setdefault("smoothing", {
            "spatial": {"iterations": 50, "dt": 0.1, "kappa": 30.0, "option": 1},
            "temporal": {"sigma": 2.0, "neighbors": 2},
        })
        config["version"] = CURRENT_VERSION

    # Ensure shapes key exists (added in v2.1)
    config.setdefault("shapes", [])

    # Convert shape dicts to tuples for points (JSON arrays → tuples)
    normalized_shapes = []
    for s in config["shapes"]:
        pts = s.get("points", ())
        # Convert nested lists to tuple of tuples for polygon
        if isinstance(pts, list) and pts and isinstance(pts[0], list):
            pts = tuple(tuple(p) for p in pts)
        elif isinstance(pts, list):
            pts = tuple(pts)
        normalized_shapes.append({
            "mode": s["mode"],
            "shape_type": s["shape_type"],
            "points": pts,
        })
    config["shapes"] = normalized_shapes

    return config
