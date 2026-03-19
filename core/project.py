"""Project save/load for full workspace state.

A project file (.s2proj) is a JSON file that captures:
- Input/output directory paths
- Model configuration
- Annotation points and shapes
- Preprocessing config
- Frame range and marked frames
- Overlay display settings
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime

from core.preprocessing import PreprocessingConfig

logger = logging.getLogger("sam2studio.project")

PROJECT_VERSION = "1.0"
PROJECT_EXTENSION = ".s2proj"


def save_project(filepath: str, state) -> None:
    """Save full project state to a .s2proj JSON file.

    Args:
        filepath: Path to save the project file.
        state: AppState instance.
    """
    # Serialize preprocessing config
    pp_config = state.preprocessing_config
    pp_dict = {
        "brightness": pp_config.brightness,
        "contrast": pp_config.contrast,
        "gain": pp_config.gain,
        "clip_min": pp_config.clip_min,
        "clip_max": pp_config.clip_max,
        "gaussian_sigma": pp_config.gaussian_sigma,
        "bilateral_enabled": pp_config.bilateral_enabled,
        "bilateral_d": pp_config.bilateral_d,
        "bilateral_sigma_color": pp_config.bilateral_sigma_color,
        "bilateral_sigma_space": pp_config.bilateral_sigma_space,
        "clahe_enabled": pp_config.clahe_enabled,
        "clahe_clip_limit": pp_config.clahe_clip_limit,
        "clahe_tile_size": pp_config.clahe_tile_size,
        "nlm_enabled": pp_config.nlm_enabled,
        "nlm_h": pp_config.nlm_h,
        "diffusion_enabled": pp_config.diffusion_enabled,
        "diffusion_iterations": pp_config.diffusion_iterations,
        "diffusion_kappa": pp_config.diffusion_kappa,
    }

    project = {
        "version": PROJECT_VERSION,
        "saved_at": datetime.now().isoformat(),
        "paths": {
            "input_dir": state.input_dir,
            "output_dir": state.output_dir,
        },
        "model": {
            "name": state.model_name,
            "device": state.device,
            "threshold": state.threshold,
            "intermediate_format": state.intermediate_format,
            "mask_output_format": state.mask_output_format,
        },
        "annotation": {
            "points": state.points,
            "labels": state.labels,
        },
        "frames": {
            "start": state.start_frame,
            "end": state.end_frame,
            "marked": sorted(state.marked_frames),
        },
        "preprocessing": pp_dict,
        "overlay": {
            "alpha": state.overlay_alpha,
            "color": list(state.overlay_color),
        },
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2, ensure_ascii=False)
    logger.info(f"Project saved to {filepath}")


def load_project(filepath: str) -> dict:
    """Load project from a .s2proj JSON file.

    Returns:
        Parsed project dict.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        project = json.load(f)

    version = project.get("version", "1.0")
    if "paths" not in project:
        raise ValueError("Invalid project file: missing 'paths' section")

    logger.info(f"Loaded project v{version} from {filepath}")
    return project


def apply_project_to_state(project: dict, state) -> None:
    """Apply loaded project data to AppState.

    Args:
        project: Parsed project dict from load_project().
        state: AppState instance to update.
    """
    paths = project.get("paths", {})
    model = project.get("model", {})
    annotation = project.get("annotation", {})
    frames = project.get("frames", {})
    pp = project.get("preprocessing", {})
    overlay = project.get("overlay", {})

    # Set paths (triggers image discovery)
    if paths.get("input_dir"):
        state.set_input_dir(paths["input_dir"])
    if paths.get("output_dir"):
        state.set_output_dir(paths["output_dir"])

    # Model config
    if model.get("name"):
        state.set_model_name(model["name"])
    if model.get("device"):
        state.set_device(model["device"])
    if "threshold" in model:
        state.set_threshold(model["threshold"])
    if model.get("intermediate_format"):
        state.set_intermediate_format(model["intermediate_format"])
    if model.get("mask_output_format"):
        state.set_mask_output_format(model["mask_output_format"])

    # Annotation
    points = annotation.get("points", [])
    labels = annotation.get("labels", [])
    if points and labels:
        state.set_points(points, labels)

    # Frame range
    if "start" in frames and "end" in frames:
        state.set_frame_range(frames["start"], frames["end"])

    # Marked frames
    for f in frames.get("marked", []):
        state._marked_frames.add(f)
    if state._marked_frames:
        state.marked_frames_changed.emit(set(state._marked_frames))

    # Preprocessing config
    if pp:
        config = PreprocessingConfig(**{
            k: v for k, v in pp.items()
            if k in PreprocessingConfig.__dataclass_fields__
        })
        state.set_preprocessing_config(config)

    # Overlay settings
    if "alpha" in overlay:
        state.set_overlay_alpha(overlay["alpha"])
    if "color" in overlay:
        state.set_overlay_color(tuple(overlay["color"]))
