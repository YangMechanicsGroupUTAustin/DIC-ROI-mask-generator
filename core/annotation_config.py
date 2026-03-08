"""Save and load annotation configurations (points, labels, parameters)."""

import json
from datetime import datetime


def save_annotation_config(
    filepath,
    input_points,
    input_labels,
    model_name="",
    device="",
    threshold=0.0,
    start_frame=1,
    end_frame=0,
    intermediate_format="JPEG",
):
    """Save annotation points and experiment config to JSON file."""
    config = {
        "version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "annotation": {
            "points": input_points,
            "labels": input_labels,
        },
        "parameters": {
            "model": model_name,
            "device": device,
            "threshold": threshold,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "intermediate_format": intermediate_format,
        },
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def load_annotation_config(filepath):
    """Load annotation config from JSON file. Returns parsed dict."""
    with open(filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if "annotation" not in config:
        raise ValueError("Invalid config file: missing 'annotation' section")
    if "points" not in config["annotation"] or "labels" not in config["annotation"]:
        raise ValueError("Invalid config file: missing points or labels")

    return config
