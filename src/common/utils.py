"""
utils.py

Shared utility functions used across EDA, modeling, and tuning pipelines.

Key responsibilities:
- Generate consistent timestamps
- Create temporary local artifact directories
- Keep artifact handling MLflow-friendly
"""

from datetime import datetime
from pathlib import Path
import shutil
import uuid


def generate_timestamp() -> str:
    """
    Generate a human-readable timestamp for run identification.

    Returns
    -------
    str
        Timestamp in YYYY-MM-DD_HH-MM-SS format
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_temp_artifact_dir(
    base_dir: str,
    prefix: str
) -> Path:
    """
    Create a temporary local directory for artifacts
    that will later be logged to MLflow.

    Structure:
    artifacts/<base_dir>/<prefix>_<timestamp>_<uuid>/

    Parameters
    ----------
    base_dir : str
        High-level category (eda, baseline, tuning)
    prefix : str
        Logical name for the run

    Returns
    -------
    Path
        Path to the created artifact directory
    """

    timestamp = generate_timestamp()
    run_id = uuid.uuid4().hex[:8]

    artifact_dir = Path("artifacts") / base_dir / f"{prefix}_{timestamp}_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    return artifact_dir


def cleanup_artifact_dir(artifact_dir: Path) -> None:
    """
    Delete local artifact directory after successful MLflow logging.

    Parameters
    ----------
    artifact_dir : Path
        Local artifact directory to remove
    """
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
