import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _write_synthetic_image(path: Path) -> None:
    image = np.full((24, 24, 3), 255, dtype=np.uint8)
    cv2.circle(image, (12, 12), 7, (40, 180, 220), thickness=-1)
    cv2.imwrite(str(path), image)


def test_cli_writes_json_and_creates_gif(tmp_path):
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "generated"
    _write_synthetic_image(input_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sketch_gen",
            str(input_path),
            str(output_dir),
            "--frames",
            "6",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    output_path = Path(payload["output_path"])
    assert output_path.parent == output_dir
    assert output_path.name.startswith("input-")
    assert output_path.suffix == ".gif"
    assert payload["frames"] == 6
    assert payload["width"] == 24
    assert payload["height"] == 24
    assert output_path.exists()


def test_cli_defaults_output_directory_to_input_directory(tmp_path):
    input_path = tmp_path / "source-photo.png"
    _write_synthetic_image(input_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sketch_gen",
            str(input_path),
            "--frames",
            "6",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    output_path = Path(payload["output_path"])
    assert output_path.parent == tmp_path
    assert output_path.name.startswith("source-photo-")
    assert output_path.suffix == ".gif"
    assert output_path.exists()


def test_cli_rejects_even_blur_kernel(tmp_path):
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "generated"
    _write_synthetic_image(input_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sketch_gen",
            str(input_path),
            str(output_dir),
            "--blur-kernel",
            "4",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "--blur-kernel must be a positive odd integer" in completed.stderr
