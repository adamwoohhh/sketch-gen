# Sketch GIF CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python 3 CLI that converts an input image into a GIF where a Canny line sketch appears gradually and then fills with the original colors.

**Architecture:** Keep reusable image logic in `sketch_gen/renderer.py` and process-boundary behavior in `sketch_gen/cli.py`. The CLI prints JSON only on stdout for success and writes errors to stderr so a future npm wrapper can call it reliably.

**Tech Stack:** Python 3.10+, OpenCV (`opencv-python`), Pillow, NumPy, pytest.

---

## File Structure

- `pyproject.toml`: package metadata, dependencies, pytest config.
- `README.md`: setup and CLI examples.
- `sketch_gen/__init__.py`: package version.
- `sketch_gen/__main__.py`: module entry point for `python -m sketch_gen`.
- `sketch_gen/cli.py`: argument parsing, validation, JSON stdout, stderr errors.
- `sketch_gen/renderer.py`: image loading, OpenCV processing, frame generation, GIF writing.
- `tests/test_renderer.py`: core renderer tests using synthetic images.
- `tests/test_cli.py`: subprocess tests for CLI success and validation failures.

## Tasks

### Task 1: Project Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `sketch_gen/__init__.py`
- Create: `sketch_gen/__main__.py`

- [ ] Create package metadata with dependencies: `opencv-python`, `Pillow`, `numpy`, `pytest`.
- [ ] Add `python -m sketch_gen` entry through `__main__.py`.
- [ ] Add README with install and CLI usage.
- [ ] Commit with message `chore: scaffold Python CLI project`.

### Task 2: Renderer Red-Green Cycle

**Files:**
- Create: `tests/test_renderer.py`
- Create: `sketch_gen/renderer.py`

- [ ] Write a failing test that creates a small synthetic PNG, calls `render_sketch_gif(...)`, and asserts the GIF exists with the requested frame count.
- [ ] Run `python -m pytest tests/test_renderer.py -v` and verify it fails because `sketch_gen.renderer` does not exist yet.
- [ ] Implement `RenderOptions`, `RenderResult`, validation helpers, and `render_sketch_gif(...)`.
- [ ] Add clear comments explaining OpenCV BGR/RGB handling, grayscale conversion, Gaussian blur, Canny, line reveal, and color fill.
- [ ] Run `python -m pytest tests/test_renderer.py -v` and verify it passes.
- [ ] Commit with message `feat: add sketch GIF renderer`.

### Task 3: CLI Red-Green Cycle

**Files:**
- Create: `tests/test_cli.py`
- Modify: `sketch_gen/cli.py`
- Modify: `sketch_gen/__main__.py`

- [ ] Write a failing subprocess test for `python -m sketch_gen input.png output.gif --frames 6`, checking exit code `0`, parseable stdout JSON, and GIF creation.
- [ ] Write a failing subprocess test for invalid `--blur-kernel 4`, checking non-zero exit and stderr.
- [ ] Run `python -m pytest tests/test_cli.py -v` and verify it fails because CLI code is missing.
- [ ] Implement `argparse` CLI with stable stdout JSON and stderr error handling.
- [ ] Run `python -m pytest tests/test_cli.py -v` and verify it passes.
- [ ] Commit with message `feat: add sketch GIF CLI`.

### Task 4: Full Verification

**Files:**
- Modify: `README.md` if command details changed during implementation.

- [ ] Run `python -m pytest -v`.
- [ ] Run the CLI against the sample image in `asserts/` and write a GIF under `/tmp`.
- [ ] Confirm stdout is JSON and the GIF file exists.
- [ ] Commit any README corrections with message `docs: update CLI usage`.
