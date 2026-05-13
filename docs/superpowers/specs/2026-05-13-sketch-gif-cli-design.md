# Sketch GIF CLI Design

## Goal

Build a Python 3 command-line tool that accepts an input image, applies Gaussian blur and Canny edge detection with OpenCV, then renders a GIF that looks like a hand-drawn line sketch gradually completing and finally filling with the original image colors.

The first release is a CLI tool. The implementation must keep a stable process boundary so a future npm package can call it from a NodeJS service through `child_process.spawn()` without rewriting the image pipeline.

## Command Interface

Primary command:

```bash
python -m sketch_gen input.png output.gif
```

Optional parameters:

- `--frames`: total GIF frame count.
- `--duration`: per-frame duration in milliseconds.
- `--blur-kernel`: odd integer kernel size for `cv2.GaussianBlur`.
- `--canny-low`: lower threshold for `cv2.Canny`.
- `--canny-high`: upper threshold for `cv2.Canny`.
- `--color-frames-ratio`: fraction of frames used for the color-fill phase.

The CLI writes a single JSON object to stdout on success, including the output path, frame count, and image dimensions. Human-readable errors go to stderr. Exit code `0` means success; non-zero means failure.

## Image Pipeline

1. Load the input image with OpenCV.
2. Convert to grayscale.
3. Apply `cv2.GaussianBlur`.
4. Run `cv2.Canny` to create an edge mask.
5. Convert the edge mask into a dark line layer on a white canvas.
6. Generate line-progress frames by revealing the line pixels in a deterministic order.
7. Generate color-fill frames by blending from the completed line sketch to the original image color while preserving visible sketch lines.
8. Write the frames as a GIF.

The first version uses deterministic top-to-bottom reveal ordering. This is simple, reproducible, and easy to test. More natural stroke ordering can be added later without changing the CLI contract.

## Package Structure

- `sketch_gen/renderer.py`: pure image-processing and frame-generation functions.
- `sketch_gen/cli.py`: argument parsing, validation, JSON stdout, stderr errors, and exit codes.
- `sketch_gen/__main__.py`: enables `python -m sketch_gen`.
- `tests/`: focused pytest tests for rendering and CLI behavior.
- `pyproject.toml`: project metadata and dependencies.
- `README.md`: installation and usage.

## Code Readability

The user does not have a Python development background, so the implementation should include clear comments around important steps. Comments should explain why each major step exists, especially:

- Image loading and OpenCV color channel handling.
- Grayscale conversion, Gaussian blur, and Canny edge detection.
- How the line reveal frames are generated.
- How color-fill frames blend the sketch with the original image.
- CLI validation and JSON output for future NodeJS integration.

Comments should be educational but still tied to the code. Avoid comments that only restate a variable name or a single obvious assignment.

## NodeJS Compatibility

The Python CLI is the stable integration contract. A future npm package can:

1. Bundle or require installation of the Python package.
2. Spawn `python -m sketch_gen`.
3. Pass file paths and numeric options.
4. Parse stdout JSON.
5. Treat non-zero exit codes and stderr as failures.

The CLI should avoid interactive prompts, progress bars, localized output, or mixed stdout logs because those make NodeJS process wrapping brittle.

## Validation And Errors

The CLI validates that:

- The input path exists and can be decoded as an image.
- The output path has a `.gif` suffix.
- `--blur-kernel` is a positive odd integer.
- `--frames` is at least `2`.
- Canny thresholds are non-negative and `canny-low <= canny-high`.
- `--color-frames-ratio` is greater than `0` and less than `1`.

## Testing

Tests should cover:

- Rendering creates a GIF at the requested path.
- Generated GIF contains the requested number of frames.
- CLI success returns parseable JSON on stdout.
- Invalid CLI parameters fail with a non-zero exit code and stderr message.

Tests use small synthetic images so they do not depend on external assets.
