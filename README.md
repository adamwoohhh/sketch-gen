# sketch-gen

`sketch-gen` is a Python 3 command-line tool that turns an input image into a GIF where a line sketch is gradually drawn and then filled with the original colors.

## Install

```bash
make setup
```

## Usage

```bash
make run INPUT=input.png OUTPUT=output
```

`OUTPUT` is an output directory, not the final GIF filename. The generated file
is named `{original-file-name}-{timestamp}.gif`. If you do not pass an output
directory, the CLI writes the GIF next to the input image.

Useful options:

```bash
.venv/bin/python -m sketch_gen input.png output \
  --frames 24 \
  --duration 80 \
  --blur-kernel 5 \
  --canny-low 80 \
  --canny-high 160 \
  --color-frames-ratio 0.65 \
  --loop-once \
  --log
```

`--loop-once` writes GIF loop metadata for a single loop. Without it, the GIF
uses infinite looping.

`--log` writes a log file under `./logs`. stdout remains JSON-only, so NodeJS
callers can still parse the command result directly.

Run tests:

```bash
make test
```

On success, the CLI prints JSON to stdout. Errors are printed to stderr and return a non-zero exit code, which keeps the command easy to wrap from a future NodeJS npm package.

`make` is only a shortcut layer. Internally, it still creates a local `.venv` and runs `.venv/bin/python`, so project dependencies stay isolated from your system Python.
