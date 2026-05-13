# sketch-gen

`sketch-gen` is a Python 3 command-line tool that turns an input image into a GIF where a line sketch is gradually drawn and then filled with the original colors.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

## Usage

```bash
python3 -m sketch_gen input.png output.gif
```

Useful options:

```bash
python3 -m sketch_gen input.png output.gif \
  --frames 24 \
  --duration 80 \
  --blur-kernel 5 \
  --canny-low 80 \
  --canny-high 160 \
  --color-frames-ratio 0.35
```

On success, the CLI prints JSON to stdout. Errors are printed to stderr and return a non-zero exit code, which keeps the command easy to wrap from a future NodeJS npm package.
