PYTHON := .venv/bin/python
INPUT ?= input.png
OUTPUT ?= output.gif

.PHONY: setup test run clean

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -v

run:
	$(PYTHON) -m sketch_gen $(INPUT) $(OUTPUT)

clean:
	rm -rf .pytest_cache sketch_gen.egg-info
