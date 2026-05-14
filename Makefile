PYTHON := .venv/bin/python
INPUT ?= asserts/c.png
OUTPUT ?= output

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
