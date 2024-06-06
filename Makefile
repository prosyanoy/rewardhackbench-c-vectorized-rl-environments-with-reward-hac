VENV    ?= .venv
PYTHON  ?= $(VENV)/bin/python
THREADS ?= $(or $(RHB_THREADS),2)
export PYTHONPATH := python
export PATH := $(abspath $(VENV))/bin:$(PATH)

.PHONY: setup build test bench demo clean

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(VENV)/bin/pip install -r requirements.txt

build:
	cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(abspath $(PYTHON))
	cmake --build build

test: build
	$(PYTHON) -m pytest -q

bench: build
	$(PYTHON) -m rewardhackbench.benchmark

# One command: train sparkle/shaping/aligned, run the monitor, write docs/*.png + exploit.gif.
demo: build
	for v in sparkle shaping aligned; do \
	  $(PYTHON) -W ignore -m rewardhackbench.train --variant $$v --threads $(THREADS) --out runs/$$v && \
	  $(PYTHON) -W ignore -m rewardhackbench.report --run runs/$$v --plot; \
	done
	$(PYTHON) -W ignore -m rewardhackbench.assets --runs runs --out docs

clean:
	rm -rf build runs python/rewardhackbench/*.so
