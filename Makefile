# FlowForge — one-command reproduction (author: 晨星)
PY ?= python
PKG := flowforge

.PHONY: install test lint fmt demo selftest clean ci

install:
	$(PY) -m pip install -r requirements.lock.txt

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

test:
	$(PY) -m pytest -q

cov:
	$(PY) -m pytest -q --cov=$(PKG) --cov-report=term

selftest:
	$(PY) -m $(PKG).cli selftest

demo:
	$(PY) $(PKG)/examples/run_demo.py

demo-quick:
	$(PY) $(PKG)/examples/run_demo.py --quick

ci: lint test selftest demo-quick

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ */__pycache__ *.egg-info .coverage
