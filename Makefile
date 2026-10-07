# FlowForge · 作者：晨星
PY ?= python

.PHONY: help install dev test lint fmt demo check clean

help:
	@echo "make install  安装运行依赖"
	@echo "make dev      安装开发依赖（pytest/ruff）"
	@echo "make test     运行单元测试"
	@echo "make lint     ruff check + format --check（硬门禁）"
	@echo "make fmt      ruff format"
	@echo "make demo     端到端演示，产出 benchmark.json"
	@echo "make check    自检全部数学不变量（I1~I5）"
	@echo "make clean    清理生成物"

install:
	$(PY) -m pip install -r requirements.txt

dev:
	$(PY) -m pip install -r requirements.txt pytest-cov ruff==0.16.10

test:
	$(PY) -m pytest -q -W ignore::UserWarning --cov=flowforge --cov-report=term

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:
	$(PY) -m ruff format .

demo:
	$(PY) examples/run_demo.py

check:
	$(PY) -m flowforge.cli --check

clean:
	rm -f benchmark.json bench_before.json
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
