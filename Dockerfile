# FlowForge · 作者：晨星
# 纯 CPU、零权重下载、零密钥依赖的最小可复现镜像。
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 依赖层单独缓存
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt pytest ruff==0.16.10

# 代码层
COPY pyproject.toml README.md LICENSE ./
COPY flowforge ./flowforge
COPY tests ./tests
COPY examples ./examples

# 安装为包，使 `flowforge` CLI 可用
RUN pip install --no-cache-dir -e .

# 门禁：lint 是硬门禁（不是 `|| true`）
RUN python -m ruff check . \
 && python -m ruff format --check . \
 && python -m pytest -q -W ignore::UserWarning

# 默认动作：端到端演示
CMD ["python", "examples/run_demo.py"]
