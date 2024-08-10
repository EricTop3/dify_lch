# base image 基础镜像阶段
# 设置基础镜像为 Python 3.10 版本的 slim 版本，基于 Debian bookworm。这种镜像较小，适合生产环境使用。并将此阶段命名为 base
FROM python:3.10-slim-bookworm AS base

WORKDIR /app/api

# Install Poetry  安装依赖包阶段
ENV POETRY_VERSION=1.8.3
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Configure Poetry
ENV POETRY_CACHE_DIR=/tmp/poetry_cache
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_VIRTUALENVS_CREATE=true

# 从基础镜像 base 开始新的构建阶段，命名为 packages
FROM base AS packages
# 安装 Python 应用构建和运行时可能需要的系统依赖包，包括编译器和库文件
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libc-dev libffi-dev libgmp-dev libmpfr-dev libmpc-dev

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --sync --no-cache --no-root

# production stage  生产阶段设置
# 从基础镜像 base 开始新的构建阶段，命名为 production
FROM base AS production
# 设置 Flask 环境变量和其他运行时环境变量
ENV FLASK_APP=app.py
ENV EDITION=SELF_HOSTED
ENV DEPLOY_ENV=PRODUCTION
ENV CONSOLE_API_URL=http://127.0.0.1:5001
ENV CONSOLE_WEB_URL=http://127.0.0.1:3000
ENV SERVICE_API_URL=http://127.0.0.1:5001
ENV APP_WEB_URL=http://127.0.0.1:3000
# 声明容器将在端口 5001 上运行服务
EXPOSE 5001

# set timezone
ENV TZ=UTC
# 设置工作目录
WORKDIR /app/api
# 更新包索引并 安装 生产环境 所需的其他系统依赖，最后 清理 不再需要的包和临时文件
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl nodejs libgmp-dev libmpfr-dev libmpc-dev \
    && echo "deb http://deb.debian.org/debian testing main" > /etc/apt/sources.list \
    && apt-get update \
    # For Security
    && apt-get install -y --no-install-recommends zlib1g=1:1.3.dfsg+really1.3.1-1 expat=2.6.2-1 libldap-2.5-0=2.5.18+dfsg-2 perl=5.38.2-5 libsqlite3-0=3.46.0-1 \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment and packages
ENV VIRTUAL_ENV=/app/api/.venv
COPY --from=packages ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Copy source code 复制当前目录下所有文件到工作目录
COPY . /app/api/

# Copy entrypoint 复制入口脚本到容器
COPY docker/entrypoint.sh /entrypoint.sh
# 赋予入口脚本执行权限
RUN chmod +x /entrypoint.sh

# 接收并设置构建参数，用于标记版本
ARG COMMIT_SHA
ENV COMMIT_SHA=${COMMIT_SHA}
# 设置容器启动时执行的命令
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
