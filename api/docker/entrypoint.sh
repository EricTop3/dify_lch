#!/bin/bash
# 指示操作系统使用 Bash 解释器执行此脚本
# 设置了 Bash 的 -e 选项，意味着如果脚本中任何命令执行失败（返回非零状态），脚本将立即退出
set -e

# 检查环境变量 MIGRATION_ENABLED 是否设置为 "true"，如果是，执行数据库迁移命令 flask db upgrade，并在执行前打印 “Running migrations”
if [[ "${MIGRATION_ENABLED}" == "true" ]]; then
  echo "Running migrations"
  flask upgrade-db
fi
# 根据环境变量 MODE 的值启动 Celery 如果 MODE 是 "worker"，则启动 Celery worker，并配置相关的选项（如并发方式、日志级别等）
if [[ "${MODE}" == "worker" ]]; then

  # Get the number of available CPU cores
  if [ "${CELERY_AUTO_SCALE,,}" = "true" ]; then
    # Set MAX_WORKERS to the number of available cores if not specified
    AVAILABLE_CORES=$(nproc)
    MAX_WORKERS=${CELERY_MAX_WORKERS:-$AVAILABLE_CORES}
    MIN_WORKERS=${CELERY_MIN_WORKERS:-1}
    CONCURRENCY_OPTION="--autoscale=${MAX_WORKERS},${MIN_WORKERS}"
  else
    CONCURRENCY_OPTION="-c ${CELERY_WORKER_AMOUNT:-1}"
  fi

  exec celery -A app.celery worker -P ${CELERY_WORKER_CLASS:-gevent} $CONCURRENCY_OPTION --loglevel INFO \
    -Q ${CELERY_QUEUES:-dataset,generation,mail,ops_trace,app_deletion}
# 如果 MODE 是 "beat"，则启动 Celery beat 进程
elif [[ "${MODE}" == "beat" ]]; then
  exec celery -A app.celery beat --loglevel INFO
else
  if [[ "${DEBUG}" == "true" ]]; then
    exec flask run --host=${DIFY_BIND_ADDRESS:-0.0.0.0} --port=${DIFY_PORT:-5001} --debug
  else
    exec gunicorn \
      --bind "${DIFY_BIND_ADDRESS:-0.0.0.0}:${DIFY_PORT:-5001}" \
      --workers ${SERVER_WORKER_AMOUNT:-1} \
      --worker-class ${SERVER_WORKER_CLASS:-gevent} \
      --timeout ${GUNICORN_TIMEOUT:-200} \
      --preload \
      app:app
  fi
fi

# 这部分代码决定如何启动 Flask 应用。如果 DEBUG 环境变量被设置为 "true"，则使用 Flask 的开发服务器启动应用。
# 否则，使用 Gunicorn 作为 WSGI HTTP 服务器来运行应用，配置包括绑定的 IP 和端口、工作进程数量、工作进程类型、超时设置等
# 整体上，这个脚本提供了灵活的启动配置，使得根据不同的环境需求（开发、生产、任务执行等）可以灵活地启动相应的服务
