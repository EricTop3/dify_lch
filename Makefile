# Variables
# Docker 注册表的名称，这里是 langgenius
DOCKER_REGISTRY=langgenius
# Web 镜像的名称，基于 DOCKER_REGISTRY
WEB_IMAGE=$(DOCKER_REGISTRY)/dify-web
# API 镜像的名称，基于 DOCKER_REGISTRY
API_IMAGE=$(DOCKER_REGISTRY)/dify-api
# 镜像的版本，这里设置为 latest
VERSION=latest

# Build Docker images 构建 Web Docker 镜像的目标
# 1 打印构建 Web Docker 镜像的信息
# 2 执行 docker build 命令，将 ./web 目录下的内容构建成 Docker 镜像，并打标签为 $(WEB_IMAGE):$(VERSION)
# 3 打印构建成功的信息
build-web:
	@echo "Building web Docker image: $(WEB_IMAGE):$(VERSION)..."
	docker build -t $(WEB_IMAGE):$(VERSION) ./web
	@echo "Web Docker image built successfully: $(WEB_IMAGE):$(VERSION)"

build-api:
	@echo "Building API Docker image: $(API_IMAGE):$(VERSION)..."
	docker build -t $(API_IMAGE):$(VERSION) ./api
	@echo "API Docker image built successfully: $(API_IMAGE):$(VERSION)"

# Push Docker images 推送 Web Docker 镜像的目标
# 打印推送 Web Docker 镜像的信息。
# 执行 docker push 命令，将 $(WEB_IMAGE):$(VERSION) 镜像推送到注册表。
# 打印推送成功的信息
push-web:
	@echo "Pushing web Docker image: $(WEB_IMAGE):$(VERSION)..."
	docker push $(WEB_IMAGE):$(VERSION)
	@echo "Web Docker image pushed successfully: $(WEB_IMAGE):$(VERSION)"
# 推送 API Docker 镜像的目标
push-api:
	@echo "Pushing API Docker image: $(API_IMAGE):$(VERSION)..."
	docker push $(API_IMAGE):$(VERSION)
	@echo "API Docker image pushed successfully: $(API_IMAGE):$(VERSION)"

# Build all images 构建所有 Docker 镜像的目标，依赖于 build-web 和 build-api 目标
build-all: build-web build-api

# Push all images 推送所有 Docker 镜像的目标，依赖于 push-web 和 push-api 目标
push-all: push-web push-api
# 构建并推送 API Docker 镜像的目标，依赖于 build-api 和 push-api 目标
build-push-api: build-api push-api
# 构建并推送 Web Docker 镜像的目标，依赖于 build-web 和 push-web 目标
build-push-web: build-web push-web

# Build and push all images 构建并推送所有 Docker 镜像的目标，依赖于 build-all 和 push-all 目标。完成后打印所有镜像已构建和推送成功的信息
build-push-all: build-all push-all
	@echo "All Docker images have been built and pushed."

# Phony targets 定义伪目标，确保这些目标总是被执行，即使存在与目标同名的文件
.PHONY: build-web build-api push-web push-api build-all push-all build-push-all
