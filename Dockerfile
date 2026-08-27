# =============================================================================
# ECO-Gradian IoT Edge — Multi-stage Docker Build
# =============================================================================
# Target: Headless Linux IoT devices (Orange Pi, RPi, generic ARM64/AMD64)
# RAM Limit: 1 GB enforced at container + application level
#
# Build:
#   docker build -t eco-iot:latest .
#   docker build --platform linux/arm64 -t eco-iot:arm64 .   # Cross-compile for Orange Pi
#
# Run:
#   docker run -d --name eco_iot \
#     --memory=1g --memory-swap=1g \
#     --device=/dev/video0 \
#     -p 8080:8080 \
#     -v eco_iot_data:/app/data \
#     -v eco_iot_logs:/app/logs \
#     eco-iot:latest
# =============================================================================

# ── Stage 1: Build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

# Avoid interactive prompts, reduce image layer noise
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System deps for OpenCV headless (no X11/GTK) + V4L2
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        gcc \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libgl1-mesa-glx \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install --no-warn-script-location -r requirements.txt


# ── Stage 2: Runtime (minimal) ─────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="ECO-Gradian Team" \
      description="AI Edge Waste Classifier for IoT Smart Bins" \
      version="2.0.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Disable TF/ONNX GPU probing — pure CPU on IoT
    CUDA_VISIBLE_DEVICES="" \
    TF_CPP_MIN_LOG_LEVEL=3 \
    # ECO-Gradian config via env
    ECO_IOT_PORT=8080 \
    ECO_IOT_CAMERA=0 \
    ECO_UPSTREAM_URL="http://host.docker.internal:8000" \
    # Memory tuning: force Python to release pages aggressively
    MALLOC_TRIM_THRESHOLD_=65536 \
    MALLOC_MMAP_THRESHOLD_=131072 \
    PYTHONMALLOC=malloc

WORKDIR /app

# Minimal runtime packages — no GUI, no X11
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libsm6 \
        libxrender1 \
        v4l-utils \
        procps \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY main.py config.py ai_engine.py camera.py key_manager.py memory_guard.py ./
COPY models/ ./models/

# Create data & log directories (will be overridden by volumes)
RUN mkdir -p /app/data /app/logs /app/models \
    && chmod 755 /app/data /app/logs

# Non-root user for security (but needs /dev/video access via --group-add video)
RUN groupadd -r ecoiot && useradd -r -g ecoiot -d /app ecoiot \
    && chown -R ecoiot:ecoiot /app
USER ecoiot

# Persistent storage volumes
VOLUME ["/app/data", "/app/logs", "/app/models"]

# Health check — lightweight RAM + camera status probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

EXPOSE 8080

# Entry point with crash-resilient defaults
ENTRYPOINT ["python", "main.py"]
CMD ["--port", "8080", "--camera", "0"]
