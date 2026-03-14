# syntax=docker/dockerfile:1

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    vulkan-tools \
    libvulkan-dev \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies for Omnilingual + pyannote
# Note: GPU acceleration depends on host /dev/dri passthrough and Vulkan ICD availability.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch pyannote.audio fairseq2 omnilingual-asr

FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    ca-certificates \
    vulkan-tools \
    libvulkan1 \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Vulkan ICD path (AMD), used by tooling like vulkaninfo
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json

RUN if getent passwd 1000 >/dev/null; then useradd -m tenepal; else useradd -m -u 1000 tenepal; fi
USER tenepal
WORKDIR /app

COPY --chown=tenepal:tenepal healthcheck.py /app/healthcheck.py
COPY --chown=tenepal:tenepal diarize_worker.py /app/diarize_worker.py

VOLUME ["/models", "/audio"]

HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD ["python3.12", "/app/healthcheck.py"]
