# GR00T dual-arm UR5e inference client image (arm64 / Jetson L4T host).
#
# ur_rtde ships NO aarch64 wheel, so it is compiled from source here (needs
# Boost headers + CMake + a C++ toolchain). opencv-python-headless, numpy,
# pyzmq, msgpack and requests all have aarch64 wheels and install fast.
#
# The container defaults to DRY RUN (no robot motion). The Deployment passes
# connection args; add --execute --assume-yes to actually command the arms.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Build deps for ur_rtde (Boost + CMake + g++) and the one runtime shared lib
# opencv-python-headless needs (libglib2.0). Kept in the image because ur_rtde's
# compiled extension links Boost at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libboost-all-dev \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY robot_inference_client.py .

# Connection/safety args are supplied by the Deployment. Defaults to dry run.
ENTRYPOINT ["python3", "robot_inference_client.py"]
