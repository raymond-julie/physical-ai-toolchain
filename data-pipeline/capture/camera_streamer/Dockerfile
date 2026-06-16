# camera_streamer container image.
#
# pyorbbecsdk is a from-source build pinned to this platform (aarch64,
# CPython 3.10, Ubuntu 22.04 glibc), so the image base MUST match. Rather than
# rebuild the SDK in-container, the prebuilt artifacts are vendored into the
# build context (see docker/stage-orbbec.sh) and copied into site-packages.
#
# The GMSL kernel modules (g300 / max9296 / max96712) live on the HOST kernel
# and are loaded there (reconnect.sh). The container only needs the resulting
# /dev/video* device nodes passed through at run time.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Runtime deps for the Orbbec SDK (libusb/libudev) + Python + OpenCV runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libusb-1.0-0 \
        libudev1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (pyorbbecsdk is added separately from vendored artifacts).
COPY requirements.txt ./
RUN pip3 install --no-cache-dir \
        flask>=2.0 \
        opencv-python-headless>=4.5 \
        numpy>=1.21 \
        PyYAML>=5.4

# Vendored, prebuilt Orbbec SDK (staged by docker/stage-orbbec.sh).
# Lands in the system site-packages so `import pyorbbecsdk` works for any user.
ARG SITE_PACKAGES=/usr/lib/python3/dist-packages
COPY docker/vendor/orbbec/ ${SITE_PACKAGES}/
# The SDK .so files resolve their siblings + extensions/ via RPATH=$ORIGIN, so
# placing them all in site-packages is sufficient — no LD_LIBRARY_PATH needed.

# Application code.
COPY camera_streamer/ ./camera_streamer/

EXPOSE 8000

# Host networking is recommended (see docker-compose.yml) so the streams are
# reachable on the LAN and the printed share URLs use the host's real IP.
ENTRYPOINT ["python3", "-m", "camera_streamer"]
CMD ["--port", "8000"]
