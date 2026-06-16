# GR00T N1.5 inference server image for Jetson Orin
# (JetPack 6.2 / L4T r36.4 / Python 3.10 / CUDA 12.6).
#
# Mirrors NVIDIA's OFFICIAL Isaac-GR00T *N1.5* Orin build (orin.Dockerfile +
# deployment_scripts/setup_env.sh on the n1.5-release tag): a system-python
# install of `gr00t[orin]` from the jetson-ai-lab precompiled wheel index, cuDSS,
# and a from-source decord/ffmpeg. Pinned to N1.5 on purpose: the deployed
# checkpoint declares model_type `gr00t_n1_5`, which the newer N1.6/N1.7 code on
# Isaac-GR00T main no longer registers, so a main-based image cannot load it.
#
# The runtime entrypoint is GR00T N1.5's ZMQ inference service
# (scripts/inference_service.py --server). N1.5 needs --data-config to build the
# modality transform (it is not stored in the checkpoint); the Helm chart supplies it.
#
# Model weights are NOT baked in — they are mounted at runtime (/models/gr00t)
# from the ACR model artifact via the chart's modctl init container. This image
# is only the runtime that loads those weights.
#
# Override the base at build time with --build-arg BASE_IMAGE=... if needed.
#
# The push registry is parameterized in build_and_push.sh (REGISTRY env, default
# immitationlearning.azurecr.io); this runtime image pulls only from BASE_IMAGE.
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

# Pin the Isaac-GR00T revision. Default to the N1.5 release tag, which is the
# version that registers the gr00t_n1_5 architecture this checkpoint needs.
ARG GR00T_REF=n1.5-release

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1

# System dependencies (matches the official N1.5 orin.Dockerfile), plus
# git/git-lfs/curl/wget for cloning the pinned revision with its LFS payloads.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-setuptools \
        libsm6 libxext6 ffmpeg \
        libhdf5-serial-dev libtesseract-dev libgtk-3-0 \
        libtbb12 libtbb2 libatlas-base-dev libopenblas-dev \
        build-essential make cmake nasm \
        git git-lfs curl wget ca-certificates \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

# Install cuDSS (the CUDA direct sparse solver library torch needs on Tegra).
# Same package/version as NVIDIA's official N1.5 orin.Dockerfile.
RUN wget -q https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb \
    && dpkg -i cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb \
    && cp /var/cudss-local-tegra-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/ \
    && chmod 777 /tmp \
    && apt-get update \
    && apt-get -y install cudss \
    && rm -f cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Clone the validated revision WITH git-lfs payloads.
RUN git clone --depth 1 --branch ${GR00T_REF} https://github.com/NVIDIA/Isaac-GR00T.git . \
    && git lfs pull

# Install build tooling + torch first (pytorch3d compiles against an installed
# torch). torch/torchvision come from the jetson-ai-lab precompiled wheel index.
RUN PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126 \
    PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io \
    pip3 install --upgrade pip setuptools wheel ninja \
    && PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126 \
       PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io \
       pip3 install torch==2.8.0 torchvision==0.23.0

# gr00t[orin] pins pytorch3d==0.7.8, which has NO installable wheel for aarch64
# (PyPI stops at 0.7.4; the jetson index only proxies old x86 wheels). Build it
# from the official GitHub source tag with CUDA ops enabled, so the pin below is
# already satisfied and pip doesn't try (and fail) to fetch it from an index.
#
# TORCH_CUDA_ARCH_LIST must be set explicitly: no GPU is visible during `docker
# build`, so pytorch3d's arch auto-detection returns an empty list and crashes
# with IndexError. 8.7 is the Jetson AGX Orin compute capability.
RUN FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.7" \
    pip3 install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

# Install gr00t and its remaining Orin extras from the jetson-ai-lab precompiled
# wheel index (flash-attn, triton, tensorflow, etc.); torch + pytorch3d are
# already satisfied above. Matches NVIDIA's official N1.5 setup_env.sh.
RUN PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126 \
    PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io \
    pip3 install -e .[orin]

# Build and install decord (video decoding) from source against ffmpeg n4.4.2,
# exactly as NVIDIA's official N1.5 Orin build does.
#
# decord's cmake build produces build/libdecord.so, but `setup.py install` does
# NOT bundle that native lib into the installed Python package. At runtime
# decord._ffi searches a fixed candidate list (and LD_LIBRARY_PATH) for
# libdecord.so. Copy the built lib into both /root/.local/decord/ (a candidate
# path, also on LD_LIBRARY_PATH below) and the installed package dir BEFORE the
# source tree is deleted, otherwise `import decord` fails with
# "RuntimeError: Cannot find the files."
RUN git clone https://git.ffmpeg.org/ffmpeg.git \
    && cd ffmpeg \
    && git checkout n4.4.2 \
    && ./configure --enable-shared --enable-pic --prefix=/usr \
    && make -j"$(nproc)" \
    && make install \
    && cd .. \
    && git clone --recursive https://github.com/dmlc/decord \
    && cd decord \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j"$(nproc)" \
    && cd ../python \
    && python3 setup.py install \
    && mkdir -p /root/.local/decord \
    && cp /workspace/decord/build/libdecord.so /root/.local/decord/libdecord.so \
    && cp /workspace/decord/build/libdecord.so /usr/local/lib/libdecord.so \
    && DECORD_PKG_DIR="$(python3 -c 'import os,decord; print(os.path.dirname(decord.__file__))' 2>/dev/null || true)" \
    && if [ -n "$DECORD_PKG_DIR" ]; then cp /workspace/decord/build/libdecord.so "$DECORD_PKG_DIR/libdecord.so"; fi \
    && ldconfig \
    && cd /workspace \
    && rm -rf /workspace/ffmpeg /workspace/decord

# Expose CUDA libs and decord's installed shared lib so imports resolve at runtime.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:/root/.local/decord/:${LD_LIBRARY_PATH}

# ZMQ REQ/REP inference port exposed by the N1.5 RobotInferenceServer.
EXPOSE 5555

# GR00T N1.5 ZMQ inference service. The Helm chart supplies --server,
# --model-path, --embodiment-tag, --data-config, --host, --port and
# --denoising-steps as args.
ENTRYPOINT ["python3", "scripts/inference_service.py"]
CMD ["--help"]
