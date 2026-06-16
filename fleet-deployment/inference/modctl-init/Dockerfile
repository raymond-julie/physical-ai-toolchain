# modctl init image: bakes the modctl CLI onto a small Debian base so the
# gr00t-inference chart's weight-fetch init container can pull CNCF ModelPack
# (modctl-built) artifacts from ACR.
#
# The gr00t model weights in ACR (gr00t-n15-teradyne-dual-arm) are a CNCF
# ModelPack artifact (manifest annotation org.cncf.modctl.modelfile; layer
# media types application/vnd.cncf.model.weight.v1.raw with org.cncf.model.filepath).
# `oras pull` skips those layers (it only materializes org.opencontainers.image.title
# layers), so weights never land on disk. modctl understands the format and
# extracts the files directly with:
#   modctl pull <ref> --extract-dir <dir> --extract-from-remote
#
# Auth: modctl loads registry credentials via the standard Docker config.json
# (credentials.NewStoreFromDocker, honoring $DOCKER_CONFIG), so the chart's
# existing projected ACR pull secret works unchanged.
ARG BASE_IMAGE=debian:bookworm-slim
FROM ${BASE_IMAGE}

# Pin the modctl release that understands the org.cncf.* ModelPack media types
# used by the artifact. linux-arm64 binary for Jetson Orin (aarch64).
ARG MODCTL_VERSION=0.2.1-cnai
ARG TARGETARCH=arm64

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    curl -fsSL -o /tmp/modctl.tar.gz \
        "https://github.com/modelpack/modctl/releases/download/v${MODCTL_VERSION}/modctl-${MODCTL_VERSION}-linux-${TARGETARCH}.tar.gz"; \
    mkdir -p /tmp/modctl; \
    tar -xzf /tmp/modctl.tar.gz -C /tmp/modctl; \
    install -m 0755 "$(find /tmp/modctl -type f -name modctl | head -n1)" /usr/local/bin/modctl; \
    rm -rf /tmp/modctl /tmp/modctl.tar.gz; \
    apt-get purge -y curl; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*; \
    modctl version

ENTRYPOINT ["modctl"]
CMD ["--help"]
