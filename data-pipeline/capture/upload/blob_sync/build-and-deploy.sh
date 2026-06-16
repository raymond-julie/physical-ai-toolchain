#!/usr/bin/env bash
# Build the blob_sync Docker image and push it to Azure Container Registry.
#
# Usage:
#   ACR_USERNAME=devToken ACR_PASSWORD='<token>' ./build-and-deploy.sh
#
#   # or log in to ACR yourself beforehand and skip the login step:
#   az acr login --name <registry>
#   SKIP_LOGIN=1 ./build-and-deploy.sh
#
# Optional overrides:
#   REGISTRY   (default: immitationlearning.azurecr.io)
#   IMAGE      (default: blob_sync)
#   TAG        (default: current git short SHA, or "latest")
set -euo pipefail

cd "$(dirname "$0")"

REGISTRY="${REGISTRY:-immitationlearning.azurecr.io}"
IMAGE="${IMAGE:-blob_sync}"
if [[ -z "${TAG:-}" ]]; then
  TAG="$(git rev-parse --short HEAD 2>/dev/null || echo latest)"
fi

IMAGE_REF="${REGISTRY}/${IMAGE}:${TAG}"
LATEST_REF="${REGISTRY}/${IMAGE}:latest"

echo ">> Building ${IMAGE_REF}"
# --network=host avoids creating a Docker bridge endpoint, which fails on
# kernels lacking the iptables 'raw' table (e.g. NVIDIA Jetson/Tegra).
docker build --network="${BUILD_NETWORK:-host}" -t "${IMAGE_REF}" -t "${LATEST_REF}" .

if [[ "${SKIP_LOGIN:-0}" != "1" ]]; then
  if [[ -z "${ACR_USERNAME:-}" || -z "${ACR_PASSWORD:-}" ]]; then
    echo "ERROR: set ACR_USERNAME and ACR_PASSWORD env vars, or run with SKIP_LOGIN=1 after 'az acr login'." >&2
    exit 1
  fi
  echo ">> Logging in to ${REGISTRY}"
  # Password is piped via stdin so it never appears in the process list.
  printf '%s' "${ACR_PASSWORD}" | docker login "${REGISTRY}" -u "${ACR_USERNAME}" --password-stdin
fi

echo ">> Pushing ${IMAGE_REF}"
docker push "${IMAGE_REF}"
echo ">> Pushing ${LATEST_REF}"
docker push "${LATEST_REF}"

echo ">> Done: ${IMAGE_REF}"
