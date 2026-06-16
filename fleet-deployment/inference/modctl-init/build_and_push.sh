#!/usr/bin/env bash
# Build and push the modctl init image to a container registry.
#
# This image bakes the modctl CLI; the gr00t-inference chart uses it as the
# weight-fetch init container to pull CNCF ModelPack artifacts that `oras pull`
# cannot materialize. Build on the arm64/L4T host so the image arch matches the
# Jetson cluster.
#
# The registry defaults to the example ACR but is overridable so the image can
# target any registry:
#   REGISTRY=myregistry.azurecr.io ./build_and_push.sh 0.1
#
# Usage: ./build_and_push.sh <tag> [modctl-version] [base-image]
# Example (Jetson Orin, native aarch64 build — default base):
#   ./build_and_push.sh 0.1
set -o errexit -o nounset -o pipefail

REGISTRY="${REGISTRY:-immitationlearning.azurecr.io}"
IMAGE="${IMAGE:-modctl}"
TAG="${1:?Usage: build_and_push.sh <tag> [modctl-version] [base-image]}"
MODCTL_VERSION="${2:-0.2.1-cnai}"
BASE_IMAGE="${3:-debian:bookworm-slim}"
REF="${REGISTRY}/${IMAGE}:${TAG}"

echo "Building ${REF}"
echo "  MODCTL_VERSION=${MODCTL_VERSION}"
echo "  BASE_IMAGE=${BASE_IMAGE}"
# --network=host avoids BuildKit's bridge setup, which fails on hosts whose
# kernel lacks the iptable_raw module ("can't initialize iptables table 'raw'").
docker build \
  --network=host \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "MODCTL_VERSION=${MODCTL_VERSION}" \
  -t "${REF}" \
  "$(dirname "$0")"

echo "Pushing ${REF}"
docker push "${REF}"

echo "Done. Set the chart to use it:"
echo "  --set modctl.image.repository=${REGISTRY}/${IMAGE} --set modctl.image.tag=${TAG}"
