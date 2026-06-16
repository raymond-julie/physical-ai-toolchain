#!/usr/bin/env bash
# Build and push the camera_streamer image to ACR.
#
# pyorbbecsdk is vendored from the host's prebuilt install (not on PyPI), so
# this stages those artifacts into the build context first, then builds and
# pushes. Run on the Jetson/Tegra host where the SDK is already built.
#
# Usage: ./build_and_push.sh <tag>
# Example: ./build_and_push.sh 0.1
set -euo pipefail

REGISTRY="${REGISTRY:-immitationlearning.azurecr.io}"
IMAGE="${IMAGE:-ur-camera-streamer}"
TAG="${1:?Usage: build_and_push.sh <tag>}"
REF="${REGISTRY}/${IMAGE}:${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Staging vendored Orbbec SDK into the build context"
"${SCRIPT_DIR}/docker/stage-orbbec.sh"

echo "Building ${REF}"
# --network=host avoids BuildKit's bridge setup, which fails on hosts whose
# kernel lacks the iptable_raw module ("can't initialize iptables table 'raw'").
docker build --network=host -t "${REF}" "${SCRIPT_DIR}"

echo "Pushing ${REF}"
docker push "${REF}"

echo "Done. Deploy in k3s (Flux reconciles fleet-deployment/gitops/clusters/tegra):"
echo "  set image tag ${TAG} in the ur-camera-streamer deployment manifest"
echo "  git commit && push, or: kubectl set image deploy/ur-camera-streamer streamer=${REF}"
