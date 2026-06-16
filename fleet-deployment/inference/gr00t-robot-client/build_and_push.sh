#!/usr/bin/env bash
# Build and push the GR00T dual-arm inference client image to a container registry.
#
# ur_rtde is compiled from source (no aarch64 wheel), so the build is slow.
# Build on the Jetson/L4T host so the image arch (arm64) matches the cluster.
#
# The registry defaults to the example ACR but is overridable so the image can
# target any registry:
#   REGISTRY=myregistry.azurecr.io ./build_and_push.sh 0.1
#
# Usage: ./build_and_push.sh <tag>
# Example: ./build_and_push.sh 0.1
set -o errexit -o nounset -o pipefail

REGISTRY="${REGISTRY:-immitationlearning.azurecr.io}"
IMAGE="${IMAGE:-gr00t-robot-client}"
TAG="${1:?Usage: build_and_push.sh <tag>}"
REF="${REGISTRY}/${IMAGE}:${TAG}"

echo "Building ${REF}"
# --network=host avoids BuildKit's bridge setup, which fails on hosts whose
# kernel lacks the iptable_raw module ("can't initialize iptables table 'raw'").
docker build --network=host -t "${REF}" "$(dirname "$0")"

echo "Pushing ${REF}"
docker push "${REF}"

echo "Done. Enable it in the HelmRelease:"
echo "  robotClient.enabled=true robotClient.image.tag=${TAG}"
