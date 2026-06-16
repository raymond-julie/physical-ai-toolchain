#!/usr/bin/env bash
# Build and push the GR00T inference server image to a container registry.
#
# This builds only the runtime: the model weights are NOT baked in — the chart's
# modctl init container mounts them at /models/gr00t. Isaac-GR00T's Orin extras
# compile from source, so build on the Jetson/L4T host so the image arch (arm64)
# matches the cluster.
#
# The registry defaults to the example ACR but is overridable so the image can
# target any registry:
#   REGISTRY=myregistry.azurecr.io ./build_and_push.sh 0.1-l4t
#
# Usage: ./build_and_push.sh <tag> [base-image] [gr00t-ref]
# Example (Jetson Orin, JetPack 6.2 / L4T r36.4 — default base):
#   ./build_and_push.sh 0.1-l4t
set -o errexit -o nounset -o pipefail

REGISTRY="${REGISTRY:-immitationlearning.azurecr.io}"
IMAGE="${IMAGE:-gr00t-inference-server}"
TAG="${1:?Usage: build_and_push.sh <tag> [base-image] [gr00t-ref]}"
BASE_IMAGE="${2:-nvcr.io/nvidia/l4t-jetpack:r36.4.0}"
# Default to the N1.5 release tag: it registers the gr00t_n1_5 architecture the
# deployed checkpoint needs. Isaac-GR00T main has moved to N1.6/N1.7 and no
# longer loads gr00t_n1_5. Override only if you know what you are doing.
GR00T_REF="${3:-n1.5-release}"
REF="${REGISTRY}/${IMAGE}:${TAG}"

echo "Building ${REF}"
echo "  BASE_IMAGE=${BASE_IMAGE}"
echo "  GR00T_REF=${GR00T_REF}"
# --network=host avoids BuildKit's bridge setup, which fails on hosts whose
# kernel lacks the iptable_raw module ("can't initialize iptables table 'raw'").
docker build \
  --network=host \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "GR00T_REF=${GR00T_REF}" \
  -t "${REF}" \
  "$(dirname "$0")"

echo "Pushing ${REF}"
docker push "${REF}"

echo "Done. Set the chart to use it:"
echo "  helm upgrade --install gr00t ./gitops/charts/gr00t-inference \\"
echo "    --set image.repository=${REGISTRY}/${IMAGE} --set image.tag=${TAG}"
