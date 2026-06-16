#!/usr/bin/env bash
# Build and push the GR00T control UI image to a container registry.
#
# The registry defaults to the example ACR but is overridable:
#   REGISTRY=myregistry.azurecr.io ./build_and_push.sh 0.1
#
# Usage: ./build_and_push.sh <tag>
# Example: ./build_and_push.sh 0.1
set -o errexit -o nounset -o pipefail

REGISTRY="${REGISTRY:-immitationlearning.azurecr.io}"
IMAGE="${IMAGE:-gr00t-control-ui}"
TAG="${1:?Usage: build_and_push.sh <tag>}"
REF="${REGISTRY}/${IMAGE}:${TAG}"

echo "Building ${REF}"
# --network=host avoids BuildKit's bridge setup, which fails on hosts whose
# kernel lacks the iptable_raw module ("can't initialize iptables table 'raw'").
docker build --network=host -t "${REF}" "$(dirname "$0")"

echo "Pushing ${REF}"
docker push "${REF}"

echo "Done. Enable it in the chart:"
echo "  helm upgrade --install gr00t ./gitops/charts/gr00t-inference \\"
echo "    --set controlUi.enabled=true --set controlUi.image.tag=${TAG}"
