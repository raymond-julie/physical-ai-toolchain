#!/usr/bin/env bash
# Stage the prebuilt Orbbec SDK artifacts into the Docker build context.
#
# pyorbbecsdk is not on PyPI and is built from source for this exact platform
# (aarch64 / CPython 3.10 / Ubuntu 22.04). This copies the already-installed
# artifacts from the host's user site-packages into docker/vendor/orbbec/ so the
# Dockerfile can COPY them into the image without rebuilding the SDK.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/vendor/orbbec"
SRC="${ORBBEC_SITE_PACKAGES:-${HOME}/.local/lib/python3.10/site-packages}"

if [[ ! -f "${SRC}/pyorbbecsdk.cpython-310-aarch64-linux-gnu.so" ]]; then
  echo "ERROR: pyorbbecsdk not found in ${SRC}" >&2
  echo "Set ORBBEC_SITE_PACKAGES to the dir containing the built SDK." >&2
  exit 1
fi

rm -rf "${DEST}"
mkdir -p "${DEST}"

# The Python binding, the SDK shared libs, and the runtime extensions.
cp -av "${SRC}"/pyorbbecsdk.cpython-310-aarch64-linux-gnu.so "${DEST}"/
cp -av "${SRC}"/libOrbbecSDK.so* "${DEST}"/
cp -av "${SRC}"/extensions "${DEST}"/

echo "Staged Orbbec SDK into ${DEST}:"
ls -1 "${DEST}"
