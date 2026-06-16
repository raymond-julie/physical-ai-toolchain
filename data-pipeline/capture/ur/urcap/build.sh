#!/usr/bin/env bash
#
# Build the Camera Stream URCap (.urcap) for PolyScope 5 / e-Series.
#
# Requires the UR URCap SDK Maven artifacts (com.ur.urcap:api and the
# urcap-maven-plugin) to be resolvable — these ship with the SDK, not Maven
# Central. See README.md "Prerequisites" for how to install them.
#
# This script forces JDK 8 (URCaps must target Java 8) regardless of the system
# default java, and can seed the SDK's bundled Maven repo into ~/.m2 when you
# point it at the SDK with URCAP_SDK_DIR=/path/to/unpacked/sdk.
#
set -euo pipefail

cd "$(dirname "$0")"

# --- Force JDK 8 -------------------------------------------------------------
if [[ -z "${JAVA_HOME:-}" || ! -x "${JAVA_HOME}/bin/javac" || "$("${JAVA_HOME}/bin/javac" -version 2>&1)" != *" 1.8."* ]]; then
    arch="$(dpkg --print-architecture 2>/dev/null || echo amd64)"
    for candidate in \
        "/usr/lib/jvm/java-8-openjdk-${arch}" \
        "/usr/lib/jvm/java-1.8.0-openjdk-${arch}" \
        /usr/lib/jvm/java-8-openjdk-amd64 \
        /usr/lib/jvm/java-8-openjdk-arm64; do
        if [[ -x "${candidate}/bin/javac" ]]; then
            export JAVA_HOME="${candidate}"
            break
        fi
    done
fi

if [[ -z "${JAVA_HOME:-}" || ! -x "${JAVA_HOME}/bin/javac" ]]; then
    echo "ERROR: JDK 8 not found. Install it (e.g. apt-get install openjdk-8-jdk) or set JAVA_HOME." >&2
    exit 1
fi
export PATH="${JAVA_HOME}/bin:${PATH}"
echo "Using JAVA_HOME=${JAVA_HOME}"
javac -version

# --- Optionally seed the SDK's bundled Maven repo ----------------------------
# Set URCAP_SDK_DIR to the unpacked UR URCap SDK directory. The SDK ships a
# Maven repository (folder named 'repository' or 'artifacts'); we merge its jars
# into the local ~/.m2 so the build can resolve the com.ur.urcap:* artifacts.
if [[ -n "${URCAP_SDK_DIR:-}" ]]; then
    if [[ -d "${URCAP_SDK_DIR}" ]]; then
        echo "Seeding SDK artifacts from ${URCAP_SDK_DIR} ..."
        mkdir -p "${HOME}/.m2/repository"
        while IFS= read -r -d '' repo; do
            echo "  merging Maven repo: ${repo}"
            cp -an "${repo}/." "${HOME}/.m2/repository/" 2>/dev/null || true
        done < <(find "${URCAP_SDK_DIR}" -type d \( -name 'repository' -o -name 'artifacts' \) -print0 2>/dev/null)
    else
        echo "WARNING: URCAP_SDK_DIR=${URCAP_SDK_DIR} does not exist; skipping seed." >&2
    fi
fi

# --- Build -------------------------------------------------------------------
mvn clean install "$@"

URCAP_FILE="$(find . -name '*.urcap' -print -quit || true)"
if [[ -n "${URCAP_FILE}" ]]; then
    echo
    echo "Built URCap: ${URCAP_FILE}"
    echo "Install it via PolyScope: Settings -> System -> URCaps -> '+'"
else
    echo "Build finished but no .urcap artifact was found." >&2
    exit 1
fi
