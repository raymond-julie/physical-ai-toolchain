#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "requirements.txt not found in $SCRIPT_DIR"
    exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
    echo "This installer currently supports apt-based systems only."
    exit 1
fi

SUDO=""
if [[ "$EUID" -ne 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
        echo "sudo is required to install system packages."
        exit 1
    fi
    SUDO="sudo"
fi

APT_PACKAGES=(
    python3-pip
    python3-colcon-common-extensions
    python3-opencv
    build-essential
    cmake
    git
    pkg-config
    libgl1
    libglib2.0-0
    # Required to build ur_rtde from source (especially on Jetson/ARM where no
    # prebuilt wheel is available). Without these, pip install ur_rtde fails
    # with CMake errors about boost_system / boost_thread.
    libboost-all-dev
    curl
    gnupg2
    ca-certificates
    lsb-release
    software-properties-common
)

ROS_PACKAGES=(
    "ros-${ROS_DISTRO}-rclpy"
    "ros-${ROS_DISTRO}-sensor-msgs"
    "ros-${ROS_DISTRO}-std-msgs"
    "ros-${ROS_DISTRO}-ur-msgs"
    "ros-${ROS_DISTRO}-launch"
    "ros-${ROS_DISTRO}-launch-ros"
    "ros-${ROS_DISTRO}-ros2bag"
    # cv_bridge is required by the recorder nodes to convert
    # /camera/.../image_raw + /image_rect_raw into numpy arrays before they
    # are written into the LeRobotDataset.
    "ros-${ROS_DISTRO}-cv-bridge"
    # Core ROS 2 CLI verbs. A base ROS install on Jetson may ship without
    # these, which causes `ros2 run` / `ros2 pkg list` / `ros2 topic echo`
    # to fail with "invalid choice" errors.
    "ros-${ROS_DISTRO}-ros2run"
    "ros-${ROS_DISTRO}-ros2pkg"
    "ros-${ROS_DISTRO}-ros2node"
    "ros-${ROS_DISTRO}-ros2topic"
)

# Intel RealSense ROS 2 driver, used by run_episode_recorder.sh to publish
# /camera/camera/color/image_raw and /camera/camera/depth/image_rect_raw.
# Without this, the GUI camera preview stays blank and LeRobot episodes have
# no images. Installed best-effort: on some Jetson images the Ubuntu archive
# does not carry these packages, in which case the user must add the Intel
# librealsense apt repo (see README troubleshooting). We do not let a
# missing realsense package abort the rest of the install.
REALSENSE_PACKAGES=(
    "ros-${ROS_DISTRO}-realsense2-camera"
    "ros-${ROS_DISTRO}-realsense2-description"
)

ensure_ros_apt_repo() {
    local list_file="/etc/apt/sources.list.d/ros2.list"
    local key_file="/usr/share/keyrings/ros-archive-keyring.gpg"

    if [[ -f "$list_file" ]]; then
        return 0
    fi

    echo "Configuring ROS 2 apt repository for ${ROS_DISTRO}..."
    "$SUDO" add-apt-repository -y universe
    "$SUDO" curl -sSL \
        https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o "$key_file"
    local arch codename
    arch="$(dpkg --print-architecture)"
    codename="$(. /etc/os-release && echo "$UBUNTU_CODENAME")"
    echo "deb [arch=${arch} signed-by=${key_file}] http://packages.ros.org/ros2/ubuntu ${codename} main" \
        | "$SUDO" tee "$list_file" > /dev/null
    "$SUDO" apt-get update
}

echo "Installing apt packages..."
"$SUDO" apt-get update
"$SUDO" apt-get install -y curl gnupg2 ca-certificates lsb-release software-properties-common
ensure_ros_apt_repo
"$SUDO" apt-get install -y "${APT_PACKAGES[@]}"

# Install ROS packages one-by-one; some packages (e.g. ros2run/ros2pkg, ur-msgs)
# are not split out the same way on every distro (jazzy bundles many ros2cli
# verbs together, and ur-msgs may be unavailable). Don't abort on a single
# missing package.
echo "Installing ROS 2 packages (best-effort per package)..."
for _pkg in "${ROS_PACKAGES[@]}"; do
    if ! "$SUDO" apt-get install -y "$_pkg" 2>/dev/null; then
        echo "  WARN: $_pkg not available on ${ROS_DISTRO} \u2014 skipping."
    fi
done

# RealSense is best-effort: do not let a missing package abort the install.
echo "Installing RealSense ROS 2 driver (best-effort)..."
if "$SUDO" apt-get install -y "${REALSENSE_PACKAGES[@]}"; then
    echo "  RealSense driver installed."
else
    cat <<'WARN'
  WARNING: Failed to install ros-${ROS_DISTRO}-realsense2-camera.
  The mirror+record stack will still run, but the GUI camera preview will
  be blank and recorded episodes will have no images.

  On Jetson the driver may only be available via the Intel librealsense
  apt repo. To add it and retry:

    sudo mkdir -p /etc/apt/keyrings
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
      | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" \
      | sudo tee /etc/apt/sources.list.d/librealsense.list
    sudo apt-get update
    sudo apt-get install -y ros-${ROS_DISTRO}-realsense2-camera \
                            ros-${ROS_DISTRO}-realsense2-description \
                            librealsense2-dev librealsense2-utils
WARN
fi

PIP_ARGS=()
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Using active virtual environment: $VIRTUAL_ENV"
elif [[ "$EUID" -ne 0 ]]; then
    PIP_ARGS+=(--user)
    echo "Installing pip packages into the current user's site-packages"
else
    echo "Installing pip packages system-wide as root"
fi

# Ubuntu 23.04+ (incl. 24.04 noble) marks the system Python as
# externally-managed (PEP 668). Pip refuses to install into it without
# --break-system-packages. We add the flag when no venv is active.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
        PIP_ARGS+=(--break-system-packages)
    fi
fi

# On ROS 2 Humble, cv_bridge is built against NumPy 1.x and we MUST pin
# numpy<2. On Jazzy (Ubuntu 24.04), cv_bridge is built against NumPy 2.x,
# so the pin would break `from cv_bridge import CvBridge`. Pick the right
# constraint based on the active ROS distro.
case "$ROS_DISTRO" in
    humble|iron) NUMPY_PIN="numpy<2" ;;
    *)           NUMPY_PIN="numpy" ;;
esac

echo "Installing pip packages from requirements.txt..."
python3 -m pip install "${PIP_ARGS[@]}" --upgrade pip

LIGHT_PIP_PACKAGES=(
    setuptools
    "$NUMPY_PIN"
    pandas
    flask
    flask-socketio
    opencv-python
)
python3 -m pip install "${PIP_ARGS[@]}" --upgrade "${LIGHT_PIP_PACKAGES[@]}"

# LeRobot dataset writer used by the episode recorder node.
python3 -m pip install "${PIP_ARGS[@]}" lerobot

# Re-apply the numpy constraint in case lerobot's resolver shifted it.
python3 -m pip install "${PIP_ARGS[@]}" --upgrade "$NUMPY_PIN"

# Now install ur_rtde (slow source build on ARM).
python3 -m pip install "${PIP_ARGS[@]}" ur_rtde

# Finally, reconcile against the pinned requirements file.
python3 -m pip install "${PIP_ARGS[@]}" -r "$REQUIREMENTS_FILE"

cat <<EOF

Dependency install complete.

Next steps:
  source /opt/ros/${ROS_DISTRO}/setup.bash
  cd ${SCRIPT_DIR}
  ./run_episode_recorder.sh --help

EOF
