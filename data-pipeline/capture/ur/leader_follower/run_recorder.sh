#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# UrRecorder — One-Script Launcher
#
# Starts:
#   1. Source reader  (reads source UR5e via RTDE + Robotiq)
#   2. Destination writer (state-machine: align → idle → mirror)
#   3. Recorder node  (ROS bag recording triggered by DI0)
#   4. Web dashboard  (serve_log.py on port 8080)
#
# Usage:
#   ./run_recorder.sh               # default (motion enabled)
#   ./run_recorder.sh --help        # all options
# ─────────────────────────────────────────────────────────────

set -eo pipefail

# ── Defaults ──────────────────────────────────────────────────
SOURCE_IP="192.168.1.80"
DEST_IP="192.168.1.90"
ENABLE_MOTION=true
USE_HOME=false
RECORD_DEPTH=false
# RealSense camera streaming profiles. Format: WxHxFPS.
# Lowering fps reduces USB bandwidth + Jetson CPU load, which can be
# the cause of jerky teleoperation when the source RTDE thread is
# CPU-starved. Override on the command line with --camera-fps.
# NOTE: D4xx hardware only accepts a discrete set of fps values
# (6, 15, 30, 60, 90 depending on resolution/sensor). Picking an
# unsupported value silently falls back to the driver default.
CAMERA_FPS=15
CAMERA_COLOR_PROFILE="640x480x${CAMERA_FPS}"
CAMERA_DEPTH_PROFILE="640x480x${CAMERA_FPS}"
MAX_VELOCITY=1.5
MAX_ACCELERATION=3.0
ALIGNMENT_SPEED=0.1
ALIGNMENT_THRESHOLD=0.02
SERVO_TIME=0.008
# Tuned 2026-04-28 for jerk debugging:
#   - lower lookahead + lower gain track better with less oscillation
#   - higher max_velocity / max_accel prevents servoJ saturation when
#     the operator moves the source quickly
SERVO_LOOKAHEAD=0.05
SERVO_GAIN=200
GRIPPER_SPEED=255
GRIPPER_FORCE=50

# Camera serial numbers (optional). When unset, the launcher will auto-detect
# connected RealSense devices via `rs-enumerate-devices` and assign them
# in the order they enumerate. Override on the command line with
# --cam1-serial / --cam2-serial or via the CAM1_SERIAL / CAM2_SERIAL env vars.
CAM1_SERIAL="${CAM1_SERIAL:-}"
CAM2_SERIAL="${CAM2_SERIAL:-}"

# ── Simulated camera (no RealSense hardware) ──────────────────
# When --video <path> is set the launcher skips realsense2_camera and
# starts video_to_camera.py to replay an mp4 on the camera1 topic.
# --video2 <path> adds a second simulated stream on camera2. Use
# --synthetic-camera to publish a moving test pattern instead of an
# mp4 (useful when no mp4 fixture is available at all).
VIDEO_PATH=""
VIDEO_PATH2=""
SYNTHETIC_CAMERA=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$SCRIPT_DIR/recorder_output.log"

# ── LeRobot dataset destination ───────────────────────────────
# When running locally the recorder writes under ./recordings_lerobot.
# In the cluster pod the launcher overrides these env vars to point at
# /cloud-sync/lerobot-recordings (an ACSA-backed PVC that mirrors
# asynchronously to the lerobot-recordings blob container).
LEROBOT_ROOT="${LEROBOT_ROOT:-$SCRIPT_DIR/recordings_lerobot}"
LEROBOT_REPO_ID="${LEROBOT_REPO_ID:-local/ur5_mirror}"

# Local-disk retention sidecar. Off by default so local-dev launches stay
# unchanged; the cluster Deployment sets LEROBOT_RETENTION_ENABLED=true.
LEROBOT_RETENTION_ENABLED="${LEROBOT_RETENTION_ENABLED:-false}"
LEROBOT_RETENTION_DAYS="${LEROBOT_RETENTION_DAYS:-7}"
LEROBOT_CLEANUP_INTERVAL_MINUTES="${LEROBOT_CLEANUP_INTERVAL_MINUTES:-60}"

# ── Colors ────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# Host IP shown in the "Web GUI: http://...:8080" hint. Detected from the
# route to the source robot so it stays correct whether this host is on
# wired or wifi. Falls back to hostname -I, then to 0.0.0.0.
detect_host_ip() {
    local ip=""
    if command -v ip >/dev/null 2>&1; then
        ip="$(ip -4 -o route get "${SOURCE_IP:-1.1.1.1}" 2>/dev/null \
            | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
    fi
    if [[ -z "$ip" ]] && command -v hostname >/dev/null 2>&1; then
        ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    fi
    echo "${ip:-0.0.0.0}"
}

# ── Help ──────────────────────────────────────────────────────
show_help() {
    echo ""
    echo -e "${BOLD}UrRecorder — Launcher${NC}"
    echo ""
    echo "Usage: ./run_recorder.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --source-ip IP          Source robot IP        (default: $SOURCE_IP)"
    echo "  --dest-ip IP            Destination robot IP   (default: $DEST_IP)"
    echo "  --max-velocity VAL      ServoJ velocity rad/s  (default: $MAX_VELOCITY)"
    echo "  --max-accel VAL         ServoJ accel rad/s²    (default: $MAX_ACCELERATION)"
    echo "  --alignment-speed VAL   Alignment speed rad/s  (default: $ALIGNMENT_SPEED)"
    echo "  --gripper-speed VAL     Gripper speed 0-255    (default: $GRIPPER_SPEED)"
    echo "  --gripper-force VAL     Gripper force 0-255    (default: $GRIPPER_FORCE)"
    echo "  --cam1-serial SN        RealSense serial for cam1 (default: auto)"
    echo "  --cam2-serial SN        RealSense serial for cam2 (default: auto)"
    echo "  --video PATH            Replay mp4 on camera1 topic (no RealSense)"
    echo "  --video2 PATH           Replay mp4 on camera2 topic (no RealSense)"
    echo "  --synthetic-camera      Publish a test pattern on camera1 (no mp4)"
    echo "  --home                  Go to home position at start/end"
    echo "  --no-home               Skip home position — mirror directly (default)"
    echo "  --depth                 Enable depth streams + recording"
    echo "  --no-depth              Disable depth streams + recording (default)"
    echo "  --camera-fps N          RealSense color/depth fps (default: $CAMERA_FPS)"
    echo "  --no-motion             Disable robot motion   (dry run)"
    echo "  -h, --help              Show this help"
    echo ""
    exit 0
}

# ── Parse args ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-ip)        SOURCE_IP="$2"; shift 2 ;;
        --dest-ip)          DEST_IP="$2"; shift 2 ;;
        --max-velocity)     MAX_VELOCITY="$2"; shift 2 ;;
        --max-accel)        MAX_ACCELERATION="$2"; shift 2 ;;
        --alignment-speed)  ALIGNMENT_SPEED="$2"; shift 2 ;;
        --gripper-speed)    GRIPPER_SPEED="$2"; shift 2 ;;
        --gripper-force)    GRIPPER_FORCE="$2"; shift 2 ;;
        --cam1-serial)      CAM1_SERIAL="$2"; shift 2 ;;
        --cam2-serial)      CAM2_SERIAL="$2"; shift 2 ;;
        --video)            VIDEO_PATH="$2"; shift 2 ;;
        --video2)           VIDEO_PATH2="$2"; shift 2 ;;
        --synthetic-camera) SYNTHETIC_CAMERA=true; shift ;;
        --home)             USE_HOME=true; shift ;;
        --no-home)          USE_HOME=false; shift ;;
        --depth)            RECORD_DEPTH=true; shift ;;
        --no-depth)         RECORD_DEPTH=false; shift ;;
        --camera-fps)       CAMERA_FPS="$2"
                            CAMERA_COLOR_PROFILE="640x480x${CAMERA_FPS}"
                            CAMERA_DEPTH_PROFILE="640x480x${CAMERA_FPS}"
                            shift 2 ;;
        --no-motion)        ENABLE_MOTION=false; shift ;;
        -h|--help)          show_help ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; show_help ;;
    esac
done

# ── PID tracking ─────────────────────────────────────────────
SOURCE_PID=""
DEST_PID=""
RECORDER_PID=""
DASHBOARD_PID=""
CAMERA_PID=""
CAMERA2_PID=""
RETENTION_PID=""
CLEANED_UP=false

cleanup() {
    if [[ "$CLEANED_UP" == "true" ]]; then return; fi
    CLEANED_UP=true
    echo ""
    echo -e "${YELLOW}╔═════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  Shutting down all components...    ║${NC}"
    echo -e "${YELLOW}╚═════════════════════════════════════╝${NC}"

    # PID + display-name pairs for the supervised processes. Order matters
    # only for log readability — Phase 1 sends SIGINT to all of them at
    # once so they can shut down in parallel.
    local pids=("$RECORDER_PID" "$DEST_PID" "$SOURCE_PID" "$CAMERA_PID" \
                "$CAMERA2_PID" "$DASHBOARD_PID" "$RETENTION_PID")
    local names=("Recorder" "Destination" "Source" "Camera1" "Camera2" \
                 "Dashboard" "Retention")

    # Phase 1 — polite SIGINT so each component runs its rclpy shutdown
    # hook (closes RTDE, flushes LeRobot meta, releases /dev/video*).
    for i in "${!pids[@]}"; do
        local pid="${pids[$i]}"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo -e "${CYAN}  SIGINT  ${names[$i]} (PID $pid)${NC}"
            kill -INT "$pid" 2>/dev/null || true
        fi
    done

    # Phase 2 — bounded wait. The previous implementation used `wait $pid`
    # which blocks indefinitely if a child ignores SIGINT (the
    # realsense2_camera launch wrapper does this in some versions). When
    # the launcher itself was SIGTERMed by `timeout` or systemd, that
    # block prevented Phase 3/4 from ever running, leaving orphans
    # holding the live arm and cameras. Poll every 0.5 s for up to 3 s.
    for _ in 1 2 3 4 5 6; do
        local alive=0
        for pid in "${pids[@]}"; do
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                alive=1
                break
            fi
        done
        [[ "$alive" == "0" ]] && break
        sleep 0.5
    done

    # Phase 3 — SIGTERM the survivors.
    for i in "${!pids[@]}"; do
        local pid="${pids[$i]}"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}  SIGTERM ${names[$i]} (PID $pid) — ignored SIGINT${NC}"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 1

    # Phase 4 — SIGKILL anything still standing.
    for i in "${!pids[@]}"; do
        local pid="${pids[$i]}"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo -e "${RED}  SIGKILL ${names[$i]} (PID $pid) — refused SIGTERM${NC}"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    # Phase 5 — orphan sweep. `ros2 launch realsense2_camera rs_launch.py`
    # forks a separate component process whose PID differs from the
    # launch-wrapper PID we tracked; when the wrapper exits without
    # forwarding the signal the camera node lingers.
    #
    # IMPORTANT: each pattern is anchored to the actual launching binary
    # (python3 <script> or the realsense2_camera C++ binary path). Earlier
    # versions used bare script names like "source_reader", which also
    # matched any parent shell or SSH wrapper whose argv mentioned that
    # string (e.g. a survivor-check pgrep). Anchored patterns prevent the
    # cleanup from killing the very process running it.
    local orphan_pats=(
        "python3.*destination_writer\.py"
        "python3.*source_reader\.py"
        "python3.*lerobot_recorder_node\.py"
        "python3.*gui_node\.py"
        "python3.*rs_launch\.py"
        "realsense2_camera/realsense2_camera_node"
        "python3.*video_to_camera\.py"
        "python3.*serve_log\.py"
        "python3.*local_retention\.py"
    )
    for pat in "${orphan_pats[@]}"; do
        if pgrep -f "$pat" >/dev/null 2>&1; then
            pkill -INT -f "$pat" 2>/dev/null || true
        fi
    done
    sleep 1
    for pat in "${orphan_pats[@]}"; do
        if pgrep -f "$pat" >/dev/null 2>&1; then
            pkill -KILL -f "$pat" 2>/dev/null || true
        fi
    done
    echo -e "${GREEN}All processes stopped.${NC}"
}

trap cleanup SIGINT SIGTERM EXIT

# ── Banner ────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║     UrRecorder — Mirror + Record Launcher           ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Source robot:       ${BOLD}${SOURCE_IP}${NC}"
echo -e "  Destination robot:  ${BOLD}${DEST_IP}${NC}"
if [[ "$ENABLE_MOTION" == "true" ]]; then
    echo -e "  Motion:             ${RED}${BOLD}ENABLED — ROBOT WILL MOVE${NC}"
else
    echo -e "  Motion:             ${YELLOW}${BOLD}DISABLED (dry run)${NC}"
fi
if [[ "$USE_HOME" == "true" ]]; then
    echo -e "  Home position:      ${BOLD}ENABLED — align at start, return at end${NC}"
else
    echo -e "  Home position:      ${YELLOW}${BOLD}DISABLED — direct mirroring${NC}"
fi
if [[ "$RECORD_DEPTH" == "true" ]]; then
    echo -e "  Depth streams:      ${BOLD}ENABLED — depth published & recorded${NC}"
else
    echo -e "  Depth streams:      ${YELLOW}${BOLD}DISABLED — RGB only${NC}"
fi
echo -e "  Camera fps:         ${BOLD}${CAMERA_FPS}${NC}"
echo -e "  Max velocity:       ${BOLD}${MAX_VELOCITY} rad/s${NC}"
echo -e "  Alignment speed:    ${BOLD}${ALIGNMENT_SPEED} rad/s${NC}"
echo ""

# ── Step 1: Network check ────────────────────────────────────
echo -e "${CYAN}[1/6] Checking network connectivity...${NC}"
if ! ping -c 1 -W 2 "$SOURCE_IP" &>/dev/null; then
    echo -e "${RED}  ✗ Cannot reach source robot at ${SOURCE_IP}${NC}"; exit 1
fi
echo -e "${GREEN}  ✓ Source ${SOURCE_IP} reachable${NC}"

if ! ping -c 1 -W 2 "$DEST_IP" &>/dev/null; then
    echo -e "${RED}  ✗ Cannot reach destination robot at ${DEST_IP}${NC}"; exit 1
fi
echo -e "${GREEN}  ✓ Destination ${DEST_IP} reachable${NC}"

# ── Step 2: ROS2 environment ─────────────────────────────────
echo -e "${CYAN}[2/6] Setting up ROS2 environment...${NC}"
# Auto-detect ROS distro: prefer $ROS_DISTRO, then jazzy (24.04), then humble.
if [[ -z "${ROS_DISTRO:-}" ]]; then
    for _d in jazzy humble rolling iron; do
        if [[ -f "/opt/ros/${_d}/setup.bash" ]]; then ROS_DISTRO="$_d"; break; fi
    done
fi
if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    echo -e "${GREEN}  ✓ Sourced /opt/ros/${ROS_DISTRO}/setup.bash${NC}"
else
    echo -e "${RED}  ✗ No ROS 2 installation found under /opt/ros/${NC}"; exit 1
fi

# Source workspace overlay if available (parent OR this folder).
for _overlay in "$ROOT_DIR/install/setup.bash" "$SCRIPT_DIR/install/setup.bash"; do
    if [[ -f "$_overlay" ]]; then
        source "$_overlay"
        echo -e "${GREEN}  ✓ Sourced overlay: $_overlay${NC}"
        break
    fi
done

# Kill any stragglers from a previous run that didn't shut down cleanly.
# A stale destination_writer in IDLE/ALIGNING would otherwise make the new
# GUI skip its motion-confirmation overlay, because gui_node flips
# motion_confirmed=True the moment it sees a non-WAITING /destination/state.
#
# Patterns are anchored to the launching executable (python3 ... or the
# realsense2_camera binary path) so they only match the real component
# processes, never a parent shell or SSH wrapper whose argv happens to
# mention the script name in a pgrep/echo string.
echo -e "${CYAN}  Cleaning up stale processes from previous runs...${NC}"
for pat in "python3.*destination_writer\.py" \
           "python3.*source_reader\.py" \
           "python3.*lerobot_recorder_node\.py" \
           "python3.*gui_node\.py" \
           "python3.*rs_launch\.py" \
           "realsense2_camera/realsense2_camera_node"; do
    if pgrep -f "$pat" >/dev/null 2>&1; then
        pkill -f "$pat" 2>/dev/null || true
    fi
done
fuser -k 8080/tcp 2>/dev/null || true
sleep 0.5

# ── Step 3: Cameras (RealSense, mp4 replay, or synthetic) ────
echo -e "${CYAN}[3/6] Starting cameras...${NC}"
SIM_CAMERA_REQUESTED=false
if [[ -n "$VIDEO_PATH" || -n "$VIDEO_PATH2" || "$SYNTHETIC_CAMERA" == "true" ]]; then
    SIM_CAMERA_REQUESTED=true
fi

if [[ "$SIM_CAMERA_REQUESTED" == "true" ]]; then
    # Simulated-camera path. Replaces realsense2_camera with one or two
    # video_to_camera.py processes that publish on the same topic names
    # the recorder subscribes to (/camera1/camera1/color/image_raw,
    # /camera2/camera2/color/image_raw). Used when no RealSense devices
    # are connected (lab cameras down, CI smoke tests, replaying a
    # previously recorded LeRobot mp4 chunk through the pipeline).
    VIDEO_NODE="$SCRIPT_DIR/video_to_camera.py"
    if [[ ! -f "$VIDEO_NODE" ]]; then
        echo -e "${RED}  ✗ Missing $VIDEO_NODE${NC}"; exit 1
    fi

    if [[ "$SYNTHETIC_CAMERA" == "true" ]]; then
        python3 "$VIDEO_NODE" --ros-args \
            -p camera_name:=camera1 \
            -p synthetic:=true \
            -p fps:="$CAMERA_FPS" \
            >"$SCRIPT_DIR/camera1.log" 2>&1 </dev/null &
        CAMERA_PID=$!
        CAM1_MODEL="SYN"
        echo -e "${YELLOW}  ⊘ RealSense skipped — publishing synthetic test pattern on camera1 (PID $CAMERA_PID)${NC}"
    elif [[ -n "$VIDEO_PATH" ]]; then
        if [[ ! -f "$VIDEO_PATH" ]]; then
            echo -e "${RED}  ✗ --video file not found: $VIDEO_PATH${NC}"; exit 1
        fi
        python3 "$VIDEO_NODE" --ros-args \
            -p camera_name:=camera1 \
            -p video_path:="$VIDEO_PATH" \
            -p fps:="$CAMERA_FPS" \
            -p loop:=true \
            >"$SCRIPT_DIR/camera1.log" 2>&1 </dev/null &
        CAMERA_PID=$!
        CAM1_MODEL="MP4"
        echo -e "${YELLOW}  ⊘ RealSense skipped — replaying $VIDEO_PATH on camera1 (PID $CAMERA_PID)${NC}"
    fi

    if [[ -n "$VIDEO_PATH2" ]]; then
        if [[ ! -f "$VIDEO_PATH2" ]]; then
            echo -e "${RED}  ✗ --video2 file not found: $VIDEO_PATH2${NC}"; exit 1
        fi
        python3 "$VIDEO_NODE" --ros-args \
            -p camera_name:=camera2 \
            -p video_path:="$VIDEO_PATH2" \
            -p fps:="$CAMERA_FPS" \
            -p loop:=true \
            >"$SCRIPT_DIR/camera2.log" 2>&1 </dev/null &
        CAMERA2_PID=$!
        CAM2_MODEL="MP4"
        echo -e "${YELLOW}  ⊘ RealSense skipped — replaying $VIDEO_PATH2 on camera2 (PID $CAMERA2_PID)${NC}"
    fi

    sleep 2
    if [[ -n "$CAMERA_PID" ]] && ! kill -0 "$CAMERA_PID" 2>/dev/null; then
        echo -e "${RED}  ✗ camera1 simulator failed — see $SCRIPT_DIR/camera1.log${NC}"
        CAMERA_PID=""
    fi
    if [[ -n "$CAMERA2_PID" ]] && ! kill -0 "$CAMERA2_PID" 2>/dev/null; then
        echo -e "${RED}  ✗ camera2 simulator failed — see $SCRIPT_DIR/camera2.log${NC}"
        CAMERA2_PID=""
    fi
elif ros2 pkg list 2>/dev/null | grep -q realsense2_camera; then
    # Strategy: probe sysfs for all Intel-VID (0x8086) RealSense USB
    # devices and disambiguate them via realsense2_camera's
    # `usb_port_id:=<bus-port>` parameter. The sysfs USB port (e.g.
    # `2-3.2`) is unambiguous and stable, and avoids the well-known mess
    # where the kernel's USB descriptor serial differs from the
    # librealsense ASIC serial reported by `rs-enumerate-devices`.
    #
    # Explicit overrides via --cam1-serial / --cam2-serial still take
    # priority; those go through `serial_no:=_<SERIAL>`.
    #
    # We also extract the USB product id and map it to a RealSense model
    # name so the launcher works for both D435 and D405 (and friends)
    # without code changes. The set of recognised PIDs is taken from
    # librealsense's `rs_pid.h`. Only devices with a known RealSense PID
    # are launched; this avoids picking up unrelated Intel USB devices
    # (e.g. some Jetson modules ship Intel WiFi cards with VID 8086).
    rs_model_for_pid() {
        # Lower-case 4-digit hex PID -> human-readable model name.
        case "${1,,}" in
            0ad1|0ad2|0ad3|0ad4|0ad5|0ad6) echo "D4xx" ;;
            0b07) echo "D435" ;;
            0b3a) echo "D435i" ;;
            0b5b) echo "D405" ;;
            0b5c) echo "D455" ;;
            0b48) echo "D415" ;;
            0b49) echo "D416" ;;
            0b52) echo "D416-RGB" ;;
            0b4b|0b4d) echo "D4xx" ;;
            0b00|0b01|0b03|0b0c) echo "D4xx" ;;
            *) echo "" ;;
        esac
    }

    DETECTED_PORTS=()
    DETECTED_PORT_SERIALS=()
    DETECTED_MODELS=()
    while IFS= read -r dev; do
        vendor_file="$dev/idVendor"
        product_file="$dev/idProduct"
        [[ -r "$vendor_file" && -r "$product_file" ]] || continue
        vendor="$(cat "$vendor_file" 2>/dev/null)"
        [[ "$vendor" == "8086" ]] || continue
        # Skip USB hubs.
        class="$(cat "$dev/bDeviceClass" 2>/dev/null)"
        [[ "$class" == "09" ]] && continue
        product="$(cat "$product_file" 2>/dev/null)"
        model="$(rs_model_for_pid "$product")"
        # Skip Intel devices that are not RealSense cameras (e.g. WiFi).
        [[ -z "$model" ]] && continue
        # Port id is the basename of the sysfs path (e.g. "2-3.2").
        port="$(basename "$dev")"
        sysfs_serial="$(cat "$dev/serial" 2>/dev/null | tr -d '[:space:]')"
        DETECTED_PORTS+=("$port")
        DETECTED_PORT_SERIALS+=("${sysfs_serial:-?}")
        DETECTED_MODELS+=("$model")
    done < <(ls -d /sys/bus/usb/devices/*/ 2>/dev/null)

    USB_DEVICE_COUNT="${#DETECTED_PORTS[@]}"
    CAM1_PORT=""
    CAM2_PORT=""
    CAM1_MODEL=""
    CAM2_MODEL=""
    if [[ -z "$CAM1_SERIAL" && -n "${DETECTED_PORTS[0]:-}" ]]; then
        CAM1_PORT="${DETECTED_PORTS[0]}"
        CAM1_MODEL="${DETECTED_MODELS[0]:-}"
    fi
    if [[ -z "$CAM2_SERIAL" && -n "${DETECTED_PORTS[1]:-}" ]]; then
        CAM2_PORT="${DETECTED_PORTS[1]}"
        CAM2_MODEL="${DETECTED_MODELS[1]:-}"
    fi

    if [[ "$USB_DEVICE_COUNT" -ge 1 ]]; then
        msg="  Detected $USB_DEVICE_COUNT RealSense device(s):"
        for i in "${!DETECTED_PORTS[@]}"; do
            msg+=" [${DETECTED_MODELS[$i]} @ ${DETECTED_PORTS[$i]}=${DETECTED_PORT_SERIALS[$i]}]"
        done
        echo -e "${CYAN}${msg}${NC}"
    else
        echo -e "${YELLOW}  ⚠ No RealSense USB device detected via sysfs${NC}"
    fi

    # Launch a single camera in the background. Stdout/stderr are redirected
    # to a per-camera log file so the launcher does not block waiting for
    # `ros2 launch` to close its file descriptors.
    LOG_DIR="$SCRIPT_DIR"
    launch_camera() {
        # $1 = camera name (used as namespace and name)
        # $2 = librealsense serial (optional, takes priority)
        # $3 = USB port id (e.g. "2-3.2", optional fallback)
        # $4 = camera model name (e.g. "D435", "D405"), used to pick
        #      model-specific launch args. Optional; empty = generic.
        local name="$1" serial="$2" port="$3" model="${4:-}"
        local logfile="$LOG_DIR/${name}.log"
        local args=(
            camera_namespace:="$name"
            camera_name:="$name"
            enable_color:=true
            enable_depth:="$RECORD_DEPTH"
            align_depth.enable:="$RECORD_DEPTH"
            rgb_camera.color_profile:="${CAMERA_COLOR_PROFILE}"
            depth_module.depth_profile:="${CAMERA_DEPTH_PROFILE}"
        )
        # Model-specific tweaks. The D405 has no IR projector and a
        # different recommended depth profile; the rs_launch.py defaults
        # already handle that, so we only override where it actually
        # matters. Disable IR emitter explicitly on D405 to silence the
        # driver warning about an unsupported control.
        case "$model" in
            D405)
                args+=("depth_module.emitter_enabled:=0")
                ;;
            D435|D435i|D455)
                # Defaults are fine for these.
                :
                ;;
        esac
        if [[ -n "$serial" ]]; then
            args+=("serial_no:=_$serial")
        elif [[ -n "$port" ]]; then
            args+=("usb_port_id:=$port")
        fi
        ros2 launch realsense2_camera rs_launch.py "${args[@]}" \
            >"$logfile" 2>&1 </dev/null &
    }

    launch_camera camera1 "$CAM1_SERIAL" "$CAM1_PORT" "$CAM1_MODEL"
    CAMERA_PID=$!
    sleep 2

    # Launch camera2 only when a *distinct* second device exists. We
    # disambiguate by sysfs USB port, which is stable and unambiguous.
    LAUNCH_CAM2=false
    if [[ -n "$CAM2_SERIAL" && "$CAM2_SERIAL" != "$CAM1_SERIAL" ]]; then
        LAUNCH_CAM2=true
    elif [[ -n "$CAM2_PORT" && "$CAM2_PORT" != "$CAM1_PORT" ]]; then
        LAUNCH_CAM2=true
    fi

    if $LAUNCH_CAM2; then
        launch_camera camera2 "$CAM2_SERIAL" "$CAM2_PORT" "$CAM2_MODEL"
        CAMERA2_PID=$!
        sleep 4
    else
        echo -e "${YELLOW}  ⊘ Only ${USB_DEVICE_COUNT} RealSense device(s) detected — skipping Camera2${NC}"
        echo -e "${YELLOW}    (recorder will run with record_camera2:=false)${NC}"
        CAMERA2_PID=""
        sleep 2
    fi

    if kill -0 "$CAMERA_PID" 2>/dev/null; then
        echo -e "${GREEN}  ✓ Camera1 running (PID $CAMERA_PID${CAM1_MODEL:+, model $CAM1_MODEL}${CAM1_SERIAL:+, serial $CAM1_SERIAL}${CAM1_PORT:+, port $CAM1_PORT}) — log: $LOG_DIR/camera1.log${NC}"
    else
        echo -e "${YELLOW}  ⚠ Camera1 failed to start — see $LOG_DIR/camera1.log${NC}"
        CAMERA_PID=""
    fi
    if [[ -n "$CAMERA2_PID" ]]; then
        if kill -0 "$CAMERA2_PID" 2>/dev/null; then
            echo -e "${GREEN}  ✓ Camera2 running (PID $CAMERA2_PID${CAM2_MODEL:+, model $CAM2_MODEL}${CAM2_SERIAL:+, serial $CAM2_SERIAL}${CAM2_PORT:+, port $CAM2_PORT}) — log: $LOG_DIR/camera2.log${NC}"
        else
            echo -e "${YELLOW}  ⚠ Camera2 failed to start — see $LOG_DIR/camera2.log${NC}"
            CAMERA2_PID=""
        fi
    fi
else
    echo -e "${YELLOW}  ⊘ realsense2_camera package not found — cameras skipped${NC}"
    CAMERA_PID=""
    CAMERA2_PID=""
fi

# Recorder will skip camera2 features when no second camera is running.
if [[ -n "$CAMERA2_PID" ]]; then
    RECORD_CAMERA2=true
else
    RECORD_CAMERA2=false
fi

# ── Step 4: Web GUI ───────────────────────────────────────────
echo -e "${CYAN}[4/6] Starting web GUI...${NC}"
fuser -k 8080/tcp 2>/dev/null || true
sleep 0.3

GUI_NODE="$SCRIPT_DIR/gui_node.py"
if [[ -f "$GUI_NODE" ]]; then
    cd "$SCRIPT_DIR"
    CAM1_MODEL="$CAM1_MODEL" CAM2_MODEL="$CAM2_MODEL" \
    USE_HOME_DEFAULT="$USE_HOME" RECORD_DEPTH_DEFAULT="$RECORD_DEPTH" \
        python3 "$GUI_NODE" &>/dev/null &
    DASHBOARD_PID=$!
    sleep 2
    if kill -0 "$DASHBOARD_PID" 2>/dev/null; then
        HOST_IP="$(detect_host_ip)"
        echo -e "${GREEN}  ✓ Web GUI on http://${HOST_IP}:8080 (PID $DASHBOARD_PID)${NC}"
    else
        echo -e "${YELLOW}  ⚠ Web GUI failed to start${NC}"
        DASHBOARD_PID=""
    fi
else
    echo -e "${YELLOW}  ⊘ gui_node.py not found — GUI skipped${NC}"
fi

# ── Step 5: Start source reader ──────────────────────────────
echo -e "${CYAN}[5/6] Starting source reader...${NC}"
cd "$SCRIPT_DIR"
python3 source_reader.py --ros-args -p robot_ip:="$SOURCE_IP" &
SOURCE_PID=$!
sleep 3

if ! kill -0 "$SOURCE_PID" 2>/dev/null; then
    echo -e "${RED}  ✗ Source reader failed to start${NC}"; exit 1
fi
echo -e "${GREEN}  ✓ Source reader running (PID $SOURCE_PID)${NC}"

# ── Safety warning ───────────────────────────────────────────
if [[ "$ENABLE_MOTION" == "true" ]]; then
    echo ""
    echo -e "${YELLOW}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  ${RED}${BOLD}⚠  MOTION IS ABOUT TO BE ENABLED  ⚠${NC}${YELLOW}                ║${NC}"
    echo -e "${YELLOW}║                                                      ║${NC}"
    if [[ "$USE_HOME" == "true" ]]; then
        echo -e "${YELLOW}║  The destination robot WILL move slowly to home.     ║${NC}"
    else
        echo -e "${YELLOW}║  The destination robot WILL mirror directly.         ║${NC}"
    fi
    echo -e "${YELLOW}║  Make sure:                                          ║${NC}"
    echo -e "${YELLOW}║    1. Workspace is clear of people/obstacles         ║${NC}"
    echo -e "${YELLOW}║    2. E-stop is within reach                         ║${NC}"
    echo -e "${YELLOW}║    3. Destination robot brakes are released           ║${NC}"
    echo -e "${YELLOW}║                                                      ║${NC}"
    echo -e "${YELLOW}║  Press DI0 or use the Web GUI to start mirroring    ║${NC}"
    echo -e "${YELLOW}║  + recording.                                        ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
fi

# ── Step 6: Start recorder + destination writer ──────────────
echo -e "${CYAN}[6/6] Starting recorder + destination writer...${NC}"

# Recorder node (background) — writes LeRobotDataset (parquet + mp4)
cd "$SCRIPT_DIR"
RECORDER_LOG="${LOG_DIR:-$SCRIPT_DIR}/lerobot_recorder.log"
echo -e "${CYAN}    Recorder log: $RECORDER_LOG${NC}"
python3 -u lerobot_recorder_node.py --ros-args \
    -p root:="$LEROBOT_ROOT" \
    -p repo_id:="$LEROBOT_REPO_ID" \
    -p fps:="$CAMERA_FPS" \
    -p record_camera2:="$RECORD_CAMERA2" \
    -p record_depth:="$RECORD_DEPTH" \
    > "$RECORDER_LOG" 2>&1 &
RECORDER_PID=$!
echo -e "${GREEN}  ✓ LeRobot recorder running (PID $RECORDER_PID)${NC}"
echo -e "${GREEN}    Dataset: $LEROBOT_ROOT/$LEROBOT_REPO_ID${NC}"

# Optional: local-disk retention loop. Off by default so local-dev launches
# stay unchanged; the cluster Deployment enables it so the ACSA-backed PVC
# does not grow without bound (ACSA mirrors to blob but never deletes
# locally).
if [[ "${LEROBOT_RETENTION_ENABLED,,}" == "true" || "$LEROBOT_RETENTION_ENABLED" == "1" ]]; then
    RETENTION_LOG="${LOG_DIR:-$SCRIPT_DIR}/local_retention.log"
    echo -e "${CYAN}    Retention log: $RETENTION_LOG${NC}"
    LEROBOT_ROOT="$LEROBOT_ROOT" \
    LEROBOT_REPO_ID="$LEROBOT_REPO_ID" \
    LEROBOT_RETENTION_DAYS="$LEROBOT_RETENTION_DAYS" \
    LEROBOT_CLEANUP_INTERVAL_MINUTES="$LEROBOT_CLEANUP_INTERVAL_MINUTES" \
        python3 -u "$SCRIPT_DIR/local_retention.py" \
        > "$RETENTION_LOG" 2>&1 &
    RETENTION_PID=$!
    echo -e "${GREEN}  ✓ Local retention sidecar running (PID $RETENTION_PID, ${LEROBOT_RETENTION_DAYS}d window)${NC}"
fi

sleep 1

echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║  System running — waiting for GUI confirmation      ║${NC}"
echo -e "${GREEN}${BOLD}║                                                      ║${NC}"
echo -e "${GREEN}${BOLD}║  Open the Web GUI and confirm motion to proceed:    ║${NC}"
echo -e "${GREEN}${BOLD}║  Web GUI: $(printf 'http://%-44s' "${HOST_IP:-$(detect_host_ip)}:8080")║${NC}"
echo -e "${GREEN}${BOLD}║                                                      ║${NC}"
echo -e "${GREEN}${BOLD}║  Press Ctrl+C to shut down everything               ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Destination writer runs in FOREGROUND — Ctrl+C triggers cleanup trap
python3 destination_writer.py --ros-args \
    -p robot_ip:="$DEST_IP" \
    -p enable_motion:="$ENABLE_MOTION" \
    -p max_velocity:="$MAX_VELOCITY" \
    -p max_acceleration:="$MAX_ACCELERATION" \
    -p servo_time:="$SERVO_TIME" \
    -p servo_lookahead:="$SERVO_LOOKAHEAD" \
    -p servo_gain:="$SERVO_GAIN" \
    -p gripper_speed:="$GRIPPER_SPEED" \
    -p gripper_force:="$GRIPPER_FORCE" \
    -p alignment_speed:="$ALIGNMENT_SPEED" \
    -p alignment_threshold:="$ALIGNMENT_THRESHOLD" \
    -p use_home:="$USE_HOME"
