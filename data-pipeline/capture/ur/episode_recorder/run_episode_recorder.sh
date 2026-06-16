#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# UrEpisodeRecorder — One-Script Launcher
#
# Starts:
#   1. Two RealSense cameras (auto-detected, dual)
#   2. Robot1 reader  (RTDE + Robotiq, read-only)
#   3. Robot2 reader  (RTDE + Robotiq, read-only)
#   4. Tool-IO trigger (subscribes to /robot1/digital_input/di0)
#   5. GUI trigger    (web Record button, fallback when no physical button)
#   6. Episode recorder (LeRobotDataset writer)
#
# Neither robot is commanded. Either the physical DI0 on Robot1 or the
# Record button in the web GUI toggles /recorder/active.
#
# Usage:
#   ./run_episode_recorder.sh              # default
#   ./run_episode_recorder.sh --help       # all options
# ─────────────────────────────────────────────────────────────

set -eo pipefail

# ── Defaults ──────────────────────────────────────────────────
STATE_SOURCE="rtde"              # rtde | nova
ROBOT1_IP="192.168.1.80"
ROBOT2_IP="192.168.1.90"
ROBOT1_NAME="robot1"
ROBOT2_NAME="robot2"
ROBOT1_DRIVER="ur_rtde"
ROBOT2_DRIVER="ur_rtde"
GRIPPER1_DRIVER="robotiq_socket"
GRIPPER2_DRIVER="robotiq_socket"
RECORD_DEPTH=false
CAMERA_FPS=15
CAMERA_COLOR_PROFILE="640x480x${CAMERA_FPS}"
CAMERA_DEPTH_PROFILE="640x480x${CAMERA_FPS}"
GUI_PORT=8080
TRIGGER_INPUT=""   # auto: /<robot1-name>/digital_input/di0
ENABLE_TOOL_TRIGGER=true
ENABLE_CAMERAS=true
EPISODES_PER_SESSION=10   # roll over to a new session_<ts> after N saved episodes; 0=disable

# Camera source: where the recorder gets image frames from.
#   local : launch realsense2_camera on this host (USB-attached cameras).
#   nova  : connect to Wandelbots Nova's RealSense app (WebRTC) and
#           republish each camera's color track as a ROS Image topic.
CAMERA_SOURCE="local"
CAMERA_SOURCE_EXPLICIT=false
NOVA_CAM_API_BASE="${NOVA_CAM_API_BASE:-http://192.168.1.71/cell/realsense}"
# Comma-separated Nova camera serials (== device_id). When empty, the
# launcher auto-discovers them via GET /api/devices/.
NOVA_CAM_SERIALS="${NOVA_CAM_SERIALS:-}"
NOVA_CAM_STREAM_TYPES="${NOVA_CAM_STREAM_TYPES:-color}"

# ── Nova-specific defaults (only used when --state-source nova) ─
NOVA_NATS_URL="nats://127.0.0.1:4222"
NOVA_CELL="cell"
NOVA_CTRL1="ur5-left"
NOVA_CTRL2="ur-right"
NOVA_NATS_USER=""
NOVA_NATS_PASSWORD=""
NOVA_NATS_CREDS_FILE=""
# Optional kubectl port-forward to expose the Nova NATS svc on localhost.
# Auto-enabled when NOVA_NATS_URL host is 127.0.0.1/localhost.
NOVA_PORT_FORWARD="auto"          # auto | on | off
NOVA_PF_NAMESPACE="wandelbots"
NOVA_PF_SERVICE="svc/nats"
NOVA_PF_LOCAL_PORT="4222"
NOVA_PF_REMOTE_PORT="4222"
NOVA_KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

CAM1_SERIAL="${CAM1_SERIAL:-}"
CAM2_SERIAL="${CAM2_SERIAL:-}"
CAM3_SERIAL="${CAM3_SERIAL:-}"
CAM4_SERIAL="${CAM4_SERIAL:-}"
MAX_CAMERAS=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR"

# Make the package importable without installation.
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# ── Colors ────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

detect_host_ip() {
    local ip=""
    if command -v ip >/dev/null 2>&1; then
        ip="$(ip -4 -o route get "${ROBOT1_IP:-1.1.1.1}" 2>/dev/null \
            | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
    fi
    if [[ -z "$ip" ]] && command -v hostname >/dev/null 2>&1; then
        ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    fi
    echo "${ip:-0.0.0.0}"
}

# ── Help ──────────────────────────────────────────────────────
show_help() {
    cat <<EOF

UrEpisodeRecorder — Launcher

Usage: ./run_episode_recorder.sh [OPTIONS]

State source:
  --state-source MODE       rtde | nova                  (default: $STATE_SOURCE)
                            rtde: each reader talks to UR controllers directly via RTDE.
                            nova: each reader subscribes to Wandelbots Nova's NATS
                                  controller-state subject. Auto-disables --no-tool-trigger
                                  because Nova v2 state does not expose tool DIs.

Robots (RTDE mode):
  --robot1-ip IP            Robot1 IP                    (default: $ROBOT1_IP)
  --robot2-ip IP            Robot2 IP                    (default: $ROBOT2_IP)
  --robot1-name NAME        Robot1 topic namespace       (default: $ROBOT1_NAME)
  --robot2-name NAME        Robot2 topic namespace       (default: $ROBOT2_NAME)
  --robot1-driver DRV       Robot1 state driver          (default: $ROBOT1_DRIVER)
  --robot2-driver DRV       Robot2 state driver          (default: $ROBOT2_DRIVER)
  --gripper1-driver DRV     Robot1 gripper driver        (default: $GRIPPER1_DRIVER)
  --gripper2-driver DRV     Robot2 gripper driver        (default: $GRIPPER2_DRIVER)
                            (Use "none" for no gripper.)

Nova mode:
  --nova-env-file FILE      Source NOVA_* vars from FILE before parsing flags
                            (default: ./nova.env if it exists; gitignored by *.env)
  --nova-nats-url URL       NATS server URL              (default: $NOVA_NATS_URL)
  --nova-cell ID            Nova cell id                 (default: $NOVA_CELL)
  --nova-ctrl1 ID           Robot1 Nova controller id    (default: $NOVA_CTRL1)
  --nova-ctrl2 ID           Robot2 Nova controller id    (default: $NOVA_CTRL2)
  --nova-user USER          NATS username                (optional)
  --nova-password PW        NATS password                (optional)
  --nova-creds FILE         NATS JWT credentials file    (optional)

Cameras:
  --camera-source MODE      local | nova                (default: local; auto=nova when --state-source nova)
                            local: launch realsense2_camera on this host.
                            nova : pull WebRTC color streams from the Nova
                                   RealSense app and republish as ROS Image.
  --nova-cam-api-base URL   Nova RealSense API base URL  (default: $NOVA_CAM_API_BASE)
  --nova-cam-serials CSV    Comma-separated Nova camera serials/device_ids
                            (default: auto-discover via GET /api/devices/)
  --nova-cam-stream-types CSV  Nova stream kinds to subscribe (default: $NOVA_CAM_STREAM_TYPES)
  --cam1-serial SN          RealSense serial for cam1    (default: auto)
  --cam2-serial SN          RealSense serial for cam2    (default: auto)
  --cam3-serial SN          RealSense serial for cam3    (default: auto)
  --cam4-serial SN          RealSense serial for cam4    (default: auto)
  --depth                   Enable depth streams + recording (local source only)
  --no-depth                Disable depth                (default)
  --no-cameras              Skip camera launch entirely (record robot state only)
  --camera-fps N            RealSense color/depth fps    (default: $CAMERA_FPS)

Triggers:
  --gui-port N              Web GUI port                 (default: $GUI_PORT)
  --trigger-input TOPIC     DI0 topic to listen on       (default: /<robot1>/digital_input/di0)
  --no-tool-trigger         Don't start the physical-button trigger node
  --episodes-per-session N  Start a new session_<ts> every N saved episodes
                            (default: $EPISODES_PER_SESSION; 0 disables rollover)

  -h, --help                Show this help

Press the physical DI0 on Robot1 OR the Record button at
http://<this-host>:${GUI_PORT}/ to start/stop episodes.

EOF
    exit 0
}

# ── Load Nova env-file (between defaults and CLI flags) ──────
# Precedence: in-script defaults < nova.env < CLI flags.
# Default file: ./nova.env next to this script. Override with
# --nova-env-file <path> (which we pre-scan from $@ here so the
# main parser below still wins).
NOVA_ENV_FILE_DEFAULT="$SCRIPT_DIR/nova.env"
NOVA_ENV_FILE=""
_args=("$@")
for ((i=0; i<${#_args[@]}; i++)); do
    if [[ "${_args[$i]}" == "--nova-env-file" && $((i+1)) -lt ${#_args[@]} ]]; then
        NOVA_ENV_FILE="${_args[$((i+1))]}"
        break
    fi
done
if [[ -z "$NOVA_ENV_FILE" && -f "$NOVA_ENV_FILE_DEFAULT" ]]; then
    NOVA_ENV_FILE="$NOVA_ENV_FILE_DEFAULT"
fi
if [[ -n "$NOVA_ENV_FILE" ]]; then
    if [[ ! -f "$NOVA_ENV_FILE" ]]; then
        echo -e "${RED}--nova-env-file '$NOVA_ENV_FILE' not found${NC}" >&2
        exit 1
    fi
    # Allow comments + blank lines. set -a so plain "KEY=value" lines export.
    set -a
    # shellcheck disable=SC1090
    source "$NOVA_ENV_FILE"
    set +a
fi

# ── Parse args ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --state-source)     STATE_SOURCE="$2"; shift 2 ;;
        --robot1-ip)        ROBOT1_IP="$2"; shift 2 ;;
        --robot2-ip)        ROBOT2_IP="$2"; shift 2 ;;
        --robot1-name)      ROBOT1_NAME="$2"; shift 2 ;;
        --robot2-name)      ROBOT2_NAME="$2"; shift 2 ;;
        --robot1-driver)    ROBOT1_DRIVER="$2"; shift 2 ;;
        --robot2-driver)    ROBOT2_DRIVER="$2"; shift 2 ;;
        --gripper1-driver)  GRIPPER1_DRIVER="$2"; shift 2 ;;
        --gripper2-driver)  GRIPPER2_DRIVER="$2"; shift 2 ;;
        --nova-env-file)    shift 2 ;;   # already handled above
        --nova-nats-url)    NOVA_NATS_URL="$2"; shift 2 ;;
        --nova-cell)        NOVA_CELL="$2"; shift 2 ;;
        --nova-ctrl1)       NOVA_CTRL1="$2"; shift 2 ;;
        --nova-ctrl2)       NOVA_CTRL2="$2"; shift 2 ;;
        --nova-user)        NOVA_NATS_USER="$2"; shift 2 ;;
        --nova-password)    NOVA_NATS_PASSWORD="$2"; shift 2 ;;
        --nova-creds)       NOVA_NATS_CREDS_FILE="$2"; shift 2 ;;
        --cam1-serial)      CAM1_SERIAL="$2"; shift 2 ;;
        --cam2-serial)      CAM2_SERIAL="$2"; shift 2 ;;
        --cam3-serial)      CAM3_SERIAL="$2"; shift 2 ;;
        --cam4-serial)      CAM4_SERIAL="$2"; shift 2 ;;
        --camera-source)    CAMERA_SOURCE="$2"; CAMERA_SOURCE_EXPLICIT=true; shift 2 ;;
        --nova-cam-api-base) NOVA_CAM_API_BASE="$2"; shift 2 ;;
        --nova-cam-serials) NOVA_CAM_SERIALS="$2"; shift 2 ;;
        --nova-cam-stream-types) NOVA_CAM_STREAM_TYPES="$2"; shift 2 ;;
        --nova-port-forward) NOVA_PORT_FORWARD="$2"; shift 2 ;;
        --depth)            RECORD_DEPTH=true; shift ;;
        --no-depth)         RECORD_DEPTH=false; shift ;;
        --no-cameras)       ENABLE_CAMERAS=false; shift ;;
        --camera-fps)       CAMERA_FPS="$2"
                            CAMERA_COLOR_PROFILE="640x480x${CAMERA_FPS}"
                            CAMERA_DEPTH_PROFILE="640x480x${CAMERA_FPS}"
                            shift 2 ;;
        --gui-port)         GUI_PORT="$2"; shift 2 ;;
        --trigger-input)    TRIGGER_INPUT="$2"; shift 2 ;;
        --no-tool-trigger)  ENABLE_TOOL_TRIGGER=false; shift ;;
        --episodes-per-session) EPISODES_PER_SESSION="$2"; shift 2 ;;
        -h|--help)          show_help ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; show_help ;;
    esac
done

# Validate state source and apply Nova-mode overrides.
case "$CAMERA_SOURCE" in
    local|nova) ;;
    *)
        echo -e "${RED}Invalid --camera-source '$CAMERA_SOURCE' (expected: local | nova)${NC}"
        exit 1 ;;
esac

case "$STATE_SOURCE" in
    rtde) ;;
    nova)
        ROBOT1_DRIVER="nova"
        ROBOT2_DRIVER="nova"
        # Nova v2 state stream does not include tool DIs, so a
        # physical DI0 trigger cannot fire. Disable it; the Web GUI
        # button remains available.
        if [[ "$ENABLE_TOOL_TRIGGER" == "true" ]]; then
            echo -e "${YELLOW}  ℹ Nova mode: auto-disabling --tool-trigger (no DI in Nova v2 state)${NC}"
            ENABLE_TOOL_TRIGGER=false
        fi
        # When the state source is Nova and the user hasn't explicitly
        # asked otherwise, pull camera frames from Nova too — that's
        # the whole point of running in Nova mode.
        if [[ "$CAMERA_SOURCE_EXPLICIT" != "true" && "$ENABLE_CAMERAS" == "true" ]]; then
            CAMERA_SOURCE="nova"
            echo -e "${YELLOW}  ℹ Nova mode: auto-selecting --camera-source nova (use --camera-source local to override)${NC}"
        fi
        ;;
    *)
        echo -e "${RED}Invalid --state-source '$STATE_SOURCE' (expected: rtde | nova)${NC}"
        exit 1 ;;
esac

# Default trigger input topic if not overridden.
if [[ -z "$TRIGGER_INPUT" ]]; then
    TRIGGER_INPUT="/${ROBOT1_NAME}/digital_input/di0"
fi

# ── PID tracking ─────────────────────────────────────────────
ROBOT1_PID=""
ROBOT2_PID=""
RECORDER_PID=""
TRIGGER_TOOL_PID=""
TRIGGER_GUI_PID=""
CAMERA_PIDS=()
NOVA_PF_PID=""
CLEANED_UP=false

cleanup() {
    if [[ "$CLEANED_UP" == "true" ]]; then return; fi
    CLEANED_UP=true
    echo ""
    echo -e "${YELLOW}╔═════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  Shutting down all components...    ║${NC}"
    echo -e "${YELLOW}╚═════════════════════════════════════╝${NC}"

    for name_pid in \
        "Recorder:$RECORDER_PID" \
        "Trigger-GUI:$TRIGGER_GUI_PID" \
        "Trigger-Tool:$TRIGGER_TOOL_PID" \
        "Robot2:$ROBOT2_PID" \
        "Robot1:$ROBOT1_PID" \
        "Nova-PortForward:$NOVA_PF_PID"; do
        IFS=: read -r name pid <<< "$name_pid"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo -e "${CYAN}  Stopping $name (PID $pid)...${NC}"
            kill -INT "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            echo -e "${GREEN}    ✓ $name stopped${NC}"
        fi
    done
    for i in "${!CAMERA_PIDS[@]}"; do
        pid="${CAMERA_PIDS[$i]}"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo -e "${CYAN}  Stopping Camera$((i+1)) (PID $pid)...${NC}"
            kill -INT "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            echo -e "${GREEN}    ✓ Camera$((i+1)) stopped${NC}"
        fi
    done
    echo -e "${GREEN}All processes stopped.${NC}"
}

trap cleanup SIGINT SIGTERM EXIT

# ── Banner ────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║     UrEpisodeRecorder — Dual-Robot Recorder         ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  State source:       ${BOLD}${STATE_SOURCE}${NC}"
if [[ "$STATE_SOURCE" == "nova" ]]; then
    echo -e "  Nova NATS:          ${BOLD}${NOVA_NATS_URL}${NC}  cell=${NOVA_CELL}"
    echo -e "  Robot1:             nova-controller=${BOLD}${NOVA_CTRL1}${NC}  ns=/${ROBOT1_NAME}  gripper=${GRIPPER1_DRIVER}@${ROBOT1_IP}"
    echo -e "  Robot2:             nova-controller=${BOLD}${NOVA_CTRL2}${NC}  ns=/${ROBOT2_NAME}  gripper=${GRIPPER2_DRIVER}@${ROBOT2_IP}"
else
    echo -e "  Robot1:             ${BOLD}${ROBOT1_IP}${NC}  ns=/${ROBOT1_NAME}  driver=${ROBOT1_DRIVER}/${GRIPPER1_DRIVER}"
    echo -e "  Robot2:             ${BOLD}${ROBOT2_IP}${NC}  ns=/${ROBOT2_NAME}  driver=${ROBOT2_DRIVER}/${GRIPPER2_DRIVER}"
fi
echo -e "  Motion:             ${GREEN}${BOLD}NONE — read-only observation${NC}"
echo -e "  Camera source:      ${BOLD}${CAMERA_SOURCE}${NC}$([[ "$CAMERA_SOURCE" == "nova" ]] && echo "   api=${NOVA_CAM_API_BASE}")"
if [[ "$RECORD_DEPTH" == "true" ]]; then
    echo -e "  Depth streams:      ${BOLD}ENABLED${NC}"
else
    echo -e "  Depth streams:      ${YELLOW}${BOLD}DISABLED${NC}"
fi
echo -e "  Camera fps:         ${BOLD}${CAMERA_FPS}${NC}"
echo -e "  Trigger (DI):       ${BOLD}${TRIGGER_INPUT}${NC} ($([[ "$ENABLE_TOOL_TRIGGER" == "true" ]] && echo enabled || echo disabled))"
echo -e "  Trigger (GUI):      ${BOLD}http://$(detect_host_ip):${GUI_PORT}/${NC}"
echo ""

# ── Step 1: Network check ────────────────────────────────────
start_nova_port_forward() {
    # Spawn `kubectl port-forward -n <ns> <svc> <local>:<remote>` so the
    # local cluster's NATS ClusterIP becomes reachable on 127.0.0.1.
    # Idempotent: if something already listens on the local port we skip.
    local lp="$NOVA_PF_LOCAL_PORT" rp="$NOVA_PF_REMOTE_PORT"
    if ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${lp}$"; then
        echo -e "${GREEN}  ✓ Nova NATS port-forward already active on 127.0.0.1:${lp}${NC}"
        return 0
    fi
    if ! command -v kubectl >/dev/null 2>&1; then
        echo -e "${YELLOW}  ⚠ kubectl not found — cannot auto port-forward Nova NATS${NC}"
        return 1
    fi
    if [[ ! -r "$NOVA_KUBECONFIG" ]]; then
        echo -e "${YELLOW}  ⚠ kubeconfig $NOVA_KUBECONFIG not readable — cannot port-forward${NC}"
        return 1
    fi
    echo -e "${CYAN}  Starting kubectl port-forward -n $NOVA_PF_NAMESPACE $NOVA_PF_SERVICE ${lp}:${rp} ...${NC}"
    KUBECONFIG="$NOVA_KUBECONFIG" kubectl -n "$NOVA_PF_NAMESPACE" port-forward \
        "$NOVA_PF_SERVICE" "${lp}:${rp}" \
        > "$LOG_DIR/nova_port_forward.log" 2>&1 &
    NOVA_PF_PID=$!
    # Wait up to 5s for the local port to come up.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        sleep 0.5
        if ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${lp}$"; then
            echo -e "${GREEN}  ✓ Nova NATS forwarded to 127.0.0.1:${lp} (PID $NOVA_PF_PID)${NC}"
            return 0
        fi
        if ! kill -0 "$NOVA_PF_PID" 2>/dev/null; then
            echo -e "${RED}  ✗ Nova NATS port-forward died — see $LOG_DIR/nova_port_forward.log${NC}"
            NOVA_PF_PID=""
            return 1
        fi
    done
    echo -e "${YELLOW}  ⚠ Nova NATS port-forward did not bind 127.0.0.1:${lp} in 5s${NC}"
    return 1
}

echo -e "${CYAN}[1/6] Checking network connectivity...${NC}"
if [[ "$STATE_SOURCE" == "nova" ]]; then
    # Extract host from nats://host[:port]
    NOVA_HOST="$(echo "$NOVA_NATS_URL" | sed -E 's#^[a-z]+://([^:/]+).*#\1#')"
    NOVA_IS_LOCAL=false
    case "$NOVA_HOST" in
        127.0.0.1|localhost|::1) NOVA_IS_LOCAL=true ;;
    esac
    # Auto-start a kubectl port-forward when the URL is localhost and
    # the user hasn't disabled it. Explicit "on" forces it regardless.
    case "$NOVA_PORT_FORWARD" in
        on)   start_nova_port_forward || true ;;
        auto) [[ "$NOVA_IS_LOCAL" == "true" ]] && start_nova_port_forward || true ;;
        off)  : ;;
        *)    echo -e "${YELLOW}  ⚠ Unknown --nova-port-forward '$NOVA_PORT_FORWARD' — ignoring${NC}" ;;
    esac
    if [[ "$NOVA_IS_LOCAL" == "true" ]]; then
        echo -e "${GREEN}  ✓ Nova NATS host ${NOVA_HOST} (local)${NC}"
    elif [[ -n "$NOVA_HOST" ]] && ! ping -c 1 -W 2 "$NOVA_HOST" &>/dev/null; then
        echo -e "${RED}  ✗ Cannot reach Nova NATS host ${NOVA_HOST}${NC}"; exit 1
    else
        echo -e "${GREEN}  ✓ Nova NATS host ${NOVA_HOST} reachable${NC}"
    fi
    # Grippers still talk to UR controllers directly via socket port 63352.
    if [[ "$GRIPPER1_DRIVER" != "none" && "$GRIPPER1_DRIVER" != "noop" ]]; then
        if ! ping -c 1 -W 2 "$ROBOT1_IP" &>/dev/null; then
            echo -e "${YELLOW}  ⚠ Cannot reach Robot1 gripper host ${ROBOT1_IP} — gripper telemetry will be unavailable${NC}"
        else
            echo -e "${GREEN}  ✓ Robot1 gripper host ${ROBOT1_IP} reachable${NC}"
        fi
    fi
    if [[ "$GRIPPER2_DRIVER" != "none" && "$GRIPPER2_DRIVER" != "noop" ]]; then
        if ! ping -c 1 -W 2 "$ROBOT2_IP" &>/dev/null; then
            echo -e "${YELLOW}  ⚠ Cannot reach Robot2 gripper host ${ROBOT2_IP} — gripper telemetry will be unavailable${NC}"
        else
            echo -e "${GREEN}  ✓ Robot2 gripper host ${ROBOT2_IP} reachable${NC}"
        fi
    fi
else
    if ! ping -c 1 -W 2 "$ROBOT1_IP" &>/dev/null; then
        echo -e "${RED}  ✗ Cannot reach Robot1 at ${ROBOT1_IP}${NC}"; exit 1
    fi
    echo -e "${GREEN}  ✓ Robot1 ${ROBOT1_IP} reachable${NC}"
    if ! ping -c 1 -W 2 "$ROBOT2_IP" &>/dev/null; then
        echo -e "${RED}  ✗ Cannot reach Robot2 at ${ROBOT2_IP}${NC}"; exit 1
    fi
    echo -e "${GREEN}  ✓ Robot2 ${ROBOT2_IP} reachable${NC}"
fi

# ── Step 2: ROS2 environment ─────────────────────────────────
echo -e "${CYAN}[2/6] Setting up ROS2 environment...${NC}"
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
for _overlay in "$ROOT_DIR/install/setup.bash" "$SCRIPT_DIR/install/setup.bash"; do
    if [[ -f "$_overlay" ]]; then
        source "$_overlay"
        echo -e "${GREEN}  ✓ Sourced overlay: $_overlay${NC}"
        break
    fi
done

echo -e "${CYAN}  Cleaning up stale processes from previous runs...${NC}"
for proc in \
    "episode_recorder.nodes.robot_reader" \
    "episode_recorder.nodes.episode_recorder" \
    "episode_recorder.nodes.trigger_tool_io" \
    "episode_recorder.nodes.trigger_gui" \
    "episode_recorder.nodes.nova_camera_bridge" \
    "rs_launch.py" "realsense2_camera_node"; do
    if pgrep -f "$proc" >/dev/null 2>&1; then
        pkill -f "$proc" 2>/dev/null || true
    fi
done
fuser -k "${GUI_PORT}/tcp" 2>/dev/null || true
sleep 0.5

# ── Step 3: Cameras (local realsense2_camera OR Nova WebRTC bridge) ─
if [[ "$CAMERA_SOURCE" == "nova" ]]; then
    echo -e "${CYAN}[3/6] Starting Nova camera bridge(s)...${NC}"
else
    echo -e "${CYAN}[3/6] Starting RealSense cameras...${NC}"
fi
if [[ "$ENABLE_CAMERAS" != "true" ]]; then
    echo -e "${YELLOW}  ⊘ Cameras disabled (--no-cameras) — recording robot state only${NC}"
    CAMERA_PIDS=()
    CAMERA_NAMES=()
elif [[ "$CAMERA_SOURCE" == "nova" ]]; then
    # ── Nova WebRTC camera bridge ─────────────────────────────
    # Pull color streams from the Nova RealSense app and republish
    # each one as /<camera_name>/<camera_name>/color/image_raw, so
    # the recorder + GUI consume Nova frames transparently.
    echo -e "${CYAN}  Nova RealSense API: ${BOLD}${NOVA_CAM_API_BASE}${NC}"

    # Resolve camera serials.
    if [[ -n "$NOVA_CAM_SERIALS" ]]; then
        # Trim whitespace around comma-separated entries.
        IFS=',' read -r -a NOVA_CAM_SERIAL_ARR <<< "$NOVA_CAM_SERIALS"
        for i in "${!NOVA_CAM_SERIAL_ARR[@]}"; do
            NOVA_CAM_SERIAL_ARR[$i]="$(echo "${NOVA_CAM_SERIAL_ARR[$i]}" | xargs)"
        done
    else
        echo -e "${CYAN}  Auto-discovering cameras from ${NOVA_CAM_API_BASE}/api/devices/ ...${NC}"
        # Best-effort discovery via curl + python (stdlib only).
        DISCOVERY_JSON="$(curl -fsS --max-time 5 \
            "${NOVA_CAM_API_BASE}/api/devices/" 2>/dev/null || true)"
        if [[ -z "$DISCOVERY_JSON" ]]; then
            echo -e "${RED}  ✗ Failed to query Nova devices at ${NOVA_CAM_API_BASE}/api/devices/${NC}"
            exit 1
        fi
        mapfile -t NOVA_CAM_SERIAL_ARR < <(python3 -c '
import json, sys
data = json.loads(sys.stdin.read())
for d in data:
    sid = d.get("device_id") or d.get("serial_number")
    if sid:
        print(sid)
' <<< "$DISCOVERY_JSON")
    fi

    if [[ "${#NOVA_CAM_SERIAL_ARR[@]}" -eq 0 ]]; then
        echo -e "${YELLOW}  ⊘ No Nova cameras found — recording robot state only${NC}"
        CAMERA_PIDS=()
        CAMERA_NAMES=()
    else
        # Cap at MAX_CAMERAS slots (matches IMAGE_TOPICS slot layout).
        if [[ "${#NOVA_CAM_SERIAL_ARR[@]}" -gt "$MAX_CAMERAS" ]]; then
            echo -e "${YELLOW}  ⚠ ${#NOVA_CAM_SERIAL_ARR[@]} Nova cameras found, capping at $MAX_CAMERAS${NC}"
            NOVA_CAM_SERIAL_ARR=("${NOVA_CAM_SERIAL_ARR[@]:0:$MAX_CAMERAS}")
        fi

        # Normalize stream types to a Python-list literal:
        #   "color"          -> [color]
        #   "color,depth"    -> [color,depth]
        STREAM_TYPES_CSV="$NOVA_CAM_STREAM_TYPES"
        STREAM_TYPES_LIST="[${STREAM_TYPES_CSV}]"

        CAMERA_PIDS=()
        CAMERA_NAMES=()
        for i in "${!NOVA_CAM_SERIAL_ARR[@]}"; do
            cam_serial="${NOVA_CAM_SERIAL_ARR[$i]}"
            [[ -z "$cam_serial" ]] && continue
            cam_name="camera$((i+1))"
            cam_log="$LOG_DIR/${cam_name}.log"
            : > "$cam_log"
            # Wrap string values in YAML single-quotes so the ROS 2
            # CLI doesn't auto-type all-digit serials (e.g.
            # "353322270772") as INTEGER and reject the override.
            python3 -u -m episode_recorder.nodes.nova_camera_bridge --ros-args \
                -r __node:="nova_camera_bridge_${cam_name}" \
                -p api_base:="'$NOVA_CAM_API_BASE'" \
                -p device_id:="'$cam_serial'" \
                -p camera_name:="'$cam_name'" \
                -p stream_types:="$STREAM_TYPES_LIST" \
                > "$cam_log" 2>&1 &
            cam_pid=$!
            sleep 1
            if kill -0 "$cam_pid" 2>/dev/null; then
                echo -e "${GREEN}  ✓ ${cam_name} bridge running (PID $cam_pid, device_id=${cam_serial})${NC}"
                CAMERA_PIDS+=("$cam_pid")
                CAMERA_NAMES+=("$cam_name")
            else
                echo -e "${YELLOW}  ⚠ ${cam_name} bridge failed to start — see ${cam_log}${NC}"
            fi
        done

        if [[ "$RECORD_DEPTH" == "true" ]]; then
            echo -e "${YELLOW}  ⚠ --depth has no effect with --camera-source nova (color-only)${NC}"
        fi
    fi
elif ros2 pkg list 2>/dev/null | grep -q realsense2_camera; then
    rs_model_for_pid() {
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
    DETECTED_PORT_SERIALS=()         # sysfs USB-device serial (fallback only)
    DETECTED_RS_SERIALS=()           # librealsense serial (preferred)
    DETECTED_MODELS=()

    # Preferred: ask librealsense itself for serial/model. This is the
    # only reliable identifier across multi-camera launches (sysfs's
    # USB serial differs from the camera's librealsense serial, and
    # usb_port_id triggers a context-wide enumeration in every node,
    # which races and produces "Device or resource busy" failures).
    if command -v rs-enumerate-devices >/dev/null 2>&1; then
        while IFS= read -r line; do
            # Example line: "Intel RealSense D405          353322270772        5.12.14.100"
            [[ "$line" == Intel\ RealSense* ]] || continue
            # Parse: take last 2 fields as fw+serial-ish then split.
            model="$(echo "$line" | awk '{print $3}')"
            serial="$(echo "$line" | awk '{print $4}')"
            [[ -z "$serial" || -z "$model" ]] && continue
            DETECTED_RS_SERIALS+=("$serial")
            DETECTED_MODELS+=("$model")
            DETECTED_PORTS+=("")          # not needed when using serial_no
            DETECTED_PORT_SERIALS+=("$serial")
        done < <(rs-enumerate-devices -s 2>/dev/null)
    fi

    # Fallback: sysfs scan (used only if rs-enumerate-devices found
    # nothing, e.g. when the realsense udev rules aren't installed).
    if [[ "${#DETECTED_RS_SERIALS[@]}" -eq 0 ]]; then
        while IFS= read -r dev; do
            vendor_file="$dev/idVendor"
            product_file="$dev/idProduct"
            [[ -r "$vendor_file" && -r "$product_file" ]] || continue
            vendor="$(cat "$vendor_file" 2>/dev/null)"
            [[ "$vendor" == "8086" ]] || continue
            class="$(cat "$dev/bDeviceClass" 2>/dev/null)"
            [[ "$class" == "09" ]] && continue
            product="$(cat "$product_file" 2>/dev/null)"
            model="$(rs_model_for_pid "$product")"
            [[ -z "$model" ]] && continue
            port="$(basename "$dev")"
            sysfs_serial="$(cat "$dev/serial" 2>/dev/null | tr -d '[:space:]')"
            DETECTED_PORTS+=("$port")
            DETECTED_PORT_SERIALS+=("${sysfs_serial:-?}")
            DETECTED_RS_SERIALS+=("")        # unknown via sysfs alone
            DETECTED_MODELS+=("$model")
        done < <(ls -d /sys/bus/usb/devices/*/ 2>/dev/null)
    fi

    USB_DEVICE_COUNT="${#DETECTED_MODELS[@]}"
    # Per-slot serial/port/model arrays sized to MAX_CAMERAS.
    CAM_SERIALS=("$CAM1_SERIAL" "$CAM2_SERIAL" "$CAM3_SERIAL" "$CAM4_SERIAL")
    CAM_PORTS=(""  ""  ""  "")
    CAM_MODELS=("" "" "" "")
    # Map each slot without an explicit serial to the next auto-detected device.
    # Prefer the librealsense serial when available so multi-camera launches
    # don't race on USB enumeration.
    _auto_idx=0
    for ((s=0; s<MAX_CAMERAS; s++)); do
        if [[ -z "${CAM_SERIALS[$s]}" && "$_auto_idx" -lt "$USB_DEVICE_COUNT" ]]; then
            rs_ser="${DETECTED_RS_SERIALS[$_auto_idx]:-}"
            if [[ -n "$rs_ser" ]]; then
                CAM_SERIALS[$s]="$rs_ser"
            else
                CAM_PORTS[$s]="${DETECTED_PORTS[$_auto_idx]}"
            fi
            CAM_MODELS[$s]="${DETECTED_MODELS[$_auto_idx]:-}"
            _auto_idx=$((_auto_idx+1))
        fi
    done

    if [[ "$USB_DEVICE_COUNT" -ge 1 ]]; then
        msg="  Detected $USB_DEVICE_COUNT RealSense device(s):"
        for i in "${!DETECTED_MODELS[@]}"; do
            id_field="${DETECTED_RS_SERIALS[$i]:-${DETECTED_PORTS[$i]:-?}}"
            msg+=" [${DETECTED_MODELS[$i]}=${id_field}]"
        done
        echo -e "${CYAN}${msg}${NC}"
    else
        echo -e "${YELLOW}  ⚠ No RealSense devices detected${NC}"
    fi

    # Hardware-reset ALL RealSense devices in one batch, then wait
    # for every serial to reappear, then enforce a single common
    # settle period before any node opens a sensor.
    #
    # Rationale: doing this per-camera right before each launch
    # leaves the FIRST camera with effectively zero XU settle time,
    # so its node aborts at `depth_module.enable_auto_exposure` with
    # "Device or resource busy". Cameras launched later (cam3/cam4)
    # benefit from the accumulated delay of earlier resets and come
    # up cleanly. The batched approach gives every camera equal
    # (and ample) FW init time.
    reset_all_rs_cameras() {
        local serials=("$@")
        [[ "${#serials[@]}" -eq 0 ]] && return 0
        echo -e "${BLUE}    · Hardware-resetting ${#serials[@]} RealSense device(s) (FW XU recovery) ...${NC}"
        python3 - "${serials[@]}" <<'PY' 2>&1 | sed 's/^/      /' || true
import sys
try:
    import pyrealsense2 as rs
except ImportError:
    print("pyrealsense2 not installed; skipping HW reset")
    sys.exit(0)
targets = set(sys.argv[1:])
ctx = rs.context()
for d in ctx.query_devices():
    try:
        sn = d.get_info(rs.camera_info.serial_number)
    except RuntimeError:
        continue
    if sn not in targets:
        continue
    try:
        d.hardware_reset()
        print(f"hardware_reset issued for {sn}")
    except Exception as e:
        print(f"hardware_reset {sn} failed: {e}")
PY
        # Wait up to 15s for ALL serials to come back on the USB bus.
        local deadline=$((SECONDS + 15))
        local missing=1
        while [[ $SECONDS -lt $deadline ]]; do
            sleep 1
            local enumerated
            enumerated=$(rs-enumerate-devices -s 2>/dev/null | awk '{print $4}') || true
            missing=0
            for s in "${serials[@]}"; do
                if ! grep -qx "$s" <<<"$enumerated"; then
                    missing=1
                    break
                fi
            done
            [[ $missing -eq 0 ]] && break
        done
        if [[ $missing -ne 0 ]]; then
            echo "      ⚠ not all cameras re-enumerated within 15 s after reset"
        fi
        # Common settle so every camera's XU is fully initialised
        # before the first node opens it. The FIRST camera launched
        # after the batch reset gets the least cumulative XU-init
        # time, and 4 s was empirically not enough on this rig — the
        # D405 firmware on 5.17 needs ≥ 6 s of quiet post-reset
        # before its XU endpoint responds to enable_auto_exposure.
        echo -e "${BLUE}    · Settling cameras (7 s) ...${NC}"
        sleep 7
    }

    launch_camera() {
        local name="$1" serial="$2" port="$3" model="${4:-}"
        local logfile="$LOG_DIR/${name}.log"
        # Base args (color-only). Depth-related params are added
        # ONLY when depth recording is enabled — the realsense2
        # driver opens & probes the depth sensor whenever ANY
        # `depth_module.*` param is set, which triggers the
        # depth_module.enable_auto_exposure XU-busy abort on FW 5.17
        # D4xx devices. Keeping the depth sensor untouched is the
        # most reliable way to avoid that race entirely.
        local args=(
            camera_namespace:="$name"
            camera_name:="$name"
            enable_color:=true
            enable_depth:="$RECORD_DEPTH"
            pointcloud.enable:=false
            rgb_camera.color_profile:="${CAMERA_COLOR_PROFILE}"
            reconnect_timeout:=6.0
            wait_for_device_timeout:=10.0
        )
        if [[ "$RECORD_DEPTH" == "true" ]]; then
            args+=(
                align_depth.enable:=true
                depth_module.depth_profile:="${CAMERA_DEPTH_PROFILE}"
            )
        fi
        if [[ -n "$serial" ]]; then
            args+=("serial_no:=_$serial")
        elif [[ -n "$port" ]]; then
            args+=("usb_port_id:=$port")
        fi
        ros2 launch realsense2_camera rs_launch.py "${args[@]}" \
            >"$logfile" 2>&1 </dev/null &
    }

    # Wait for a camera to actually be ready to publish frames.
    #
    # Detection is LOG-based, not topic-based:
    #
    #   * "RealSense Node Is Up!" → sensors opened, streaming started.
    #   * "Error starting device"  → FW XU-busy abort; node will not
    #                                stream until reset+relaunch.
    #
    # We deliberately do NOT use `ros2 topic echo` here:
    #   - On sensor_msgs/Image, `ros2 topic echo --once` is unreliable
    #     across distros — some versions exit immediately with a
    #     formatting error, others wait for the topic to appear and
    #     give a misleading 124 even when the node is up.
    #   - The realsense2_camera node creates publishers BEFORE the
    #     sensor opens, so topic existence is not a stream-readiness
    #     signal anyway.
    wait_for_camera() {
        local name="$1" timeout="${2:-25}"
        local logfile="$LOG_DIR/${name}.log"
        local deadline=$((SECONDS + timeout))
        while [[ $SECONDS -lt $deadline ]]; do
            if [[ -f "$logfile" ]]; then
                if grep -q "Error starting device" "$logfile" 2>/dev/null; then
                    return 2   # known FW XU-busy abort → caller may retry
                fi
                if grep -q "RealSense Node Is Up" "$logfile" 2>/dev/null; then
                    return 0
                fi
            fi
            sleep 0.5
        done
        return 1
    }

    CAMERA_PIDS=()
    CAMERA_NAMES=()

    # Batch FW reset of every detected RealSense BEFORE launching any
    # node. See reset_all_rs_cameras for the rationale (per-camera
    # resets give the first camera ~0 s of XU settle and it aborts).
    BATCH_SERIALS=()
    for ((s=0; s<MAX_CAMERAS; s++)); do
        [[ -n "${CAM_SERIALS[$s]:-}" ]] && BATCH_SERIALS+=("${CAM_SERIALS[$s]}")
    done
    reset_all_rs_cameras "${BATCH_SERIALS[@]}"

    for ((s=0; s<MAX_CAMERAS; s++)); do
        cam_serial="${CAM_SERIALS[$s]}"
        cam_port="${CAM_PORTS[$s]}"
        cam_model="${CAM_MODELS[$s]}"
        if [[ -z "$cam_serial" && -z "$cam_port" ]]; then
            continue   # no device for this slot
        fi
        cam_name="camera$((s+1))"
        # Launch + verify with retry. The FW 5.17 XU-busy abort
        # ("Error starting device: depth_module.enable_auto_exposure")
        # is a timing race that hits a different camera each run; if
        # it strikes, kill this node, hardware-reset the device, and
        # try again. Up to 4 attempts before giving up.
        cam_pid=""
        for attempt in 1 2 3 4; do
            # Truncate the log so wait_for_camera doesn't see a
            # stale "Error starting device" / "Node Is Up" from a
            # previous attempt.
            : > "$LOG_DIR/${cam_name}.log"
            launch_camera "$cam_name" "$cam_serial" "$cam_port" "$cam_model"
            cam_pid=$!
            # IMPORTANT: call wait_for_camera inside an `if` so its
            # non-zero return (1 = timeout, 2 = XU-busy abort) does
            # NOT trip `set -eo pipefail` and kill the whole script
            # before the retry path runs. Bare function calls that
            # return non-zero are caught by `set -e`; `if`-wrapped
            # ones are not.
            wait_rc=0
            if ! wait_for_camera "$cam_name" 25; then
                wait_rc=$?
            fi
            if [[ $wait_rc -eq 0 ]]; then
                echo -e "${GREEN}    · ${cam_name} streaming (RealSense Node Is Up)${NC}"
                break
            fi
            if [[ $wait_rc -eq 2 ]]; then
                echo -e "${YELLOW}    · ${cam_name} hit FW XU-busy abort (attempt ${attempt}/4) — resetting & retrying${NC}"
                kill "$cam_pid" 2>/dev/null || true
                wait "$cam_pid" 2>/dev/null || true
                cam_pid=""
                if [[ -n "$cam_serial" ]]; then
                    reset_all_rs_cameras "$cam_serial" || true
                    # Extra settle on retry: the previously-launched
                    # cameras are streaming on the same USB hub, so
                    # the freshly-reset device needs a longer quiet
                    # period for its XU endpoint to come up.
                    echo -e "${BLUE}    · Extended settle (${attempt}x6 s) for retry...${NC}"
                    sleep $((attempt * 6))
                fi
                continue
            fi
            echo -e "${YELLOW}    · ${cam_name} did not signal readiness within 25 s — continuing anyway${NC}"
            break
        done
        CAMERA_PIDS+=("$cam_pid")
        CAMERA_NAMES+=("$cam_name")
        # Inter-camera grace: librealsense's USB enumeration is
        # serialised at the kernel/firmware level, so opening the
        # next camera too soon after the previous one is the most
        # common trigger of the XU-busy abort on D405 hubs. 4 s is
        # the minimum that consistently brings up 4 cameras here.
        sleep 4
    done

    for i in "${!CAMERA_PIDS[@]}"; do
        cam_pid="${CAMERA_PIDS[$i]}"
        cam_name="${CAMERA_NAMES[$i]}"
        cam_model="${CAM_MODELS[$i]:-}"
        if kill -0 "$cam_pid" 2>/dev/null; then
            echo -e "${GREEN}  ✓ ${cam_name} running (PID $cam_pid${cam_model:+, $cam_model})${NC}"
        else
            echo -e "${YELLOW}  ⚠ ${cam_name} failed to start — see $LOG_DIR/${cam_name}.log${NC}"
            CAMERA_PIDS[$i]=""
        fi
    done
    if [[ "${#CAMERA_PIDS[@]}" -eq 0 ]]; then
        echo -e "${YELLOW}  ⊘ No RealSense devices launched${NC}"
    fi
else
    echo -e "${YELLOW}  ⊘ realsense2_camera package not found — cameras skipped${NC}"
    CAMERA_PIDS=()
    CAMERA_NAMES=()
fi

# ── Build image-topic lists for the recorder ─────────────────
IMAGE_TOPICS=()
DEPTH_TOPICS=()
for i in "${!CAMERA_PIDS[@]}"; do
    cam_pid="${CAMERA_PIDS[$i]}"
    cam_name="${CAMERA_NAMES[$i]}"
    [[ -z "$cam_pid" ]] && continue
    IMAGE_TOPICS+=("/${cam_name}/${cam_name}/color/image_raw")
    [[ "$RECORD_DEPTH" == "true" ]] && DEPTH_TOPICS+=("/${cam_name}/${cam_name}/depth/image_rect_raw")
done
# Comma-separated for ROS 2 string-array CLI: "[a,b,c]"
join_csv() { local IFS=','; echo "$*"; }
if [[ "${#IMAGE_TOPICS[@]}" -gt 0 ]]; then
    IMAGE_LIST="[$(join_csv "${IMAGE_TOPICS[@]}")]"
else
    # ROS 2 CLI cannot infer the type of an empty array, so pass a
    # single empty string. The recorder strips empty entries.
    IMAGE_LIST="['']"
fi
if [[ "${#DEPTH_TOPICS[@]}" -gt 0 ]]; then
    DEPTH_LIST="[$(join_csv "${DEPTH_TOPICS[@]}")]"
else
    DEPTH_LIST="['']"
fi

# ── Step 4: Robot readers ────────────────────────────────────
echo -e "${CYAN}[4/6] Starting robot readers (read-only)...${NC}"
cd "$SCRIPT_DIR"

# Build per-robot extra args. In Nova mode, pass the NATS connection
# + controller id; the gripper still uses robot_ip directly. Empty
# credential values are skipped because rclpy rejects "-p key:=".
common_nova_args=(
    -p nats_url:="$NOVA_NATS_URL"
    -p nova_cell:="$NOVA_CELL"
)
[[ -n "$NOVA_NATS_USER"        ]] && common_nova_args+=(-p nats_user:="$NOVA_NATS_USER")
[[ -n "$NOVA_NATS_PASSWORD"    ]] && common_nova_args+=(-p nats_password:="$NOVA_NATS_PASSWORD")
[[ -n "$NOVA_NATS_CREDS_FILE"  ]] && common_nova_args+=(-p nats_creds_file:="$NOVA_NATS_CREDS_FILE")

robot1_extra=()
robot2_extra=()
if [[ "$STATE_SOURCE" == "nova" ]]; then
    robot1_extra=("${common_nova_args[@]}" -p nova_controller:="$NOVA_CTRL1" -p gripper_ip:="$ROBOT1_IP")
    robot2_extra=("${common_nova_args[@]}" -p nova_controller:="$NOVA_CTRL2" -p gripper_ip:="$ROBOT2_IP")
fi

python3 -u -m episode_recorder.nodes.robot_reader --ros-args \
    -r __node:="robot_reader_${ROBOT1_NAME}" \
    -p name:="$ROBOT1_NAME" \
    -p robot_driver:="$ROBOT1_DRIVER" \
    -p gripper_driver:="$GRIPPER1_DRIVER" \
    -p robot_ip:="$ROBOT1_IP" \
    "${robot1_extra[@]}" \
    > "$LOG_DIR/${ROBOT1_NAME}.log" 2>&1 &
ROBOT1_PID=$!
sleep 3
if ! kill -0 "$ROBOT1_PID" 2>/dev/null; then
    echo -e "${RED}  ✗ Robot1 reader failed — see $LOG_DIR/${ROBOT1_NAME}.log${NC}"; exit 1
fi
echo -e "${GREEN}  ✓ Robot1 reader running (PID $ROBOT1_PID)${NC}"

python3 -u -m episode_recorder.nodes.robot_reader --ros-args \
    -r __node:="robot_reader_${ROBOT2_NAME}" \
    -p name:="$ROBOT2_NAME" \
    -p robot_driver:="$ROBOT2_DRIVER" \
    -p gripper_driver:="$GRIPPER2_DRIVER" \
    -p robot_ip:="$ROBOT2_IP" \
    "${robot2_extra[@]}" \
    > "$LOG_DIR/${ROBOT2_NAME}.log" 2>&1 &
ROBOT2_PID=$!
sleep 3
if ! kill -0 "$ROBOT2_PID" 2>/dev/null; then
    echo -e "${RED}  ✗ Robot2 reader failed — see $LOG_DIR/${ROBOT2_NAME}.log${NC}"; exit 1
fi
echo -e "${GREEN}  ✓ Robot2 reader running (PID $ROBOT2_PID)${NC}"

# ── Step 5: Trigger nodes ────────────────────────────────────
echo -e "${CYAN}[5/6] Starting trigger nodes...${NC}"

# Web GUI (always — primary fallback when no physical button is wired)
python3 -u -m episode_recorder.nodes.trigger_gui --ros-args \
    -p port:="$GUI_PORT" \
    -p image_topics:="$IMAGE_LIST" \
    -p robot_namespaces:="[$ROBOT1_NAME,$ROBOT2_NAME]" \
    > "$LOG_DIR/trigger_gui.log" 2>&1 &
TRIGGER_GUI_PID=$!
sleep 2
if kill -0 "$TRIGGER_GUI_PID" 2>/dev/null; then
    echo -e "${GREEN}  ✓ GUI trigger running — http://$(detect_host_ip):${GUI_PORT}/  (PID $TRIGGER_GUI_PID)${NC}"
else
    echo -e "${YELLOW}  ⚠ GUI trigger failed to start — see $LOG_DIR/trigger_gui.log${NC}"
    TRIGGER_GUI_PID=""
fi

# Tool-IO trigger (physical DI0 button) — optional
if [[ "$ENABLE_TOOL_TRIGGER" == "true" ]]; then
    python3 -u -m episode_recorder.nodes.trigger_tool_io --ros-args \
        -p input_topic:="$TRIGGER_INPUT" \
        > "$LOG_DIR/trigger_tool_io.log" 2>&1 &
    TRIGGER_TOOL_PID=$!
    sleep 1
    if kill -0 "$TRIGGER_TOOL_PID" 2>/dev/null; then
        echo -e "${GREEN}  ✓ Tool-IO trigger listening on ${TRIGGER_INPUT}  (PID $TRIGGER_TOOL_PID)${NC}"
    else
        echo -e "${YELLOW}  ⚠ Tool-IO trigger failed to start — see $LOG_DIR/trigger_tool_io.log${NC}"
        TRIGGER_TOOL_PID=""
    fi
else
    echo -e "${YELLOW}  ⊘ Tool-IO trigger disabled (--no-tool-trigger)${NC}"
fi

# ── Step 6: Episode recorder ─────────────────────────────────
echo -e "${CYAN}[6/6] Starting episode recorder...${NC}"
RECORDER_LOG="$LOG_DIR/episode_recorder.log"
echo -e "${CYAN}    Recorder log: $RECORDER_LOG${NC}"
python3 -u -m episode_recorder.nodes.episode_recorder --ros-args \
    -p root:="$SCRIPT_DIR/recordings_lerobot" \
    -p repo_id:="local/ur5_dual" \
    -p fps:="$CAMERA_FPS" \
    -p robot_namespaces:="[$ROBOT1_NAME,$ROBOT2_NAME]" \
    -p image_topics:="$IMAGE_LIST" \
    -p depth_topics:="$DEPTH_LIST" \
    -p episodes_per_session:="$EPISODES_PER_SESSION" \
    > "$RECORDER_LOG" 2>&1 &
RECORDER_PID=$!
sleep 2
if ! kill -0 "$RECORDER_PID" 2>/dev/null; then
    echo -e "${RED}  ✗ Episode recorder failed to start — see $RECORDER_LOG${NC}"; exit 1
fi
echo -e "${GREEN}  ✓ Episode recorder running (PID $RECORDER_PID)${NC}"
echo -e "${GREEN}    Dataset: $SCRIPT_DIR/recordings_lerobot/${NC}"

echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║  System running                                     ║${NC}"
echo -e "${GREEN}${BOLD}║                                                      ║${NC}"
if [[ "$ENABLE_TOOL_TRIGGER" == "true" ]]; then
    echo -e "${GREEN}${BOLD}║  Press DI0 on Robot1 OR click Record at:            ║${NC}"
else
    echo -e "${GREEN}${BOLD}║  Click Record at:                                   ║${NC}"
fi
echo -e "${GREEN}${BOLD}║  $(printf '%-52s' "http://$(detect_host_ip):${GUI_PORT}/") ║${NC}"
echo -e "${GREEN}${BOLD}║                                                      ║${NC}"
echo -e "${GREEN}${BOLD}║  Press Ctrl+C to shut down everything               ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Wait on the recorder PID in the foreground so Ctrl+C triggers cleanup.
wait "$RECORDER_PID"
