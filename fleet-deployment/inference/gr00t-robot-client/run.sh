#!/usr/bin/env bash
# Convenience launcher for the GR00T dual-arm inference client.
#
# Opens a port-forward to the in-cluster policy server (ClusterIP :5555) and
# runs the client. Edit the variables below for your rig, then:
#
#   ./run.sh            # dry run (default, robots do NOT move)
#   ./run.sh --execute  # stream actions to the real arms (asks for confirmation)
#
# Any extra args are passed straight through to robot_inference_client.py.
set -o errexit -o nounset -o pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Edit these for your setup.
CAMERA_URL="${CAMERA_URL:-http://192.168.1.10:8000}"   # ur-camera-streamer
ROBOT1_IP="${ROBOT1_IP:-192.168.1.80}"                  # left follower
ROBOT2_IP="${ROBOT2_IP:-192.168.1.90}"                  # right follower
TASK="${TASK:-pick up the red block and place it in the box}"
# The four streamer camera ids in color_0..color_3 order (use real serials).
read -r -a CAMERA_IDS <<< "${CAMERA_IDS:-cam_high cam_low cam_left_wrist cam_right_wrist}"
KUBE_NS="${KUBE_NS:-default}"
POLICY_SVC="${POLICY_SVC:-svc/gr00t-gr00t-inference}"
POLICY_PORT="${POLICY_PORT:-5555}"

PF_PID=""
cleanup() { [[ -n "$PF_PID" ]] && kill "$PF_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "Port-forwarding ${POLICY_SVC} ${POLICY_PORT} -> localhost:${POLICY_PORT}"
kubectl port-forward -n "$KUBE_NS" "$POLICY_SVC" "${POLICY_PORT}:${POLICY_PORT}" \
    > /tmp/gr00t-policy-pf.log 2>&1 &
PF_PID=$!
sleep 3

python3 "${HERE}/robot_inference_client.py" \
    --policy-host 127.0.0.1 --policy-port "$POLICY_PORT" \
    --camera-url "$CAMERA_URL" \
    --camera-ids "${CAMERA_IDS[@]}" \
    --robot1-ip "$ROBOT1_IP" --robot2-ip "$ROBOT2_IP" \
    --task "$TASK" \
    "$@"
