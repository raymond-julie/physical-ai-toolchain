#!/usr/bin/env bash
WF=osmo-smoke-test-1
for _ in $(seq 1 120); do
  line=$(/usr/local/bin/osmo workflow query "$WF" 2>/dev/null | tail -20)
  phase=$(echo "$line" | grep -oE '(Pending|Running|Succeeded|Failed|Error|Completed)' | tail -1)
  echo "[$(date +%T)] phase=$phase"
  if [ "$phase" = "Succeeded" ] || [ "$phase" = "Failed" ] || [ "$phase" = "Error" ] || [ "$phase" = "Completed" ]; then
    echo "=== FINAL STATUS ==="
    /usr/local/bin/osmo workflow query "$WF"
    break
  fi
  sleep 5
done
