#!/usr/bin/env bash
for _ in $(seq 1 40); do
  r=$(kubectl get pods -n osmo --no-headers 2>/dev/null | awk '{print $2}' | awk -F/ '{if($1!=$2) print}' | wc -l)
  t=$(kubectl get pods -n osmo --no-headers 2>/dev/null | wc -l)
  echo "[$(date +%T)] not-ready=$r / $t"
  if [ "$r" -eq 0 ] && [ "$t" -gt 0 ]; then
    echo ALL_READY
    break
  fi
  sleep 15
done
echo ---
kubectl get pods -n osmo
