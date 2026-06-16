#!/usr/bin/env bash
# Destructive: wipes Kubernetes state so kubeadm init/join can run cleanly.
set +e
echo "--- kubeadm reset ---"
sudo kubeadm reset -f 2>&1 | tail -20
echo "--- stopping kubelet/containerd ---"
sudo systemctl stop kubelet 2>/dev/null
sudo systemctl stop containerd 2>/dev/null
echo "--- removing state dirs ---"
sudo rm -rf /etc/kubernetes /var/lib/etcd /var/lib/kubelet /var/lib/dockershim /var/run/kubernetes ~/.kube /etc/cni/net.d
echo "--- flushing iptables ---"
sudo iptables -F 2>/dev/null
sudo iptables -t nat -F 2>/dev/null
sudo iptables -t mangle -F 2>/dev/null
sudo iptables -X 2>/dev/null
sudo ipvsadm -C 2>/dev/null
echo "--- restarting containerd ---"
sudo systemctl start containerd
sudo systemctl is-active containerd
echo "--- verify cleanup ---"
if [ -d /etc/kubernetes ] && [ -n "$(ls -A /etc/kubernetes 2>/dev/null)" ]; then
  echo "K8S_STATE_STILL_PRESENT"
  ls /etc/kubernetes/
else
  echo "CLEAN_OK"
fi
ss -ltn 2>/dev/null | awk '$4 ~ /:(6443|2379|2380|10250|10257|10259)$/ {print "STILL_LISTENING "$0}'
echo "DONE"
