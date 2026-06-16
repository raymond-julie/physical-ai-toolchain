#!/usr/bin/env bash
# Read-only pre-flight sanity check. Safe to run repeatedly.
set +e

echo "--- whoami/host ---"
whoami
hostname

echo "--- sudo NOPASSWD ---"
sudo -n true >/dev/null 2>&1 && echo "SUDO_OK" || echo "SUDO_FAIL"

echo "--- OS ---"
. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME}"

echo "--- kernel ---"
uname -r

echo "--- cpu cores / mem / disk root ---"
echo "cores=$(nproc)"
free -h | awk '/^Mem:/ {print "mem_total="$2" used="$3" avail="$7}'
df -h / | awk 'NR==2 {print "root_total="$2" used="$3" avail="$4}'

echo "--- swap (must be off for kubelet) ---"
swap_active=$(swapon --show 2>/dev/null)
if [ -z "$swap_active" ]; then echo "SWAP_OFF"; else echo "SWAP_ON"; echo "$swap_active"; fi

echo "--- /etc/fstab swap entries (uncommented) ---"
grep -E '^[^#].*\sswap\s' /etc/fstab 2>/dev/null && echo "FSTAB_SWAP_PRESENT" || echo "FSTAB_SWAP_CLEAN"

echo "--- kernel modules (br_netfilter, overlay) ---"
lsmod | grep -E '^(br_netfilter|overlay)\b' || echo "MODULES_NOT_LOADED"

echo "--- sysctl (ip_forward, bridge-nf-call-iptables) ---"
sysctl -n net.ipv4.ip_forward net.bridge.bridge-nf-call-iptables net.bridge.bridge-nf-call-ip6tables 2>/dev/null

echo "--- containerd / kubeadm / kubelet versions ---"
containerd --version 2>/dev/null || echo "containerd_MISSING"
kubeadm version -o short 2>/dev/null || echo "kubeadm_MISSING"
kubelet --version 2>/dev/null || echo "kubelet_MISSING"

echo "--- containerd status ---"
systemctl is-active containerd 2>/dev/null || echo "containerd_INACTIVE"

echo "--- listening ports of interest (6443/2379/2380/10250/10257/10259/30080) ---"
ss -ltn 2>/dev/null | awk 'NR==1 || $4 ~ /:(6443|2379|2380|10250|10257|10259|30080)$/ {print}'

echo "--- prior kubernetes state ---"
if [ -d /etc/kubernetes ] && [ -n "$(ls -A /etc/kubernetes 2>/dev/null)" ]; then
  echo "K8S_STATE_PRESENT"
  ls /etc/kubernetes/
else
  echo "NO_PRIOR_K8S"
fi
systemctl is-active kubelet 2>/dev/null || echo "kubelet_inactive"

echo "--- GPU / nvidia driver ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
  echo "NO_NVIDIA_SMI"
fi

echo "--- nvidia-container-toolkit presence ---"
dpkg -l nvidia-container-toolkit 2>/dev/null | awk '/^ii/ {print $2" "$3}' || echo "nvct_not_installed"

echo "--- UFW firewall ---"
sudo -n ufw status 2>/dev/null | head -3 || echo "no_ufw_or_inactive"

echo "--- time sync (clocks must agree across nodes) ---"
timedatectl 2>/dev/null | grep -E 'System clock|NTP|Time zone'
date -u +%s

echo "--- locale ---"
locale 2>/dev/null | grep -E '^LANG='

echo "--- /tmp writable ---"
touch /tmp/.osmo_check && rm /tmp/.osmo_check && echo "TMP_OK"

echo "--- home writable (~/osmo-deployment target) ---"
mkdir -p ~/osmo-deployment/.probe && rmdir ~/osmo-deployment/.probe && echo "HOME_OK"

echo "--- /dev/shm size (OSMO expects 24GiB via Kyverno; just info) ---"
df -h /dev/shm | awk 'NR==2 {print "shm="$2}'

echo "--- DNS resolve (github.com, nvcr.io, registry.k8s.io) ---"
for h in github.com nvcr.io registry.k8s.io quay.io; do
  getent hosts $h >/dev/null 2>&1 && echo "DNS_OK $h" || echo "DNS_FAIL $h"
done

echo "--- outbound 443 to common registries ---"
for ep in github.com:443 nvcr.io:443 registry.k8s.io:443 quay.io:443; do
  timeout 5 bash -c "</dev/tcp/${ep/:/\/}" >/dev/null 2>&1 && echo "NET_OK $ep" || echo "NET_FAIL $ep"
done
