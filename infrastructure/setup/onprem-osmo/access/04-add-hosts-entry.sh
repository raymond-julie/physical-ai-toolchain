#!/usr/bin/env bash
#
# 04-add-hosts-entry.sh
#
# Idempotently adds `127.0.0.1 quick-start.osmo` to the system hosts file
# so the browser will resolve the Osmo ingress hostname to the local end
# of the SSH tunnel opened by 03-tunnel-osmo.sh.
#
# Cross-platform:
#   * Linux/macOS  -> /etc/hosts (requires sudo).
#   * Windows      -> C:\Windows\System32\drivers\etc\hosts (requires
#                     elevated shell). When run from Git Bash / WSL we
#                     re-exec ourselves under PowerShell with -Verb RunAs
#                     so the UAC prompt appears.
#
# Re-running is safe: if the entry already exists it's left alone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ENTRY_IP="127.0.0.1"
ENTRY_HOST="${OSMO_INGRESS_HOST}"
ENTRY_LINE="${ENTRY_IP} ${ENTRY_HOST}"

case "$(uname -s)" in
  Linux*|Darwin*)
    HOSTS_FILE="/etc/hosts"
    if grep -qE "^[[:space:]]*${ENTRY_IP}[[:space:]]+${ENTRY_HOST}([[:space:]]|$)" "${HOSTS_FILE}"; then
      echo "Hosts entry already present in ${HOSTS_FILE}."
      exit 0
    fi
    echo "Adding '${ENTRY_LINE}' to ${HOSTS_FILE} (sudo required)"
    echo "${ENTRY_LINE}" | sudo tee -a "${HOSTS_FILE}" >/dev/null
    echo "Done."
    ;;

  MINGW*|MSYS*|CYGWIN*)
    # Running inside Git Bash / MSYS on Windows.
    HOSTS_FILE="/c/Windows/System32/drivers/etc/hosts"
    if [[ -r "${HOSTS_FILE}" ]] && \
       grep -qE "^[[:space:]]*${ENTRY_IP}[[:space:]]+${ENTRY_HOST}([[:space:]]|$)" "${HOSTS_FILE}"; then
      echo "Hosts entry already present in ${HOSTS_FILE}."
      exit 0
    fi

    echo "Hosts file edits on Windows require Administrator."
    echo "Launching elevated PowerShell to add '${ENTRY_LINE}'..."

    # Build a PowerShell one-liner that:
    #   - opens the hosts file
    #   - appends the entry only if not already present
    PS_SCRIPT="\$h = \"\$env:WINDIR\\System32\\drivers\\etc\\hosts\"; \
\$line = '${ENTRY_LINE}'; \
\$existing = Get-Content -LiteralPath \$h -ErrorAction SilentlyContinue; \
if (\$existing -match '^\\s*${ENTRY_IP}\\s+${ENTRY_HOST}(\\s|\$)') { \
  Write-Host 'Hosts entry already present.'; \
} else { \
  Add-Content -LiteralPath \$h -Value \$line; \
  Write-Host ('Added: ' + \$line); \
} \
Read-Host 'Press Enter to close'"

    powershell.exe -NoProfile -Command \
      "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-Command',\"${PS_SCRIPT}\""

    echo "Elevated window launched. Approve the UAC prompt to apply."
    ;;

  *)
    echo "Unsupported OS: $(uname -s)" >&2
    exit 1
    ;;
esac
