#!/usr/bin/env bash
#
# 04-install-cli.sh
# Install the NVIDIA OSMO CLI on the current machine.
#
# Run this on any machine that needs to interact with the OSMO cluster
# (developer workstations, CI runners, etc.).
#
# Usage:
#   bash scripts/04-install-cli.sh [--login-url URL]

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

main() {
  local login_url=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --login-url)
        login_url="$2"
        shift 2
        ;;
      --help|-h)
        usage
        ;;
      *)
        shift
        ;;
    esac
  done

  log_step "4" "Installing NVIDIA OSMO CLI"

  install_osmo_cli
  verify_cli

  if [[ -n "${login_url}" ]]; then
    login_osmo "${login_url}"
  else
    log_info "To log in, run:"
    log_info "  osmo login http://<OSMO_HOSTNAME> --method=dev --username=testuser"
  fi

  log_success "OSMO CLI installation complete."
}

usage() {
  cat <<EOF
Usage: ${0##*/} [OPTIONS]

Options:
  --login-url URL    OSMO service URL to log in after installation
  --help, -h         Show this help message

Examples:
  ${0##*/}
  ${0##*/} --login-url http://quick-start.osmo
EOF
  exit 1
}

##############################################################################
# CLI Installation
##############################################################################

install_osmo_cli() {
  if command -v osmo &>/dev/null; then
    log_info "OSMO CLI already installed: $(osmo version 2>/dev/null || echo 'unknown')"
    log_info "Reinstalling to ensure latest version..."
  fi

  log_info "Downloading and installing OSMO CLI..."
  curl -fsSL https://raw.githubusercontent.com/NVIDIA/OSMO/refs/heads/main/install.sh | bash

  # Ensure the CLI is in PATH
  if [[ -d "${HOME}/.osmo/bin" ]]; then
    if [[ ":${PATH}:" != *":${HOME}/.osmo/bin:"* ]]; then
      export PATH="${HOME}/.osmo/bin:${PATH}"
      log_info "Added ~/.osmo/bin to PATH for this session."
      log_info "Add to your shell profile for persistence:"
      log_info "  echo 'export PATH=\"\$HOME/.osmo/bin:\$PATH\"' >> ~/.bashrc"
    fi
  fi
}

##############################################################################
# Verification
##############################################################################

verify_cli() {
  if command -v osmo &>/dev/null; then
    log_success "OSMO CLI installed: $(osmo version 2>/dev/null || echo 'available')"
  else
    log_warning "OSMO CLI not found in PATH. You may need to restart your shell."
  fi
}

##############################################################################
# Login
##############################################################################

login_osmo() {
  local url="$1"

  log_info "Logging in to OSMO at ${url}..."
  osmo login "${url}" --method=dev --username=testuser

  log_success "Logged in to OSMO successfully."
}

main "$@"
