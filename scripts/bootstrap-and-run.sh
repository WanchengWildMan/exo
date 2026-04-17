#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_SCRIPT="${ROOT_DIR}/scripts/start-my.sh"

print_usage() {
  cat <<'EOF'
Usage:
  scripts/bootstrap-and-run.sh [all|install|start] [fg|bg|tail|stop|status] [extra exo args...]

Examples:
  scripts/bootstrap-and-run.sh
  scripts/bootstrap-and-run.sh all bg
  scripts/bootstrap-and-run.sh start fg --inference-engine mlx
  scripts/bootstrap-and-run.sh install

Environment:
  EXO_INSTALL_MACMON=1  Install pinned macmon on macOS (optional, default off)
EOF
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_brew_pkg() {
  local pkg="$1"
  if has_cmd "$pkg"; then
    return 0
  fi
  if ! has_cmd brew; then
    echo "missing dependency: ${pkg}, and brew is not available"
    return 1
  fi
  brew install "$pkg"
}

ensure_uv() {
  if has_cmd uv; then
    return 0
  fi

  if has_cmd brew; then
    brew install uv
    return 0
  fi

  if has_cmd curl; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [[ -d "${HOME}/.local/bin" ]]; then
      export PATH="${HOME}/.local/bin:${PATH}"
    fi
  fi

  if ! has_cmd uv; then
    echo "failed to install uv; install it manually and retry"
    return 1
  fi
}

ensure_rust_toolchain() {
  if ! has_cmd rustup; then
    if ! has_cmd curl; then
      echo "missing rustup and curl; cannot auto-install rust toolchain"
      return 1
    fi
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  fi

  if [[ -f "${HOME}/.cargo/env" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.cargo/env"
  fi

  rustup toolchain list | grep -q '^nightly' || rustup toolchain install nightly
}

ensure_macmon_if_requested() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    return 0
  fi

  if [[ "${EXO_INSTALL_MACMON:-0}" != "1" ]]; then
    return 0
  fi

  if ! has_cmd cargo; then
    echo "cargo not found; skip macmon install"
    return 0
  fi

  cargo install --git https://github.com/swiftraccoon/macmon \
    --rev 9154d234f763fbeffdcb4135d0bbbaf80609699b \
    macmon \
    --force
}

install_deps_and_build() {
  ensure_uv
  ensure_brew_pkg node
  ensure_rust_toolchain
  ensure_macmon_if_requested

  cd "${ROOT_DIR}"
  uv sync

  cd "${ROOT_DIR}/dashboard"
  npm install
  npm run build
}

start_exo() {
  if [[ ! -x "${START_SCRIPT}" ]]; then
    echo "start script not found or not executable: ${START_SCRIPT}"
    return 1
  fi

  local mode="fg"
  local known_mode="${1:-}"
  if [[ -n "${known_mode}" && "${known_mode}" =~ ^(fg|bg|tail|stop|status)$ ]]; then
    mode="${known_mode}"
    shift
  fi

  "${START_SCRIPT}" "${mode}" "$@"
}

case "${ACTION}" in
  all)
    install_deps_and_build
    start_exo "$@"
    ;;
  install)
    install_deps_and_build
    ;;
  start)
    start_exo "$@"
    ;;
  -h|--help|help)
    print_usage
    ;;
  *)
    print_usage
    exit 2
    ;;
esac
