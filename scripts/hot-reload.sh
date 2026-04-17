#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${EXO_API_PORT:-52415}"
TAKEOVER_PORT="${EXO_RELOAD_TAKEOVER_PORT:-1}"
RELOAD_MODE="${EXO_RELOAD_MODE:-full}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. install uv first."
  exit 1
fi

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python venv not found at ${PYTHON_BIN}"
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import watchfiles" >/dev/null 2>&1; then
  echo "watchfiles missing; installing via uv pip..."
  uv pip install --python "${PYTHON_BIN}" watchfiles
fi

if ! "${PYTHON_BIN}" -c "import watchfiles" >/dev/null 2>&1; then
  echo "watchfiles install failed for ${PYTHON_BIN}"
  exit 1
fi

EXO_ENV=(
  EXO_FAST_SYNCH=on
  EXO_PREFILL_STEP_SIZE=512
  EXO_KV_CACHE_BITS=2
  EXO_CACHE_GROUP_SIZE=32
  EXO_MAX_TOKENS=4096
  EXO_NO_BATCH=1
  EXO_REQUIRE_READY_INSTANCE=1
  EXO_USE_TOTAL_MEMORY_FOR_PLACEMENT=1
  EXO_SKIP_PLACEMENT_MEMORY_CHECK=1
  EXO_DISABLE_JACCL=1
)

EXO_ARGS=("$@")
WATCH_PATHS=(
  "${ROOT_DIR}/src/exo"
  "${ROOT_DIR}/scripts"
)

if [[ "${RELOAD_MODE}" == "api-only" ]]; then
  EXO_ARGS=(--no-worker "$@")
  WATCH_PATHS=(
    "${ROOT_DIR}/src/exo/api"
    "${ROOT_DIR}/scripts"
  )
fi

QUOTED_ARGS=()
if [[ ${#EXO_ARGS[@]} -gt 0 ]]; then
  for arg in "${EXO_ARGS[@]}"; do
    QUOTED_ARGS+=("$(printf '%q' "${arg}")")
  done
fi

CMD_SUFFIX="${QUOTED_ARGS[*]-}"
CMD="cd \"${ROOT_DIR}\" && env ${EXO_ENV[*]} uv run exo ${CMD_SUFFIX}"

echo "Hot reload enabled. Watching: ${WATCH_PATHS[*]}"
echo "Run command: ${CMD}"
if [[ "${RELOAD_MODE}" == "api-only" ]]; then
  echo "Reload mode: api-only (this process starts EXO with --no-worker)"
fi

child_pid=""
watcher_pid=""
reload_flag=""

cleanup() {
  if [[ -n "${watcher_pid}" ]] && kill -0 "${watcher_pid}" 2>/dev/null; then
    kill "${watcher_pid}" || true
    wait "${watcher_pid}" || true
  fi
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill "${child_pid}" || true
    wait "${child_pid}" || true
  fi
  if [[ -n "${reload_flag}" ]]; then
    rm -f "${reload_flag}" || true
  fi
}

trap cleanup EXIT INT TERM

port_pids="$(lsof -tiTCP:${API_PORT} -sTCP:LISTEN || true)"
if [[ -n "${port_pids}" ]]; then
  if [[ "${TAKEOVER_PORT}" == "1" || "${TAKEOVER_PORT}" == "true" || "${TAKEOVER_PORT}" == "yes" || "${TAKEOVER_PORT}" == "on" ]]; then
    echo "port ${API_PORT} in use, taking over pid(s): ${port_pids}"
    for pid in ${port_pids}; do
      if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" || true
      fi
    done
    sleep 0.2
    still_pids="$(lsof -tiTCP:${API_PORT} -sTCP:LISTEN || true)"
    if [[ -n "${still_pids}" ]]; then
      echo "failed to free port ${API_PORT}; still occupied by: ${still_pids}"
      exit 1
    fi
  else
    echo "port ${API_PORT} is already in use by pid(s): ${port_pids}"
    echo "set EXO_RELOAD_TAKEOVER_PORT=1 to auto-kill and take over this port"
    exit 1
  fi
fi

while true; do
  reload_flag="$(mktemp "${TMPDIR:-/tmp}/exo-reload-flag.XXXXXX")"
  rm -f "${reload_flag}"

  if [[ ${#EXO_ARGS[@]} -gt 0 ]]; then
    (
      cd "${ROOT_DIR}"
      env "${EXO_ENV[@]}" uv run exo "${EXO_ARGS[@]}"
    ) &
  else
    (
      cd "${ROOT_DIR}"
      env "${EXO_ENV[@]}" uv run exo
    ) &
  fi
  child_pid=$!
  echo "started exo pid=${child_pid}"

  (
    "${PYTHON_BIN}" -c 'from pathlib import Path; from watchfiles import watch, PythonFilter; import sys; flag = Path(sys.argv[1]); next(watch(*sys.argv[2:], watch_filter=PythonFilter())); flag.write_text("reload\n", encoding="utf-8")' \
      "${reload_flag}" "${WATCH_PATHS[@]}"
  ) &
  watcher_pid=$!

  while true; do
    if [[ -f "${reload_flag}" ]]; then
      echo "change detected, restarting exo..."
      if kill -0 "${child_pid}" 2>/dev/null; then
        kill "${child_pid}" || true
      fi
      wait "${child_pid}" || true
      if [[ -n "${watcher_pid}" ]] && kill -0 "${watcher_pid}" 2>/dev/null; then
        kill "${watcher_pid}" || true
      fi
      wait "${watcher_pid}" || true
      child_pid=""
      watcher_pid=""
      rm -f "${reload_flag}" || true
      reload_flag=""
      break
    fi

    if ! kill -0 "${child_pid}" 2>/dev/null; then
      wait "${child_pid}" || true
      echo "exo exited unexpectedly, restarting in 1s..."
      if [[ -n "${watcher_pid}" ]] && kill -0 "${watcher_pid}" 2>/dev/null; then
        kill "${watcher_pid}" || true
      fi
      wait "${watcher_pid}" || true
      child_pid=""
      watcher_pid=""
      rm -f "${reload_flag}" || true
      reload_flag=""
      sleep 1
      break
    fi

    sleep 0.2
  done
done
