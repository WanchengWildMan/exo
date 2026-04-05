#!/usr/bin/env bash

set -euo pipefail

# Ensure homebrew binaries (uv, etc.) are in PATH (needed for SSH/nohup sessions)
export PATH="/opt/homebrew/bin:${PATH}"

MODE="${1:-fg}"
if [[ $# -gt 0 ]]; then
	shift
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/tmp"
LOG_FILE="${LOG_DIR}/exo-start.log"
PID_FILE="${LOG_DIR}/exo-start.pid"

mkdir -p "${LOG_DIR}"

EXO_ENV=(
	EXO_FAST_SYNCH=on
	EXO_PREFILL_STEP_SIZE=16
	EXO_KV_CACHE_BITS=2
	EXO_CACHE_GROUP_SIZE=32
	EXO_MAX_KV_SIZE=1500
	EXO_KEEP_KV_SIZE=64
	EXO_PREFILL_EVAL_INTERVAL=1
	EXO_MAX_TOKENS=2048
	EXO_NO_BATCH=1
	EXO_REQUIRE_READY_INSTANCE=1
	EXO_USE_TOTAL_MEMORY_FOR_PLACEMENT=1
	EXO_SKIP_PLACEMENT_MEMORY_CHECK=1
	EXO_MEMORY_THRESHOLD=0.60
	EXO_PREFILL_MIN_AVAILABLE_GB=0.5
	EXO_PREFILL_SYNC_TIMEOUT=60
	EXO_SKILLS_DESC_MAX_CHARS=30
	EXO_MAX_SYSTEM_PROMPT_CHARS=2000
	EXO_DISABLE_JACCL=1
	# oMLX内存管理集成 (P0/P1/P2)
	EXO_PREFILL_MAX_METAL_GB=0
	EXO_GENERATION_MEMORY_CHECK_INTERVAL=64
	EXO_SSD_CACHE_ENABLED=1
	EXO_SSD_CACHE_MAX_GB=10
)

ensure_api_port_free() {
	local pids
	pids="$(lsof -tiTCP:52415 -sTCP:LISTEN 2>/dev/null || true)"
	if [[ -n "${pids}" ]]; then
		echo "port 52415 is already in use by: ${pids}"
		echo "stop the existing exo process before starting a new one"
		exit 1
	fi
}

start_bg() {
	ensure_api_port_free
	if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
		echo "exo is already running in background (pid $(cat "${PID_FILE}"))"
		return 0
	fi

	(
		cd "${ROOT_DIR}"
		env "${EXO_ENV[@]}" uv run exo "$@"
	) >"${LOG_FILE}" 2>&1 &

	echo $! >"${PID_FILE}"
	echo "started exo in background, pid=$(cat "${PID_FILE}")"
	echo "log: ${LOG_FILE}"
}

start_fg() {
	ensure_api_port_free
	cd "${ROOT_DIR}"
	exec env "${EXO_ENV[@]}" uv run exo "$@"
}

stop_bg() {
	if [[ ! -f "${PID_FILE}" ]]; then
		echo "no pid file found: ${PID_FILE}"
		return 0
	fi

	local pid
	pid="$(cat "${PID_FILE}")"
	if kill -0 "${pid}" 2>/dev/null; then
		kill "${pid}"
		echo "stopped exo background process: ${pid}"
	else
		echo "process not running: ${pid}"
	fi
	rm -f "${PID_FILE}"
}

show_status() {
	if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
		echo "background running, pid=$(cat "${PID_FILE}")"
		echo "log: ${LOG_FILE}"
	else
		echo "background not running"
	fi
}

tail_log() {
	if [[ ! -f "${LOG_FILE}" ]]; then
		echo "log file not found: ${LOG_FILE}"
		return 1
	fi
	tail -f "${LOG_FILE}"
}

case "${MODE}" in
	fg)
		start_fg "$@"
		;;
	bg)
		start_bg "$@"
		;;
	reload)
		exec "${ROOT_DIR}/scripts/hot-reload.sh" "$@"
		;;
	reload-api)
		EXO_RELOAD_MODE=api-only exec "${ROOT_DIR}/scripts/hot-reload.sh" "$@"
		;;
	tail)
		tail_log
		;;
	stop)
		stop_bg
		;;
	status)
		show_status
		;;
	*)
		echo "Usage: $0 [fg|bg|reload|reload-api|tail|stop|status] [extra exo args...]"
		exit 2
		;;
esac
