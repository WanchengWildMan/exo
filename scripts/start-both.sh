#!/usr/bin/env bash
# 同时启动两台 Mac Mini 的 exo，日志重定向到本机 logs/ 目录
# 用法: ./scripts/start-both.sh
#   - M4 (本机) 后台运行，日志写入 logs/mac-mini-m4.log
#   - M2 (远程) 通过 ssh 启动，日志写入 M2 本机的 logs/mac-mini-m2.log
#     在 M4 上通过 /Volumes/xiewancheng/exo/logs/mac-mini-m2.log 访问（无需 symlink）
#
# 停止: Ctrl+C 或 ./scripts/start-both.sh stop

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
M4_LOG="${LOG_DIR}/mac-mini-m4.log"
# M2 日志：M2 写自己本机的路径，M4 通过挂载卷访问
M2_LOG="/Volumes/xiewancheng/exo/logs/mac-mini-m2.log"

REMOTE_HOST="xiewancheng@mac-mini-2.local"
REMOTE_EXO_DIR="/Users/xiewancheng/exo"
REMOTE_LOG="${REMOTE_EXO_DIR}/logs/mac-mini-m2.log"

mkdir -p "${LOG_DIR}"

stop_all() {
    echo ">>> 停止本机 M4 exo..."
    pkill -f "uv run exo" 2>/dev/null || true

    echo ">>> 停止远程 M2 exo..."
    ssh "${REMOTE_HOST}" "pkill -f 'uv run exo'" 2>/dev/null || true

    echo ">>> 已停止"
}

if [[ "${1:-}" == "stop" ]]; then
    stop_all
    exit 0
fi

# --- 先停止旧进程 ---
echo ">>> 停止旧的 exo 进程..."
pkill -f "uv run exo" 2>/dev/null || true
sleep 1

# --- 启动远程 M2 ---
# ssh -t 分配 tty 让你输密码
# 远程端用 nohup 启动 exo 并将日志写到远端的 logs/mac-mini-m2.log
# 该文件通过 /Volumes/xiewancheng/exo/logs/ 可在本机访问
echo ""
echo ">>> 启动远程 M2 (${REMOTE_HOST})..."
echo ">>> 需要输入密码，输完后远程 exo 将自动在后台启动"
echo ""

ssh -t "${REMOTE_HOST}" "
    export PATH=/opt/homebrew/bin:\$PATH
    pkill -f 'uv run exo' 2>/dev/null || true
    sleep 1
    mkdir -p ${REMOTE_EXO_DIR}/logs
    cd ${REMOTE_EXO_DIR}
    nohup bash scripts/start-my.sh fg > ${REMOTE_LOG} 2>&1 &
    echo \">>> M2 exo 已在后台启动, PID=\$!\"
    echo \">>> 远程日志: ${REMOTE_LOG}\"
    sleep 2
    echo \">>> M2 进程状态:\"
    ps aux | grep '[u]v run exo' || echo '(未找到进程)'
"

echo ""
echo ">>> 启动本机 M4..."

# 本机后台运行
cd "${ROOT_DIR}"
nohup bash scripts/start-my.sh fg > "${M4_LOG}" 2>&1 &
M4_PID=$!

echo ">>> M4 本机 PID: ${M4_PID}"
sleep 2

# 确认本机进程是否在运行
if kill -0 "${M4_PID}" 2>/dev/null; then
    echo ">>> M4 运行中"
else
    echo ">>> ⚠️  M4 启动失败，请检查 ${M4_LOG}"
fi

# --- 等待 exo API 就绪后重启 cli-proxy-api ---
# cli-proxy 在 exo 启动前探测后端会认为 exo 不可用，导致 exo-qwen-9B 别名丢失
# 需要在 exo 就绪后重启 cli-proxy 让它重新注册
EXO_API="http://localhost:52415/v1/models"
CLI_PROXY_BIN="/usr/local/bin/cli-proxy-api"
CLI_PROXY_LOG="${ROOT_DIR}/logs/cliproxy.log"

if [[ -x "${CLI_PROXY_BIN}" ]]; then
    echo ""
    echo ">>> 等待 exo API 就绪后重启 cli-proxy-api..."
    for i in $(seq 1 30); do
        if curl -s "${EXO_API}" | grep -q '"id"'; then
            echo ">>> exo API 就绪 (${i}s)"
            pkill -f cli-proxy-api 2>/dev/null || true
            sleep 1
            nohup "${CLI_PROXY_BIN}" > "${CLI_PROXY_LOG}" 2>&1 &
            echo ">>> cli-proxy-api 已重启 PID=$!"
            break
        fi
        sleep 1
    done
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  两台机器已启动（后台运行）              ║"
echo "║  M4 日志: logs/mac-mini-m4.log           ║"
echo "║  M2 日志: /Volumes/xiewancheng/exo/logs/ ║"
echo "╠══════════════════════════════════════════╣"
echo "║  tail -f logs/mac-mini-m4.log            ║"
echo "║  停止: scripts/start-both.sh stop        ║"
echo "╚══════════════════════════════════════════╝"
