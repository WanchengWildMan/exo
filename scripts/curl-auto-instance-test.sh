#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:52415}"
MODEL_ID="${MODEL_ID:-mlx-community/Qwen3.5-9B-MLX-4bit}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-你是一个中文助手，请始终使用简体中文直接回答。}"
PROMPT="${PROMPT:-请直接回复：OK}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0}"
SHARDING="${SHARDING:-Pipeline}"
INSTANCE_META="${INSTANCE_META:-MlxRing}"
CREATE_INSTANCE="${CREATE_INSTANCE:-1}"

cleanup() {
  rm -f "${STATE_FILE:-}" "${PLACEMENT_FILE:-}" "${CREATE_BODY_FILE:-}" "${CHAT_BODY_FILE:-}" "${RESP_FILE:-}"
}
trap cleanup EXIT

STATE_FILE="$(mktemp)"
PLACEMENT_FILE="$(mktemp)"
CREATE_BODY_FILE="$(mktemp)"
CHAT_BODY_FILE="$(mktemp)"
RESP_FILE="$(mktemp)"

if [[ "${CREATE_INSTANCE}" == "1" ]]; then
  curl -fsS "${BASE_URL}/state" >"${STATE_FILE}"

  NODE_COUNT="$(/Users/xiewancheng/exo/.venv/bin/python -c 'import json,sys; s=json.load(open(sys.argv[1])); print(max(1, len((s.get("topology") or {}).get("nodes") or [])))' "${STATE_FILE}")"

  echo "cluster nodes: ${NODE_COUNT}"
  echo "model: ${MODEL_ID}"

  SELECTED_MIN_NODES=0
  for ((n=NODE_COUNT; n>=1; n--)); do
    if curl -fsS -G "${BASE_URL}/instance/placement" \
      --data-urlencode "model_id=${MODEL_ID}" \
      --data-urlencode "sharding=${SHARDING}" \
      --data-urlencode "instance_meta=${INSTANCE_META}" \
      --data-urlencode "min_nodes=${n}" >"${PLACEMENT_FILE}"; then
      SELECTED_MIN_NODES="${n}"
      break
    fi
  done

  if [[ "${SELECTED_MIN_NODES}" -eq 0 ]]; then
    echo "failed: no valid placement from min_nodes=${NODE_COUNT} down to 1"
    exit 1
  fi

  echo "selected min_nodes: ${SELECTED_MIN_NODES}"

  /Users/xiewancheng/exo/.venv/bin/python -c 'import json,sys; p=json.load(open(sys.argv[1])); json.dump({"instance": p}, open(sys.argv[2], "w"))' "${PLACEMENT_FILE}" "${CREATE_BODY_FILE}"

  curl -fsS -X POST "${BASE_URL}/instance" \
    -H "Content-Type: application/json" \
    --data-binary @"${CREATE_BODY_FILE}" >/dev/null
else
  echo "skip instance creation (CREATE_INSTANCE=0)"
fi

/Users/xiewancheng/exo/.venv/bin/python -c 'import json,sys; model,system_prompt,prompt,max_tokens,temp=sys.argv[1:6]; json.dump({"model":model,"messages":[{"role":"system","content":system_prompt},{"role":"user","content":prompt}],"max_tokens":int(max_tokens),"temperature":float(temp),"stream":False}, open(sys.argv[6], "w"))' "${MODEL_ID}" "${SYSTEM_PROMPT}" "${PROMPT}" "${MAX_TOKENS}" "${TEMPERATURE}" "${CHAT_BODY_FILE}"

for _ in $(seq 1 30); do
  curl -sS -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary @"${CHAT_BODY_FILE}" >"${RESP_FILE}"

  if /Users/xiewancheng/exo/.venv/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); msg=((d.get("error") or {}).get("message") or ""); raise SystemExit(0 if "No instance found" in msg else 1)' "${RESP_FILE}"; then
    sleep 1
    continue
  fi

  cat "${RESP_FILE}"
  exit 0
done

echo "failed: instance was created but completion still returns not-ready after retries"
cat "${RESP_FILE}"
exit 2
