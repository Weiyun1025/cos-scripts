#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source ./.env
  set +a
fi

COS_REGION="${COS_REGION:-ap-beijing}"
COS_ENDPOINT="${COS_ENDPOINT:-https://cos.${COS_REGION}.myqcloud.com}"
COS_ENDPOINT="${COS_ENDPOINT#http://}"
COS_ENDPOINT="${COS_ENDPOINT#https://}"

COS_URI="cos://sft-backup-1451541705/weiyun-sync/data/agentic_code/260720/"
LOCAL_PATH="/afs-private/workspaces/wangweiyun/data/agentic_code/260720/"

./coscli sync "${LOCAL_PATH}" "${COS_URI}" -r \
  --bucket-type COS \
  -i $COS_SECRET_ID \
  -k $COS_SECRET_KEY \
  -e "${COS_ENDPOINT}" \
  -p https \
  --part-size 32 \
  --thread-num 8 \
  --routines 64 \
  --err-retry-num 10 \
  --snapshot-path ./cos_snapshot/sft-backup/weiyun-sync/data/agentic_code/.coscli_snapshot \
  --fail-output-path ./cos_snapshot/sft-backup/weiyun-sync/data/agentic_code/coscli_failed \
  --process-log-path ./cos_snapshot/sft-backup/weiyun-sync/data/agentic_code/coscli_logs
