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
COS_URI="cos://wangweiyun-1306757789/backup-data/agentic_code/raw_resampled_1_2_3_4_with_mask/"

# ./coscli ls "${COS_URI}" --bucket-type COS -e "${COS_ENDPOINT}" -p https

./coscli sync /afs-private/workspaces/wangweiyun/data/260702/raw_resampled_1_2_3_4_with_mask/ "${COS_URI}" -r \
  --bucket-type COS \
  -e "${COS_ENDPOINT}" \
  -p https \
  --part-size 32 \
  --thread-num 8 \
  --routines 64 \
  --err-retry-num 10 \
  --snapshot-path ./cos_snapshot/backup-data/agentic_code/raw_resampled_1_2_3_4_with_mask/.coscli_snapshot \
  --fail-output-path ./cos_snapshot/backup-data/agentic_code/raw_resampled_1_2_3_4_with_mask/coscli_failed \
  --process-log-path ./cos_snapshot/backup-data/agentic_code/raw_resampled_1_2_3_4_with_mask/coscli_logs
