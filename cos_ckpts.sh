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

COS_URI="cos://sft-backup-1451541705/weiyun-sync/temp/output/mimo_v2_5_exp_0/epoch_0/"
LOCAL_PATH="/afs-private/workspaces/wangweiyun/cos/sft-backup/weiyun-sync/temp/output/mimo_v2_5_exp_0/epoch_0/"

./coscli sync "${COS_URI}" "${LOCAL_PATH}" -r \
  --bucket-type COS \
  -i $COS_SECRET_ID \
  -k $COS_SECRET_KEY \
  -e "${COS_ENDPOINT}" \
  -p https \
  --part-size 32 \
  --thread-num 8 \
  --routines 64 \
  --err-retry-num 10
