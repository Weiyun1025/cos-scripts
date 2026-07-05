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
COS_URI="cos://raw-lake-1306757789/github-data-bj/260701/"

# ./coscli ls "${COS_URI}" --bucket-type COS -e "${COS_ENDPOINT}" -p https

./coscli sync "${COS_URI}" ./cpt-data/github-data-bj -r \
  --bucket-type COS \
  -e "${COS_ENDPOINT}" \
  -p https \
  --part-size 32 \
  --thread-num 8 \
  --routines 64 \
  --err-retry-num 10 \
  --snapshot-path ./cos_snapshot/cpt-data/github-data-bj/.coscli_snapshot \
  --fail-output-path ./cos_snapshot/cpt-data/github-data-bj/coscli_failed \
  --process-log-path ./cos_snapshot/cpt-data/github-data-bj/coscli_logs
