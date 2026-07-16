#!/usr/bin/env python3
"""Estimate FinePDFs token totals from sampled shards."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import orjson
from transformers import AutoTokenizer


BUCKET_RE = re.compile(r"^block_section_(\d+)-(\d+)$")
TOKENS_PER_BUCKET_UNIT = 1_000
_TOKENIZER = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate FinePDFs token counts from sampled JSONL shards."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--files-per-bucket", type=int, default=3)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    _TOKENIZER.model_max_length = sys.maxsize


def update_extremes(items: list[dict], item: dict, reverse: bool) -> None:
    items.append(item)
    items.sort(key=lambda value: value["tokens"], reverse=reverse)
    del items[5:]


def process_file(path_s: str, lower_tokens: int, upper_tokens: int) -> dict:
    if _TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")

    path = Path(path_s)
    lengths: list[int] = []
    total_tokens = 0
    total_line_bytes = 0
    below = 0
    within = 0
    above = 0
    shortest: list[dict] = []
    longest: list[dict] = []

    with path.open("rb", buffering=1024 * 1024) as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = orjson.loads(line)
            text = item["text"]
            if not isinstance(text, str):
                raise TypeError(f"{path}:{line_no}: text is not a string")

            token_count = len(
                _TOKENIZER.encode(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                    verbose=False,
                )
            )
            line_bytes = len(line)
            lengths.append(token_count)
            total_tokens += token_count
            total_line_bytes += line_bytes

            if token_count < lower_tokens:
                below += 1
            elif token_count >= upper_tokens:
                above += 1
            else:
                within += 1

            location = {
                "file": path.name,
                "line": line_no,
                "tokens": token_count,
                "jsonl_bytes": line_bytes,
            }
            update_extremes(shortest, location, reverse=False)
            update_extremes(longest, location, reverse=True)

    stat_bytes = path.stat().st_size
    if total_line_bytes != stat_bytes:
        raise RuntimeError(
            f"{path}: read {total_line_bytes} non-empty bytes but stat reports {stat_bytes}"
        )

    return {
        "path": path_s,
        "bytes": stat_bytes,
        "records": len(lengths),
        "tokens": total_tokens,
        "below": below,
        "within": within,
        "above": above,
        "lengths": lengths,
        "shortest": shortest,
        "longest": longest,
    }


def percentile(sorted_values: list[int], percentile_value: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def bootstrap_ratio_interval(file_results: list[dict], seed: int) -> tuple[float, float]:
    if len(file_results) < 2:
        return math.nan, math.nan
    rng = random.Random(seed)
    ratios = []
    for _ in range(10_000):
        draw = [rng.choice(file_results) for _ in file_results]
        ratios.append(sum(item["tokens"] for item in draw) / sum(item["bytes"] for item in draw))
    ratios.sort()
    return percentile(ratios, 0.025), percentile(ratios, 0.975)


def discover_buckets(data_root: Path, sample_root: Path, files_per_bucket: int, seed: int) -> list[dict]:
    buckets = []
    for data_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        match = BUCKET_RE.fullmatch(data_dir.name)
        if match is None:
            continue
        lower, upper = (int(value) for value in match.groups())
        sample_dir = sample_root / data_dir.name
        full_files = sorted(data_dir.glob("*.jsonl"))
        sample_files = sorted(sample_dir.glob("*.jsonl"))
        if not full_files:
            raise RuntimeError(f"no JSONL files under {data_dir}")
        if len(sample_files) < files_per_bucket:
            raise RuntimeError(
                f"{sample_dir} has {len(sample_files)} files; need {files_per_bucket}"
            )
        rng = random.Random(f"{seed}\0{data_dir.name}")
        selected_files = sorted(rng.sample(sample_files, files_per_bucket))
        buckets.append(
            {
                "name": data_dir.name,
                "lower_tokens": lower * TOKENS_PER_BUCKET_UNIT,
                "upper_tokens": upper * TOKENS_PER_BUCKET_UNIT,
                "full_files": full_files,
                "selected_files": selected_files,
            }
        )
    if not buckets:
        raise RuntimeError(f"no bucket directories found under {data_root}")
    return buckets


def main() -> int:
    args = parse_args()
    if args.files_per_bucket <= 0:
        raise ValueError("--files-per-bucket must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    buckets = discover_buckets(
        args.data_root, args.sample_root, args.files_per_bucket, args.seed
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer_metadata = {
        "path": str(args.tokenizer.resolve()),
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "model_max_length": tokenizer.model_max_length,
        "add_special_tokens": False,
    }
    del tokenizer

    tasks = {}
    started = time.time()
    workers = min(args.workers, sum(len(bucket["selected_files"]) for bucket in buckets))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(str(args.tokenizer),),
    ) as executor:
        for bucket in buckets:
            for path in bucket["selected_files"]:
                future = executor.submit(
                    process_file,
                    str(path),
                    bucket["lower_tokens"],
                    bucket["upper_tokens"],
                )
                tasks[future] = bucket["name"]

        results_by_bucket: dict[str, list[dict]] = {bucket["name"]: [] for bucket in buckets}
        for completed, future in enumerate(as_completed(tasks), start=1):
            bucket_name = tasks[future]
            result = future.result()
            results_by_bucket[bucket_name].append(result)
            print(
                f"[{completed}/{len(tasks)}] {bucket_name}/{Path(result['path']).name}: "
                f"records={result['records']:,} tokens={result['tokens']:,}",
                flush=True,
            )

    bucket_reports = []
    for bucket_index, bucket in enumerate(buckets):
        file_results = sorted(
            results_by_bucket[bucket["name"]], key=lambda result: result["path"]
        )
        lengths = sorted(length for result in file_results for length in result["lengths"])
        sample_bytes = sum(result["bytes"] for result in file_results)
        sample_tokens = sum(result["tokens"] for result in file_results)
        sample_records = sum(result["records"] for result in file_results)
        full_bytes = sum(path.stat().st_size for path in bucket["full_files"])
        below = sum(result["below"] for result in file_results)
        within = sum(result["within"] for result in file_results)
        above = sum(result["above"] for result in file_results)
        token_per_byte = sample_tokens / sample_bytes
        estimated_tokens = token_per_byte * full_bytes
        ratio_low, ratio_high = bootstrap_ratio_interval(
            file_results, args.seed + bucket_index
        )
        file_ratios = [result["tokens"] / result["bytes"] for result in file_results]
        shortest = sorted(
            (item for result in file_results for item in result["shortest"]),
            key=lambda item: item["tokens"],
        )[:5]
        longest = sorted(
            (item for result in file_results for item in result["longest"]),
            key=lambda item: item["tokens"],
            reverse=True,
        )[:5]
        bucket_reports.append(
            {
                "bucket": bucket["name"],
                "expected_token_range": {
                    "lower_inclusive": bucket["lower_tokens"],
                    "upper_exclusive": bucket["upper_tokens"],
                },
                "full_file_count": len(bucket["full_files"]),
                "full_bytes": full_bytes,
                "sample_file_count": len(file_results),
                "sample_files": [Path(result["path"]).name for result in file_results],
                "sample_bytes": sample_bytes,
                "sample_storage_fraction": sample_bytes / full_bytes,
                "sample_records": sample_records,
                "sample_tokens": sample_tokens,
                "tokens_per_jsonl_byte": token_per_byte,
                "file_ratio_min": min(file_ratios),
                "file_ratio_max": max(file_ratios),
                "file_ratio_cv": (
                    statistics.stdev(file_ratios) / statistics.mean(file_ratios)
                    if len(file_ratios) > 1
                    else None
                ),
                "estimated_tokens": round(estimated_tokens),
                "estimated_tokens_bootstrap_95pct": {
                    "lower": round(ratio_low * full_bytes) if not math.isnan(ratio_low) else None,
                    "upper": round(ratio_high * full_bytes) if not math.isnan(ratio_high) else None,
                },
                "length_tokens": {
                    "min": lengths[0],
                    "p01": round(percentile(lengths, 0.01), 2),
                    "p05": round(percentile(lengths, 0.05), 2),
                    "p50": round(percentile(lengths, 0.50), 2),
                    "mean": round(statistics.fmean(lengths), 2),
                    "p95": round(percentile(lengths, 0.95), 2),
                    "p99": round(percentile(lengths, 0.99), 2),
                    "max": lengths[-1],
                },
                "range_check": {
                    "below": below,
                    "within": within,
                    "above": above,
                    "below_fraction": below / sample_records,
                    "within_fraction": within / sample_records,
                    "above_fraction": above / sample_records,
                },
                "shortest_records": shortest,
                "longest_records": longest,
            }
        )

    report = {
        "method": {
            "description": (
                "Randomly select complete shard files from the precomputed 10% "
                "Bernoulli row sample, tokenize every record in those files, and "
                "scale sampled tokens by full_bytes/sample_bytes within each bucket."
            ),
            "seed": args.seed,
            "files_per_bucket": args.files_per_bucket,
            "bucket_unit_tokens": TOKENS_PER_BUCKET_UNIT,
        },
        "data_root": str(args.data_root.resolve()),
        "sample_root": str(args.sample_root.resolve()),
        "tokenizer": tokenizer_metadata,
        "elapsed_seconds": round(time.time() - started, 3),
        "totals": {
            "full_bytes": sum(item["full_bytes"] for item in bucket_reports),
            "sample_bytes": sum(item["sample_bytes"] for item in bucket_reports),
            "sample_storage_fraction": (
                sum(item["sample_bytes"] for item in bucket_reports)
                / sum(item["full_bytes"] for item in bucket_reports)
            ),
            "sample_records": sum(item["sample_records"] for item in bucket_reports),
            "sample_tokens": sum(item["sample_tokens"] for item in bucket_reports),
            "estimated_tokens": sum(item["estimated_tokens"] for item in bucket_reports),
        },
        "buckets": bucket_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
