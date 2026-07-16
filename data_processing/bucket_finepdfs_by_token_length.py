#!/usr/bin/env python3
"""Bucket every JSONL record by its full tokenizer length."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import orjson
from transformers import AutoTokenizer


KIB = 1024
MIB = 1024 * KIB
SCRIPT_VERSION = 1
BUCKETS = (
    {"label": "0-32K", "directory": "0-32K", "lower": 0, "upper": 32 * KIB},
    {"label": "32K-64K", "directory": "32K-64K", "lower": 32 * KIB, "upper": 64 * KIB},
    {"label": "64K-128K", "directory": "64K-128K", "lower": 64 * KIB, "upper": 128 * KIB},
    {"label": "128K-256K", "directory": "128K-256K", "lower": 128 * KIB, "upper": 256 * KIB},
    {"label": "256K-512K", "directory": "256K-512K", "lower": 256 * KIB, "upper": 512 * KIB},
    {"label": "512K-768K", "directory": "512K-768K", "lower": 512 * KIB, "upper": 768 * KIB},
    {"label": "768K-1M", "directory": "768K-1M", "lower": 768 * KIB, "upper": MIB},
    {"label": ">=1M", "directory": "1M-plus", "lower": MIB, "upper": None},
)

_TOKENIZER = None
_OUTPUT_ROOT = None
_STATE_ROOT = None
_CONFIG_DIGEST = None
_TEXT_ONLY_OUTPUT = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize all records and write length-bucketed JSONL shards."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument(
        "--input-suffix",
        default=".jsonl",
        help="Input shard suffix, for example .jsonl or .txt.",
    )
    parser.add_argument(
        "--text-only-output",
        action="store_true",
        help='Serialize each output record as only {"text": ...}.',
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        help="Per-shard resume state. Defaults beside --stats-output.",
    )
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument(
        "--max-inflight-gib",
        type=float,
        help=(
            "Limit the summed input sizes of running shards. A shard larger than "
            "the limit still runs alone."
        ),
    )
    parser.add_argument(
        "--recycle-workers",
        action="store_true",
        help="Recreate the process pool after each bounded batch to release tokenizer memory.",
    )
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def discover_inputs(input_root: Path, input_suffix: str = ".jsonl") -> list[Path]:
    if not input_suffix.startswith("."):
        input_suffix = f".{input_suffix}"
    files = sorted(input_root.glob(f"*{input_suffix}"))
    if not files:
        files = sorted(input_root.glob(f"*/*{input_suffix}"))
    if not files:
        raise RuntimeError(f"no {input_suffix} files found under {input_root}")
    relative_paths = [path.relative_to(input_root) for path in files]
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("duplicate input paths detected")
    return files


def output_name(relative_input: Path) -> str:
    parent_parts = relative_input.parts[:-1]
    filename = relative_input.with_suffix(".jsonl").name
    if not parent_parts:
        return filename
    return f"{'__'.join(parent_parts)}__{filename}"


def state_path(state_root: Path, relative_input: Path) -> Path:
    return state_root / relative_input.parent / f"{relative_input.name}.stats.json"


def bucket_index(token_count: int) -> int:
    for index, bucket in enumerate(BUCKETS[:-1]):
        if token_count < bucket["upper"]:
            return index
    return len(BUCKETS) - 1


def empty_bucket_stats() -> list[dict]:
    return [
        {
            "label": bucket["label"],
            "directory": bucket["directory"],
            "records": 0,
            "tokens": 0,
            "bytes": 0,
            "min_tokens": None,
            "max_tokens": None,
            "output_file": None,
        }
        for bucket in BUCKETS
    ]


def init_worker(
    tokenizer_path: str,
    output_root: str,
    state_root: str,
    config_digest: str,
    text_only_output: bool,
) -> None:
    global _TOKENIZER, _OUTPUT_ROOT, _STATE_ROOT, _CONFIG_DIGEST, _TEXT_ONLY_OUTPUT
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    _TOKENIZER.model_max_length = sys.maxsize
    _OUTPUT_ROOT = Path(output_root)
    _STATE_ROOT = Path(state_root)
    _CONFIG_DIGEST = config_digest
    _TEXT_ONLY_OUTPUT = text_only_output


def process_file(input_path_s: str, input_root_s: str) -> dict:
    if _TOKENIZER is None or _OUTPUT_ROOT is None or _STATE_ROOT is None:
        raise RuntimeError("worker was not initialized")

    input_path = Path(input_path_s)
    input_root = Path(input_root_s)
    relative_input = input_path.relative_to(input_root)
    shard_output_name = output_name(relative_input)
    per_bucket = empty_bucket_stats()
    writers: dict[int, object] = {}
    tmp_paths: dict[int, Path] = {}
    final_paths: dict[int, Path] = {}
    started = time.time()
    input_bytes = input_path.stat().st_size
    records = 0
    tokens = 0
    bytes_read = 0

    try:
        with input_path.open("rb", buffering=4 * MIB) as source:
            for line_no, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(f"{input_path}:{line_no}: blank JSONL line")
                item = orjson.loads(line)
                text = item.get("text")
                if not isinstance(text, str):
                    raise TypeError(f"{input_path}:{line_no}: text is not a string")

                token_count = len(
                    _TOKENIZER.encode(
                        text,
                        add_special_tokens=False,
                        truncation=False,
                        verbose=False,
                    )
                )
                index = bucket_index(token_count)
                bucket = BUCKETS[index]
                stats = per_bucket[index]
                output_line = (
                    orjson.dumps({"text": text}, option=orjson.OPT_APPEND_NEWLINE)
                    if _TEXT_ONLY_OUTPUT
                    else line
                )

                if index not in writers:
                    output_dir = _OUTPUT_ROOT / bucket["directory"]
                    output_dir.mkdir(parents=True, exist_ok=True)
                    final_path = output_dir / shard_output_name
                    tmp_path = output_dir / f".{shard_output_name}.tmp.{os.getpid()}"
                    writers[index] = tmp_path.open("wb", buffering=4 * MIB)
                    tmp_paths[index] = tmp_path
                    final_paths[index] = final_path

                writers[index].write(output_line)
                records += 1
                tokens += token_count
                bytes_read += len(line)
                stats["records"] += 1
                stats["tokens"] += token_count
                stats["bytes"] += len(output_line)
                stats["min_tokens"] = (
                    token_count
                    if stats["min_tokens"] is None
                    else min(stats["min_tokens"], token_count)
                )
                stats["max_tokens"] = (
                    token_count
                    if stats["max_tokens"] is None
                    else max(stats["max_tokens"], token_count)
                )

        if bytes_read != input_bytes:
            raise RuntimeError(
                f"{input_path}: read {bytes_read} bytes but stat reports {input_bytes}"
            )

        for writer in writers.values():
            writer.close()
        writers.clear()

        for index, tmp_path in tmp_paths.items():
            final_path = final_paths[index]
            os.replace(tmp_path, final_path)
            per_bucket[index]["output_file"] = str(final_path.relative_to(_OUTPUT_ROOT))

        result = {
            "config_digest": _CONFIG_DIGEST,
            "input_file": str(relative_input),
            "input_size": input_bytes,
            "input_mtime_ns": input_path.stat().st_mtime_ns,
            "records": records,
            "tokens": tokens,
            "output_bytes": sum(item["bytes"] for item in per_bucket),
            "elapsed_seconds": round(time.time() - started, 3),
            "buckets": per_bucket,
        }
        atomic_write_json(state_path(_STATE_ROOT, relative_input), result)
        return result
    except BaseException:
        for writer in writers.values():
            writer.close()
        for tmp_path in tmp_paths.values():
            tmp_path.unlink(missing_ok=True)
        raise


def valid_completed_state(
    path: Path,
    input_path: Path,
    output_root: Path,
    config_digest: str,
    require_equal_bytes: bool,
) -> dict | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        stat = input_path.stat()
        if result["config_digest"] != config_digest:
            return None
        if result["input_size"] != stat.st_size or result["input_mtime_ns"] != stat.st_mtime_ns:
            return None
        if require_equal_bytes and result["output_bytes"] != result["input_size"]:
            return None
        if result["output_bytes"] != sum(bucket["bytes"] for bucket in result["buckets"]):
            return None
        for bucket in result["buckets"]:
            relative_output = bucket["output_file"]
            if bucket["records"] == 0:
                if relative_output is not None:
                    return None
                continue
            if relative_output is None:
                return None
            output_path = output_root / relative_output
            if not output_path.is_file() or output_path.stat().st_size != bucket["bytes"]:
                return None
        return result
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def aggregate_results(results: list[dict]) -> list[dict]:
    aggregate = empty_bucket_stats()
    for item in aggregate:
        item.pop("output_file")
        item["files"] = 0
    for result in results:
        for index, bucket in enumerate(result["buckets"]):
            target = aggregate[index]
            target["records"] += bucket["records"]
            target["tokens"] += bucket["tokens"]
            target["bytes"] += bucket["bytes"]
            if bucket["records"]:
                target["files"] += 1
                target["min_tokens"] = (
                    bucket["min_tokens"]
                    if target["min_tokens"] is None
                    else min(target["min_tokens"], bucket["min_tokens"])
                )
                target["max_tokens"] = (
                    bucket["max_tokens"]
                    if target["max_tokens"] is None
                    else max(target["max_tokens"], bucket["max_tokens"])
                )
    return aggregate


def add_fractions(buckets: list[dict], totals: dict) -> None:
    for bucket in buckets:
        bucket["record_fraction"] = (
            bucket["records"] / totals["records"] if totals["records"] else 0
        )
        bucket["token_fraction"] = (
            bucket["tokens"] / totals["tokens"] if totals["tokens"] else 0
        )
        bucket["byte_fraction"] = bucket["bytes"] / totals["bytes"] if totals["bytes"] else 0
        bucket["mean_tokens"] = (
            bucket["tokens"] / bucket["records"] if bucket["records"] else None
        )


def write_csv(path: Path, buckets: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "label",
        "directory",
        "lower",
        "upper",
        "records",
        "record_fraction",
        "tokens",
        "token_fraction",
        "bytes",
        "byte_fraction",
        "files",
        "min_tokens",
        "mean_tokens",
        "max_tokens",
    )
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(buckets)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        f"# Token-length buckets: {Path(report['input_root']).name}",
        "",
        f"- Input files: {report['totals']['files']:,}",
        f"- Records: {report['totals']['records']:,}",
        f"- Tokens: {report['totals']['tokens']:,}",
    ]
    if report["totals"]["input_bytes"] == report["totals"]["output_bytes"]:
        lines.append(f"- JSONL bytes: {report['totals']['output_bytes']:,}")
    else:
        lines.extend(
            [
                f"- Input bytes: {report['totals']['input_bytes']:,}",
                f"- Output JSONL bytes: {report['totals']['output_bytes']:,}",
            ]
        )
    lines.extend(
        [
            f"- K/M units: binary (K={KIB}, M={MIB})",
            "- Intervals: lower-inclusive, upper-exclusive",
            "",
            "| Bucket | Records | Records % | Tokens | Tokens % | GiB | Min | Mean | Max | Files |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket in report["buckets"]:
        minimum = f"{bucket['min_tokens']:,}" if bucket["min_tokens"] is not None else "-"
        mean = f"{bucket['mean_tokens']:,.2f}" if bucket["mean_tokens"] is not None else "-"
        maximum = f"{bucket['max_tokens']:,}" if bucket["max_tokens"] is not None else "-"
        lines.append(
            f"| {bucket['label']} | {bucket['records']:,} | "
            f"{bucket['record_fraction']:.4%} | {bucket['tokens']:,} | "
            f"{bucket['token_fraction']:.4%} | {bucket['bytes'] / (1024 ** 3):.3f} | "
            f"{minimum} | {mean} | {maximum} | {bucket['files']:,} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


def make_work_batches(
    items: list[tuple[Path, int]],
    max_files: int,
    max_bytes: int | None,
) -> list[list[tuple[Path, int]]]:
    batches = []
    current = []
    current_bytes = 0
    for item in items:
        _, size = item
        if current and (
            len(current) >= max_files
            or (max_bytes is not None and current_bytes + size > max_bytes)
        ):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.max_inflight_gib is not None and args.max_inflight_gib <= 0:
        raise ValueError("--max-inflight-gib must be positive")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    tokenizer_path = args.tokenizer.resolve()
    stats_output = args.stats_output.resolve()
    state_root = (
        args.state_root.resolve()
        if args.state_root
        else stats_output.parent / f"{stats_output.stem}_parts"
    )
    input_suffix = args.input_suffix
    if not input_suffix.startswith("."):
        input_suffix = f".{input_suffix}"
    if input_root == output_root or output_root.is_relative_to(input_root):
        raise ValueError("--output-root must not be the input root or a child of it")

    input_files = discover_inputs(input_root, input_suffix)
    tokenizer_json = tokenizer_path / "tokenizer.json"
    config = {
        "script_version": SCRIPT_VERSION,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "add_special_tokens": False,
        "truncation": False,
        "bucket_unit": {"K": KIB, "M": MIB},
        "buckets": list(BUCKETS),
    }
    # Keep the original JSONL/pass-through digest stable for existing resume states.
    if input_suffix != ".jsonl":
        config["input_suffix"] = input_suffix
    if args.text_only_output:
        config["text_only_output"] = True
    config_digest = canonical_digest(config)
    manifest_path = state_root / "manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("config_digest") != config_digest:
            raise RuntimeError(f"existing state manifest does not match this run: {manifest_path}")
    else:
        atomic_write_json(
            manifest_path,
            {"config_digest": config_digest, "config": config, "created_at": time.time()},
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for bucket in BUCKETS:
        (output_root / bucket["directory"]).mkdir(parents=True, exist_ok=True)

    completed_results = []
    pending_files = []
    for input_path in input_files:
        relative_input = input_path.relative_to(input_root)
        completed = valid_completed_state(
            state_path(state_root, relative_input),
            input_path,
            output_root,
            config_digest,
            require_equal_bytes=not args.text_only_output,
        )
        if completed is None:
            pending_files.append(input_path)
        else:
            completed_results.append(completed)

    print(
        f"Starting full bucketing: files={len(input_files):,}, "
        f"already_complete={len(completed_results):,}, pending={len(pending_files):,}, "
        f"workers={min(args.workers, max(1, len(pending_files))):,}",
        flush=True,
    )
    started = time.time()
    newly_completed = []
    if pending_files:
        workers = min(args.workers, len(pending_files))
        pending_with_sizes = sorted(
            ((path, path.stat().st_size) for path in pending_files),
            key=lambda item: item[1],
            reverse=True,
        )
        total_pending_bytes = sum(size for _, size in pending_with_sizes)
        max_inflight_bytes = (
            round(args.max_inflight_gib * (1024**3))
            if args.max_inflight_gib is not None
            else None
        )
        worker_initargs = (
            str(tokenizer_path),
            str(output_root),
            str(state_root),
            config_digest,
            args.text_only_output,
        )
        run_bytes = 0
        run_records = 0
        run_tokens = 0
        done_count = 0

        def record_completion(result: dict, inflight_bytes: int) -> None:
            nonlocal done_count, run_bytes, run_records, run_tokens
            newly_completed.append(result)
            done_count += 1
            run_bytes += result["input_size"]
            run_records += result["records"]
            run_tokens += result["tokens"]
            if (
                done_count % args.progress_every == 0
                or done_count == len(pending_with_sizes)
            ):
                elapsed = time.time() - started
                rate = run_bytes / max(elapsed, 1e-9)
                remaining_bytes = total_pending_bytes - run_bytes
                eta = remaining_bytes / rate if rate else 0
                print(
                    f"Progress {done_count:,}/{len(pending_with_sizes):,}: "
                    f"records={run_records:,}, tokens={run_tokens:,}, "
                    f"read={run_bytes / (1024 ** 3):.2f}GiB, "
                    f"inflight={inflight_bytes / (1024 ** 3):.2f}GiB, "
                    f"rate={rate / MIB:.1f}MiB/s, eta={format_duration(eta)}",
                    flush=True,
                )

        if args.recycle_workers:
            batches = make_work_batches(
                pending_with_sizes,
                max_files=workers,
                max_bytes=max_inflight_bytes,
            )
            for batch_index, batch in enumerate(batches, start=1):
                inflight_bytes = sum(size for _, size in batch)
                print(
                    f"Starting batch {batch_index:,}/{len(batches):,}: "
                    f"files={len(batch):,}, input={inflight_bytes / (1024 ** 3):.2f}GiB",
                    flush=True,
                )
                with ProcessPoolExecutor(
                    max_workers=len(batch),
                    initializer=init_worker,
                    initargs=worker_initargs,
                ) as executor:
                    futures = {
                        executor.submit(process_file, str(path), str(input_root)): size
                        for path, size in batch
                    }
                    while futures:
                        completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in completed:
                            size = futures.pop(future)
                            inflight_bytes -= size
                            record_completion(future.result(), inflight_bytes)
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_worker,
                initargs=worker_initargs,
            ) as executor:
                futures = {}
                next_index = 0
                inflight_bytes = 0
                while next_index < len(pending_with_sizes) or futures:
                    while next_index < len(pending_with_sizes) and len(futures) < workers:
                        path, size = pending_with_sizes[next_index]
                        if (
                            futures
                            and max_inflight_bytes is not None
                            and inflight_bytes + size > max_inflight_bytes
                        ):
                            break
                        future = executor.submit(process_file, str(path), str(input_root))
                        futures[future] = size
                        inflight_bytes += size
                        next_index += 1

                    completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in completed:
                        size = futures.pop(future)
                        inflight_bytes -= size
                        record_completion(future.result(), inflight_bytes)

    all_results = completed_results + newly_completed
    if len(all_results) != len(input_files):
        raise RuntimeError(
            f"completed result count {len(all_results)} does not match input count {len(input_files)}"
        )
    aggregate = aggregate_results(all_results)
    total_input_bytes = sum(path.stat().st_size for path in input_files)
    totals = {
        "files": len(input_files),
        "records": sum(result["records"] for result in all_results),
        "tokens": sum(result["tokens"] for result in all_results),
        "bytes": sum(result["output_bytes"] for result in all_results),
        "input_bytes": total_input_bytes,
        "output_bytes": sum(result["output_bytes"] for result in all_results),
    }
    if not args.text_only_output and totals["output_bytes"] != total_input_bytes:
        raise RuntimeError(
            f"output byte total {totals['output_bytes']} does not match input "
            f"{total_input_bytes}"
        )
    if totals["records"] != sum(bucket["records"] for bucket in aggregate):
        raise RuntimeError("bucket record totals do not match")
    if totals["tokens"] != sum(bucket["tokens"] for bucket in aggregate):
        raise RuntimeError("bucket token totals do not match")

    for index, bucket in enumerate(aggregate):
        bucket["lower"] = BUCKETS[index]["lower"]
        bucket["upper"] = BUCKETS[index]["upper"]
    add_fractions(aggregate, totals)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    report = {
        "config_digest": config_digest,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "state_root": str(state_root),
        "input_suffix": input_suffix,
        "text_only_output": args.text_only_output,
        "output_schema": {"text": "string"} if args.text_only_output else "unchanged",
        "tokenizer": {
            "path": str(tokenizer_path),
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "model_max_length": tokenizer.model_max_length,
            "tokenizer_json_sha256": config["tokenizer_json_sha256"],
            "add_special_tokens": False,
            "truncation": False,
        },
        "bucket_unit": config["bucket_unit"],
        "interval_semantics": "lower-inclusive, upper-exclusive; final bucket has no upper bound",
        "elapsed_seconds_this_run": round(time.time() - started, 3),
        "resumed_files": len(completed_results),
        "processed_files_this_run": len(newly_completed),
        "totals": totals,
        "buckets": aggregate,
    }
    atomic_write_json(stats_output, report)
    write_csv(stats_output.with_suffix(".csv"), aggregate)
    write_markdown(stats_output.with_suffix(".md"), report)
    (output_root / "_SUCCESS").touch()
    print(
        f"Done: records={totals['records']:,}, tokens={totals['tokens']:,}, "
        f"bytes={totals['bytes']:,}. Stats: {stats_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
