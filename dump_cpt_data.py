#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import orjson


ROOT = Path(__file__).resolve().parent

TEMP_FRAGMENT_RE = re.compile(r"\.(?:json|txt)_[0-9a-f]{32}$")

DATASETS = (
    {
        "name": "finepdfs_20260703",
        "kind": "finepdfs",
        "src": ROOT / "cpt-data/finepdfs_20260703",
        "dst": ROOT / "cpt-data-dumped/finepdfs_20260703",
        "sample_dst": ROOT / "cpt-data-dumped/finepdfs_20260703_downsampled_10_percent",
    },
    {
        "name": "long_code_v01",
        "kind": "text",
        "src": ROOT / "cpt-data/github-data-bj/long_code_v01",
        "dst": ROOT / "cpt-data-dumped/github-data-bj/long_code_v01",
        "sample_dst": None,
    },
    {
        "name": "long_code_v02",
        "kind": "text",
        "src": ROOT / "cpt-data/github-data-bj/long_code_v02",
        "dst": ROOT / "cpt-data-dumped/github-data-bj/long_code_v02",
        "sample_dst": None,
    },
)


def is_temp_fragment(path: Path) -> bool:
    if not TEMP_FRAGMENT_RE.search(path.name):
        return False
    normal_path = Path(str(path).rsplit("_", 1)[0])
    return normal_path.exists()


def output_path(src: Path, src_root: Path, dst_root: Path) -> Path:
    rel = src.relative_to(src_root)
    return (dst_root / rel).with_suffix(".jsonl")


def iter_source_files(src_root: Path):
    for path in sorted(src_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "_SUCCESS":
            continue
        if is_temp_fragment(path):
            continue
        yield path


def extract_text(item, kind: str, src: str, line_no: int) -> str:
    if kind == "text":
        text = item["text"]
    elif kind == "finepdfs":
        messages = item["messages"]
        if len(messages) != 1:
            raise AssertionError(f"{src}:{line_no}: expected len(messages) == 1, got {len(messages)}")
        text = messages[0]["content"]
    else:
        raise ValueError(f"unknown kind: {kind}")

    if not isinstance(text, str):
        raise TypeError(f"{src}:{line_no}: text is {type(text).__name__}, expected str")
    return text


def atomic_replace(tmp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path, final_path)


def process_file(args):
    src_s, dst_s, kind, sample_dst_s, sample_ratio, seed = args
    src = Path(src_s)
    dst = Path(dst_s)
    tmp = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
    sample_dst = Path(sample_dst_s) if sample_dst_s else None
    sample_tmp = sample_dst.with_name(f"{sample_dst.name}.tmp.{os.getpid()}") if sample_dst else None

    dst.parent.mkdir(parents=True, exist_ok=True)
    if sample_dst:
        sample_dst.parent.mkdir(parents=True, exist_ok=True)

    line_count = 0
    sample_count = 0
    rng = random.Random(f"{seed}\0{src_s}") if sample_dst else None

    try:
        with open(src, "rb", buffering=1024 * 1024) as fin:
            with open(tmp, "wb", buffering=1024 * 1024) as fout:
                sample_out = open(sample_tmp, "wb", buffering=1024 * 1024) if sample_tmp else None
                try:
                    for line_count, line in enumerate(fin, start=1):
                        if not line.strip():
                            continue
                        try:
                            item = orjson.loads(line)
                            text = extract_text(item, kind, src_s, line_count)
                        except Exception as exc:
                            raise RuntimeError(f"failed to parse {src_s}:{line_count}: {exc}") from exc

                        out_line = orjson.dumps({"text": text}, option=orjson.OPT_APPEND_NEWLINE)
                        fout.write(out_line)
                        if sample_out is not None and rng.random() < sample_ratio:
                            sample_out.write(out_line)
                            sample_count += 1
                finally:
                    if sample_out is not None:
                        sample_out.close()

        atomic_replace(tmp, dst)
        if sample_dst:
            atomic_replace(sample_tmp, sample_dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        if sample_tmp:
            sample_tmp.unlink(missing_ok=True)
        raise

    return {
        "src": src_s,
        "dst": dst_s,
        "lines": line_count,
        "sample_lines": sample_count,
        "bytes_in": src.stat().st_size,
        "bytes_out": dst.stat().st_size,
        "sample_bytes_out": sample_dst.stat().st_size if sample_dst else 0,
    }


def build_jobs(selected_names: set[str], sample_ratio: float, seed: int):
    jobs = []
    seen_outputs = set()
    skipped_temp = 0
    skipped_success = 0

    for dataset in DATASETS:
        if selected_names and dataset["name"] not in selected_names:
            continue
        src_root = dataset["src"]
        if not src_root.exists():
            raise FileNotFoundError(src_root)

        for path in sorted(src_root.rglob("*")):
            if not path.is_file():
                continue
            if path.name == "_SUCCESS":
                skipped_success += 1
                continue
            if is_temp_fragment(path):
                skipped_temp += 1
                continue

            dst = output_path(path, src_root, dataset["dst"])
            sample_dst = (
                output_path(path, src_root, dataset["sample_dst"])
                if dataset["sample_dst"] is not None
                else None
            )
            out_key = str(dst)
            if out_key in seen_outputs:
                raise RuntimeError(f"duplicate output path detected: {dst}")
            seen_outputs.add(out_key)
            jobs.append((str(path), str(dst), dataset["kind"], str(sample_dst) if sample_dst else None, sample_ratio, seed))

    return jobs, skipped_success, skipped_temp


def parse_args():
    parser = argparse.ArgumentParser(description="Dump CPT JSONL data to {\"text\": ...} JSONL files.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=[d["name"] for d in DATASETS],
        help="Dataset to export. Can be passed multiple times. Defaults to all required datasets.",
    )
    parser.add_argument("--jobs", type=int, default=8, help="Number of worker processes.")
    parser.add_argument("--sample-ratio", type=float, default=0.10, help="Sampling ratio for finepdfs.")
    parser.add_argument("--seed", type=int, default=20260703, help="Deterministic downsample seed.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N completed files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.sample_ratio <= 1.0:
        raise ValueError("--sample-ratio must be between 0 and 1")

    jobs, skipped_success, skipped_temp = build_jobs(set(args.dataset or []), args.sample_ratio, args.seed)
    if not jobs:
        print("No input files found.", flush=True)
        return 0

    print(
        f"Starting export: files={len(jobs)}, jobs={args.jobs}, "
        f"skipped_SUCCESS={skipped_success}, skipped_temp_fragments={skipped_temp}",
        flush=True,
    )

    started = time.time()
    total_lines = 0
    total_sample_lines = 0
    total_bytes_in = 0
    total_bytes_out = 0
    total_sample_bytes_out = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(process_file, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            total_lines += result["lines"]
            total_sample_lines += result["sample_lines"]
            total_bytes_in += result["bytes_in"]
            total_bytes_out += result["bytes_out"]
            total_sample_bytes_out += result["sample_bytes_out"]

            if done % args.progress_every == 0 or done == len(jobs):
                elapsed = max(time.time() - started, 1e-9)
                mbps = total_bytes_in / elapsed / (1024 * 1024)
                print(
                    f"Progress {done}/{len(jobs)} files, "
                    f"lines={total_lines}, sample_lines={total_sample_lines}, "
                    f"read={total_bytes_in / (1024 ** 3):.2f}GiB, "
                    f"wrote={total_bytes_out / (1024 ** 3):.2f}GiB, "
                    f"sample_wrote={total_sample_bytes_out / (1024 ** 3):.2f}GiB, "
                    f"throughput={mbps:.2f}MiB/s",
                    flush=True,
                )

    elapsed = time.time() - started
    print(
        "Done: "
        f"files={len(jobs)}, lines={total_lines}, sample_lines={total_sample_lines}, "
        f"read={total_bytes_in / (1024 ** 3):.2f}GiB, "
        f"wrote={total_bytes_out / (1024 ** 3):.2f}GiB, "
        f"sample_wrote={total_sample_bytes_out / (1024 ** 3):.2f}GiB, "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
