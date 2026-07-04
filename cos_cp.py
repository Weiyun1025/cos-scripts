#!/usr/bin/env python3
"""Copy files/directories between local paths and Tencent COS paths.

COS paths must include the bucket name:

    cos://wangweiyun-1306757789/backup/20260702/file.bin
"""

from __future__ import annotations

import argparse
import os
import posixpath
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator
from urllib.parse import urlsplit


DEFAULT_REGION = "ap-beijing"
DOTENV_KEYS = {
    "COS_SECRET_ID",
    "COS_SECRET_KEY",
    "COS_REGION",
    "COS_ENDPOINT",
}


@dataclass(frozen=True)
class LocalPath:
    raw: str
    path: Path


@dataclass(frozen=True)
class CosPath:
    raw: str
    bucket: str
    key: str


@dataclass(frozen=True)
class LocalFile:
    path: Path


@dataclass(frozen=True)
class LocalDir:
    path: Path


@dataclass(frozen=True)
class CosObject:
    bucket: str
    key: str
    size: int | None = None


@dataclass(frozen=True)
class CosPrefix:
    bucket: str
    prefix: str


class Counters:
    def __init__(self) -> None:
        self.files = 0
        self.bytes = 0
        self.skipped_files = 0
        self.skipped_bytes = 0

    def add(self, size: int | None = None) -> None:
        self.files += 1
        if size is not None:
            self.bytes += size

    def skip(self, size: int | None = None) -> None:
        self.skipped_files += 1
        if size is not None:
            self.skipped_bytes += size


def is_cos_uri(value: str) -> bool:
    return value.startswith("cos://")


def normalize_endpoint(value: str | None) -> str | None:
    if not value:
        return None
    split = urlsplit(value if "://" in value else f"//{value}")
    host = split.netloc or split.path
    return host.strip("/")


def unquote_dotenv_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
        if value:
            value = bytes(value, "utf-8").decode("unicode_escape")
    return value


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_no}: missing '='")

        key, value = line.split("=", 1)
        key = key.strip()
        if key not in DOTENV_KEYS:
            continue
        os.environ.setdefault(key, unquote_dotenv_value(value))


def parse_path(value: str) -> LocalPath | CosPath:
    if not is_cos_uri(value):
        return LocalPath(raw=value, path=Path(value).expanduser())

    split = urlsplit(value)
    if split.query or split.fragment:
        raise ValueError(f"COS path must not include query or fragment: {value!r}")
    if not split.netloc:
        raise ValueError(f"COS path must be formatted as cos://<bucket>/<key>: {value!r}")

    bucket = split.netloc
    key = split.path.lstrip("/")
    return CosPath(raw=value, bucket=bucket, key=key)


def join_cos(prefix: str, name: str) -> str:
    if not prefix:
        return name
    return posixpath.join(prefix.rstrip("/"), name)


def ensure_cos_prefix(key: str) -> str:
    if not key:
        return ""
    return key if key.endswith("/") else f"{key}/"


def destination_looks_like_dir(raw: str) -> bool:
    return raw.endswith("/") or raw.endswith(os.sep)


def cos_basename(key: str) -> str:
    name = PurePosixPath(key.rstrip("/")).name
    if not name:
        raise ValueError(f"Cannot derive a file name from COS key {key!r}")
    return name


def safe_local_relative_path(rel: str) -> Path:
    rel_path = PurePosixPath(rel)
    parts = rel_path.parts
    if rel_path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Refusing unsafe COS object key segment: {rel!r}")
    return Path(*parts)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def local_file_size(path: Path) -> int | None:
    try:
        if path.is_file():
            return path.stat().st_size
    except OSError:
        return None
    return None


def should_skip_existing_download(target: Path, expected_size: int | None, force: bool) -> bool:
    if force or expected_size is None:
        return False
    return local_file_size(target) == expected_size


def download_with_retries(
    cos: "CosStore",
    bucket: str,
    key: str,
    target: Path,
    attempts: int,
    delay_seconds: float,
) -> None:
    attempts = max(1, attempts)
    for attempt in range(1, attempts + 1):
        try:
            cos.download_file(bucket, key, target)
            return
        except Exception as exc:
            if attempt >= attempts:
                raise
            print(
                f"retry download {attempt}/{attempts - 1} after error: "
                f"cos://{bucket}/{key} -> {target}: {exc}",
                file=sys.stderr,
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)


def iter_local_files(root: Path) -> Iterator[tuple[Path, str, int]]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        yield path, rel, path.stat().st_size


def require_cos_sdk():
    try:
        from qcloud_cos import CosConfig, CosS3Client
        from qcloud_cos.cos_exception import CosServiceError
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: qcloud_cos. Install it with:\n"
            "  pip install cos-python-sdk-v5"
        ) from exc
    return CosConfig, CosS3Client, CosServiceError


class CosStore:
    def __init__(
        self,
        secret_id: str,
        secret_key: str,
        region: str,
        endpoint: str | None,
        part_size_mb: int,
        threads: int,
    ) -> None:
        CosConfig, CosS3Client, CosServiceError = require_cos_sdk()
        self.CosServiceError = CosServiceError
        cfg_kwargs = {
            "Region": region,
            "SecretId": secret_id,
            "SecretKey": secret_key,
            "Scheme": "https",
        }
        if endpoint:
            cfg_kwargs["Endpoint"] = endpoint
        self.client = CosS3Client(CosConfig(**cfg_kwargs))
        self.region = region
        self.part_size_mb = part_size_mb
        self.threads = threads

    def head_object(self, bucket: str, key: str) -> int | None:
        if not key:
            return None
        try:
            resp = self.client.head_object(Bucket=bucket, Key=key)
        except self.CosServiceError as exc:
            if exc.get_status_code() == 404:
                return None
            raise
        return int(resp.get("Content-Length", 0))

    def head_object_for_source_detection(self, bucket: str, key: str) -> int | None:
        try:
            return self.head_object(bucket, key)
        except self.CosServiceError as exc:
            if exc.get_status_code() == 403:
                return None
            raise

    def object_exists(self, bucket: str, key: str) -> bool:
        return self.head_object(bucket, key) is not None

    def iter_objects(self, bucket: str, prefix: str) -> Iterator[tuple[str, int]]:
        marker = ""
        while True:
            resp = self.client.list_objects(
                Bucket=bucket,
                Prefix=prefix,
                Marker=marker,
                MaxKeys=1000,
            )
            for item in resp.get("Contents") or []:
                key = item["Key"]
                size = int(item.get("Size", 0))
                yield key, size

            truncated = str(resp.get("IsTruncated", "")).lower() == "true"
            if not truncated:
                break
            marker = resp.get("NextMarker") or ""
            if not marker:
                contents = resp.get("Contents") or []
                if not contents:
                    break
                marker = contents[-1]["Key"]

    def prefix_has_objects(self, bucket: str, prefix: str) -> bool:
        return next(self.iter_objects(bucket, prefix), None) is not None

    def upload_file(self, bucket: str, key: str, local_file: Path) -> None:
        self.client.upload_file(
            Bucket=bucket,
            Key=key,
            LocalFilePath=str(local_file),
            PartSize=self.part_size_mb,
            MAXThread=self.threads,
            EnableCRC=True,
        )

    def download_file(self, bucket: str, key: str, local_file: Path) -> None:
        local_file.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(
            Bucket=bucket,
            Key=key,
            DestFilePath=str(local_file),
            PartSize=self.part_size_mb,
            MAXThread=self.threads,
            EnableCRC=True,
        )

    def copy_object(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> None:
        copy_source = {
            "Bucket": src_bucket,
            "Region": self.region,
            "Key": src_key,
        }
        self.client.copy(
            Bucket=dst_bucket,
            Key=dst_key,
            CopySource=copy_source,
            PartSize=self.part_size_mb,
            MAXThread=self.threads,
        )


def resolve_source(src: LocalPath | CosPath, cos: CosStore | None) -> LocalFile | LocalDir | CosObject | CosPrefix:
    if isinstance(src, LocalPath):
        if src.path.is_file():
            return LocalFile(src.path)
        if src.path.is_dir():
            return LocalDir(src.path)
        raise FileNotFoundError(f"Local source does not exist: {src.path}")

    if cos is None:
        raise RuntimeError("COS client is not initialized")

    size = cos.head_object_for_source_detection(src.bucket, src.key)
    if size is not None:
        return CosObject(src.bucket, src.key, size)

    prefix = src.key if src.key.endswith("/") or src.key == "" else f"{src.key}/"
    if cos.prefix_has_objects(src.bucket, prefix):
        return CosPrefix(src.bucket, prefix)

    raise FileNotFoundError(
        f"COS source not found as object or prefix: cos://{src.bucket}/{src.key}"
    )


def local_file_destination(src_path: Path, dst: LocalPath) -> Path:
    if dst.path.exists() and dst.path.is_dir():
        return dst.path / src_path.name
    if destination_looks_like_dir(dst.raw):
        return dst.path / src_path.name
    return dst.path


def cos_object_destination(src_key: str, dst: CosPath) -> str:
    if dst.key == "" or destination_looks_like_dir(dst.raw):
        return join_cos(dst.key, cos_basename(src_key))
    return dst.key


def copy_local_file_to_local(src: LocalFile, dst: LocalPath, dry_run: bool, counters: Counters) -> None:
    target = local_file_destination(src.path, dst)
    print(f"copy {src.path} -> {target}")
    counters.add(src.path.stat().st_size)
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src.path, target)


def copy_local_dir_to_local(src: LocalDir, dst: LocalPath, dry_run: bool, counters: Counters) -> None:
    if dst.path.exists() and not dst.path.is_dir():
        raise NotADirectoryError(f"Destination must be a directory for directory copy: {dst.path}")
    src_root = src.path.resolve()
    dst_root = dst.path.resolve(strict=False)
    if is_relative_to(dst_root, src_root):
        raise ValueError(f"Destination must not be inside source directory: {dst.path}")
    for path, rel, size in iter_local_files(src.path):
        target = dst.path / rel
        print(f"copy {path} -> {target}")
        counters.add(size)
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def copy_local_file_to_cos(
    src: LocalFile,
    dst: CosPath,
    cos: CosStore | None,
    dry_run: bool,
    counters: Counters,
) -> None:
    key = cos_object_destination(src.path.name, dst)
    print(f"upload {src.path} -> cos://{dst.bucket}/{key}")
    counters.add(src.path.stat().st_size)
    if not dry_run:
        if cos is None:
            raise RuntimeError("COS client is not initialized")
        cos.upload_file(dst.bucket, key, src.path)


def copy_local_dir_to_cos(
    src: LocalDir,
    dst: CosPath,
    cos: CosStore | None,
    dry_run: bool,
    counters: Counters,
) -> None:
    prefix = ensure_cos_prefix(dst.key)
    for path, rel, size in iter_local_files(src.path):
        key = join_cos(prefix, rel)
        print(f"upload {path} -> cos://{dst.bucket}/{key}")
        counters.add(size)
        if not dry_run:
            if cos is None:
                raise RuntimeError("COS client is not initialized")
            cos.upload_file(dst.bucket, key, path)


def copy_cos_object_to_local(
    src: CosObject,
    dst: LocalPath,
    cos: CosStore,
    dry_run: bool,
    force: bool,
    download_retries: int,
    download_retry_delay: float,
    counters: Counters,
) -> None:
    if dst.path.exists() and dst.path.is_dir():
        target = dst.path / cos_basename(src.key)
    elif destination_looks_like_dir(dst.raw):
        target = dst.path / cos_basename(src.key)
    else:
        target = dst.path

    source = f"cos://{src.bucket}/{src.key}"
    if should_skip_existing_download(target, src.size, force):
        print(f"skip {source} -> {target} (exists, size matches)")
        counters.skip(src.size)
        return

    print(f"download {source} -> {target}")
    counters.add(src.size)
    if not dry_run:
        download_with_retries(cos, src.bucket, src.key, target, download_retries, download_retry_delay)


def copy_cos_prefix_to_local(
    src: CosPrefix,
    dst: LocalPath,
    cos: CosStore,
    dry_run: bool,
    force: bool,
    download_retries: int,
    download_retry_delay: float,
    counters: Counters,
) -> None:
    if dst.path.exists() and not dst.path.is_dir():
        raise NotADirectoryError(f"Destination must be a directory for prefix copy: {dst.path}")

    for key, size in cos.iter_objects(src.bucket, src.prefix):
        rel = key[len(src.prefix) :]
        if not rel or key.endswith("/"):
            continue
        target = dst.path / safe_local_relative_path(rel)
        source = f"cos://{src.bucket}/{key}"
        if should_skip_existing_download(target, size, force):
            print(f"skip {source} -> {target} (exists, size matches)")
            counters.skip(size)
            continue

        print(f"download {source} -> {target}")
        counters.add(size)
        if not dry_run:
            download_with_retries(cos, src.bucket, key, target, download_retries, download_retry_delay)


def copy_cos_object_to_cos(
    src: CosObject,
    dst: CosPath,
    cos: CosStore,
    dry_run: bool,
    counters: Counters,
) -> None:
    dst_key = cos_object_destination(src.key, dst)
    print(f"copy cos://{src.bucket}/{src.key} -> cos://{dst.bucket}/{dst_key}")
    counters.add(src.size)
    if not dry_run:
        cos.copy_object(src.bucket, src.key, dst.bucket, dst_key)


def copy_cos_prefix_to_cos(
    src: CosPrefix,
    dst: CosPath,
    cos: CosStore,
    dry_run: bool,
    counters: Counters,
) -> None:
    dst_prefix = ensure_cos_prefix(dst.key)
    if src.bucket == dst.bucket and dst_prefix.startswith(src.prefix) and dst_prefix != src.prefix:
        raise ValueError(
            "Destination COS prefix is inside the source prefix; refusing to avoid "
            f"recursive copy: cos://{src.bucket}/{src.prefix} -> cos://{dst.bucket}/{dst_prefix}"
        )
    for key, size in cos.iter_objects(src.bucket, src.prefix):
        rel = key[len(src.prefix) :]
        if not rel or key.endswith("/"):
            continue
        dst_key = join_cos(dst_prefix, rel)
        print(f"copy cos://{src.bucket}/{key} -> cos://{dst.bucket}/{dst_key}")
        counters.add(size)
        if not dry_run:
            cos.copy_object(src.bucket, key, dst.bucket, dst_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy files/directories between local paths and Tencent COS paths.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  ./cos_cp.py ./data cos://wangweiyun-1306757789/backup/20260702/data/
  ./cos_cp.py cos://wangweiyun-1306757789/backup/20260702/data/ ./restore/
  ./cos_cp.py ./file.bin cos://wangweiyun-1306757789/backup/file.bin
  ./cos_cp.py cos://wangweiyun-1306757789/backup/file.bin ./file.bin
  ./cos_cp.py cos://src-bucket-1306757789/backup/a/ cos://dst-bucket-1306757789/backup/b/

COS environment:
  COS_SECRET_ID      required for COS paths
  COS_SECRET_KEY     required for COS paths
  COS_REGION         default: ap-beijing
  COS_ENDPOINT       optional, e.g. https://cos.ap-beijing.myqcloud.com

The script also loads these keys from ./.env when they are not already set in
the process environment.
""",
    )
    parser.add_argument("src", help="source path; use cos://... for Tencent COS")
    parser.add_argument("dst", help="destination path; use cos://... for Tencent COS")
    parser.add_argument("--region", default=os.environ.get("COS_REGION", DEFAULT_REGION))
    parser.add_argument("--endpoint", default=os.environ.get("COS_ENDPOINT"))
    parser.add_argument("--part-size", type=int, default=64, help="multipart part size in MB")
    parser.add_argument("--threads", type=int, default=16, help="parallel SDK worker threads")
    parser.add_argument(
        "--force",
        action="store_true",
        help="redownload COS objects to local paths even when the local size matches",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=3,
        help="whole-file retry attempts for COS downloads after SDK part retries fail",
    )
    parser.add_argument(
        "--download-retry-delay",
        type=float,
        default=5.0,
        help="seconds to wait between whole-file COS download retries",
    )
    parser.add_argument("--dry-run", action="store_true", help="print planned operations only")
    return parser


def make_cos_store(args: argparse.Namespace, src: LocalPath | CosPath, dst: LocalPath | CosPath) -> CosStore | None:
    needs_cos = isinstance(src, CosPath) or (isinstance(dst, CosPath) and not args.dry_run)
    if not needs_cos:
        return None

    secret_id = os.environ.get("COS_SECRET_ID")
    secret_key = os.environ.get("COS_SECRET_KEY")
    if not secret_id or not secret_key:
        raise RuntimeError(
            "COS_SECRET_ID and COS_SECRET_KEY must be set when either path uses cos://"
        )

    return CosStore(
        secret_id=secret_id,
        secret_key=secret_key,
        region=args.region,
        endpoint=normalize_endpoint(args.endpoint),
        part_size_mb=args.part_size,
        threads=args.threads,
    )


def run(argv: list[str] | None = None) -> int:
    load_dotenv(Path.cwd() / ".env")
    args = build_parser().parse_args(argv)
    src = parse_path(args.src)
    dst = parse_path(args.dst)
    cos = make_cos_store(args, src, dst)
    source = resolve_source(src, cos)
    counters = Counters()

    if isinstance(source, LocalFile) and isinstance(dst, LocalPath):
        copy_local_file_to_local(source, dst, args.dry_run, counters)
    elif isinstance(source, LocalDir) and isinstance(dst, LocalPath):
        copy_local_dir_to_local(source, dst, args.dry_run, counters)
    elif isinstance(source, LocalFile) and isinstance(dst, CosPath):
        copy_local_file_to_cos(source, dst, cos, args.dry_run, counters)
    elif isinstance(source, LocalDir) and isinstance(dst, CosPath):
        copy_local_dir_to_cos(source, dst, cos, args.dry_run, counters)
    elif isinstance(source, CosObject) and isinstance(dst, LocalPath):
        assert cos is not None
        copy_cos_object_to_local(
            source,
            dst,
            cos,
            args.dry_run,
            args.force,
            args.download_retries,
            args.download_retry_delay,
            counters,
        )
    elif isinstance(source, CosPrefix) and isinstance(dst, LocalPath):
        assert cos is not None
        copy_cos_prefix_to_local(
            source,
            dst,
            cos,
            args.dry_run,
            args.force,
            args.download_retries,
            args.download_retry_delay,
            counters,
        )
    elif isinstance(source, CosObject) and isinstance(dst, CosPath):
        assert cos is not None
        copy_cos_object_to_cos(source, dst, cos, args.dry_run, counters)
    elif isinstance(source, CosPrefix) and isinstance(dst, CosPath):
        assert cos is not None
        copy_cos_prefix_to_cos(source, dst, cos, args.dry_run, counters)
    else:
        raise RuntimeError(f"Unsupported copy combination: {type(source).__name__} -> {type(dst).__name__}")

    verb = "planned" if args.dry_run else "done"
    summary = f"{verb}: {counters.files} file(s), {counters.bytes} byte(s)"
    if counters.skipped_files:
        summary += f"; skipped: {counters.skipped_files} file(s), {counters.skipped_bytes} byte(s)"
    print(summary)
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
