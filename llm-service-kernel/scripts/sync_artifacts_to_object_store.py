from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.service.storage import SeaweedFSS3Config, SeaweedFSObjectStore


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_mapping(value: str) -> tuple[Path, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Each --map must look like LOCAL_PATH=REMOTE_PREFIX"
        )
    local, remote = value.split("=", 1)
    return Path(local).expanduser().resolve(), remote.strip().strip("/")


def iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload benchmark artifacts to SeaweedFS S3."
    )
    parser.add_argument(
        "--run-name",
        default=datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ"),
        help="Top-level logical run name inside the bucket.",
    )
    parser.add_argument(
        "--manifest-out",
        default=str(
            ROOT / "benchmark_data" / "artifact_manifests" / "latest_sync_manifest.json"
        ),
        help="Local path to write the sync manifest JSON.",
    )
    parser.add_argument(
        "--map",
        action="append",
        type=parse_mapping,
        default=[],
        help="Mapping of LOCAL_PATH=REMOTE_PREFIX. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing local paths instead of failing.",
    )
    return parser.parse_args()


def default_mappings() -> list[tuple[Path, str]]:
    return [
        (ROOT / "benchmark_data" / "workloads", "workloads"),
        (ROOT / "benchmark_data" / "manifests", "manifests"),
        (ROOT / "benchmark_data" / "results", "results"),
        (ROOT / "state", "state"),
    ]


def main() -> int:
    args = parse_args()
    mappings = args.map or default_mappings()

    config = SeaweedFSS3Config.from_env()
    store = SeaweedFSObjectStore(config)
    store.ensure_bucket()

    uploaded: list[dict[str, Any]] = []
    skipped: list[str] = []

    root_prefix = f"benchmarks/{args.run_name}"

    for local_path, remote_prefix in mappings:
        if not local_path.exists():
            if args.skip_missing:
                skipped.append(str(local_path))
                continue
            raise FileNotFoundError(f"Missing path: {local_path}")

        files = iter_files(local_path)
        base_dir = local_path.parent if local_path.is_file() else local_path

        for file_path in files:
            relative = file_path.relative_to(base_dir).as_posix()
            object_key = "/".join(
                part
                for part in [root_prefix, remote_prefix.strip("/"), relative]
                if part
            )
            digest = sha256_file(file_path)

            store.upload_file(
                file_path,
                object_key,
                metadata={
                    "sha256": digest,
                    "source_path": str(file_path),
                },
            )

            stat = file_path.stat()
            uploaded.append(
                {
                    "local_path": str(file_path),
                    "object_key": object_key,
                    "full_object_key": "/".join(
                        part
                        for part in [config.prefix.strip("/"), object_key]
                        if part
                    ),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": digest,
                }
            )

    manifest = {
        "bucket": config.bucket,
        "endpoint_url": config.endpoint_url,
        "base_prefix": config.prefix,
        "root_prefix": root_prefix,
        "run_name": args.run_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "uploaded_count": len(uploaded),
        "skipped": skipped,
        "uploaded": uploaded,
    }

    manifest_path = Path(args.manifest_out).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest_object_key = f"{root_prefix}/sync_manifest.json"
    store.put_json(manifest_object_key, manifest)

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "manifest_object_key": manifest_object_key,
                "uploaded_count": len(uploaded),
                "skipped_count": len(skipped),
                "bucket": config.bucket,
                "endpoint_url": config.endpoint_url,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
