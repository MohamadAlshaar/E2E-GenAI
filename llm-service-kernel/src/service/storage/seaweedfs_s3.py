### not used #########
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError


@dataclass(frozen=True)
class SeaweedFSS3Config:
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    region_name: str = "us-east-1"
    use_ssl: bool = False
    verify: bool | str = True
    create_bucket_if_missing: bool = True
    max_pool_connections: int = 32
    connect_timeout_seconds: int = 5
    read_timeout_seconds: int = 120
    prefix: str = ""

    @classmethod
    def from_env(cls, prefix: str = "OBJECT_STORE_") -> "SeaweedFSS3Config":
        endpoint_url = os.environ[f"{prefix}ENDPOINT_URL"].strip()
        access_key_id = os.environ[f"{prefix}ACCESS_KEY_ID"].strip()
        secret_access_key = os.environ[f"{prefix}SECRET_ACCESS_KEY"].strip()
        bucket = os.environ[f"{prefix}BUCKET"].strip()

        verify_raw = os.environ.get(f"{prefix}VERIFY", "true").strip()
        if verify_raw.lower() in {"1", "true", "yes", "on"}:
            verify: bool | str = True
        elif verify_raw.lower() in {"0", "false", "no", "off"}:
            verify = False
        else:
            verify = verify_raw

        return cls(
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket=bucket,
            region_name=os.environ.get(f"{prefix}REGION", "us-east-1").strip(),
            use_ssl=os.environ.get(
                f"{prefix}USE_SSL", "false"
            ).strip().lower() in {"1", "true", "yes", "on"},
            verify=verify,
            create_bucket_if_missing=os.environ.get(
                f"{prefix}CREATE_BUCKET_IF_MISSING", "true"
            ).strip().lower() in {"1", "true", "yes", "on"},
            max_pool_connections=int(
                os.environ.get(f"{prefix}MAX_POOL_CONNECTIONS", "32")
            ),
            connect_timeout_seconds=int(
                os.environ.get(f"{prefix}CONNECT_TIMEOUT_SECONDS", "5")
            ),
            read_timeout_seconds=int(
                os.environ.get(f"{prefix}READ_TIMEOUT_SECONDS", "120")
            ),
            prefix=os.environ.get(f"{prefix}PREFIX", "").strip().strip("/"),
        )


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip().strip("/")


def _join_key(prefix: str, key: str) -> str:
    clean_key = key.lstrip("/")
    clean_prefix = _normalize_prefix(prefix)
    if not clean_prefix:
        return clean_key
    if not clean_key:
        return clean_prefix
    return f"{clean_prefix}/{clean_key}"


class SeaweedFSObjectStore:
    def __init__(self, config: SeaweedFSS3Config) -> None:
        self.config = config
        session = boto3.session.Session()
        self.client: BaseClient = session.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            region_name=config.region_name,
            use_ssl=config.use_ssl,
            verify=config.verify,
            config=Config(
                region_name=config.region_name,
                signature_version="s3v4",
                max_pool_connections=config.max_pool_connections,
                connect_timeout=config.connect_timeout_seconds,
                read_timeout=config.read_timeout_seconds,
                tcp_keepalive=True,
                retries={"max_attempts": 3, "mode": "standard"},
                s3={
                    "addressing_style": "path",
                    "payload_signing_enabled": False,
                },
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                user_agent_extra="llm-service-kernel-seaweedfs",
            ),
        )

    @property
    def bucket(self) -> str:
        return self.config.bucket

    def ensure_bucket(self) -> None:
        if not self.config.create_bucket_if_missing:
            return

        try:
            self.client.head_bucket(Bucket=self.bucket)
            return
        except ClientError as exc:
            status = int(
                exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            )
            code = str(exc.response.get("Error", {}).get("Code", ""))

            # On some S3-compatible backends, a missing or inaccessible bucket
            # can surface as 403 rather than 404 during HeadBucket.
            if status not in {403, 404} and code not in {
                "403",
                "404",
                "Forbidden",
                "NoSuchBucket",
                "NotFound",
            }:
                raise

        try:
            self.client.create_bucket(Bucket=self.bucket)
        except ClientError as exc:
            status = int(
                exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            )
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if status in {200, 409} or code in {
                "BucketAlreadyOwnedByYou",
                "BucketAlreadyExists",
            }:
                return
            raise

    def object_exists(self, key: str) -> bool:
        full_key = _join_key(self.config.prefix, key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except ClientError as exc:
            status = int(
                exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            )
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if status == 404 or code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def list_objects(self, prefix: str = "") -> list[dict[str, Any]]:
        full_prefix = (
            _join_key(self.config.prefix, prefix)
            if prefix
            else _normalize_prefix(self.config.prefix)
        )
        paginator = self.client.get_paginator("list_objects_v2")
        results: list[dict[str, Any]] = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                results.append(
                    {
                        "key": obj["Key"],
                        "size": int(obj["Size"]),
                        "etag": obj.get("ETag", "").strip('"'),
                        "last_modified": (
                            obj.get("LastModified").isoformat()
                            if obj.get("LastModified")
                            else None
                        ),
                    }
                )
        return results

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        full_key = _join_key(self.config.prefix, key)
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": full_key,
            "Body": data,
        }
        if content_type:
            kwargs["ContentType"] = content_type
        if metadata:
            kwargs["Metadata"] = metadata
        return self.client.put_object(**kwargs)

    def put_json(self, key: str, payload: Any) -> dict[str, Any]:
        data = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        return self.put_bytes(key, data, content_type="application/json")

    def get_bytes(self, key: str) -> bytes:
        full_key = _join_key(self.config.prefix, key)
        response = self.client.get_object(Bucket=self.bucket, Key=full_key)
        return response["Body"].read()

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key).decode("utf-8"))

    def upload_file(
        self,
        local_path: str | Path,
        key: str,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        path = Path(local_path)
        guessed_content_type = (
            content_type
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )
        with path.open("rb") as fh:
            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Key": _join_key(self.config.prefix, key),
                "Body": fh,
                "ContentType": guessed_content_type,
            }
            if metadata:
                kwargs["Metadata"] = metadata
            return self.client.put_object(**kwargs)

    def download_file(self, key: str, local_path: str | Path) -> Path:
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        full_key = _join_key(self.config.prefix, key)

        with path.open("wb") as fh:
            response = self.client.get_object(Bucket=self.bucket, Key=full_key)
            fh.write(response["Body"].read())

        return path

    def upload_directory(
        self,
        local_dir: str | Path,
        remote_prefix: str = "",
    ) -> list[dict[str, Any]]:
        base = Path(local_dir)
        uploaded: list[dict[str, Any]] = []

        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            relative = path.relative_to(base).as_posix()
            key = _join_key(remote_prefix, relative)
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

            self.upload_file(
                path,
                key,
                metadata={
                    "sha256": sha256,
                    "source_path": str(path),
                },
            )
            uploaded.append(
                {
                    "local_path": str(path),
                    "key": _join_key(self.config.prefix, key),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256,
                }
            )

        return uploaded

