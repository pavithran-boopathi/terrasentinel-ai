"""Object storage abstraction. Local filesystem now; an S3 backend plugs in behind the same API.

URIs look like ``local://images/<uuid>/scene.tif``; callers never build filesystem paths.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import BinaryIO, Protocol


class StorageError(Exception):
    pass


class Storage(Protocol):
    def uri(self, key: str) -> str: ...
    def save_stream(self, key: str, stream: BinaryIO, max_bytes: int) -> tuple[str, str, int]: ...
    def put_file(self, key: str, path: Path) -> str: ...
    def local_path(self, uri: str) -> Path: ...
    def workdir(self, prefix: str) -> Path: ...
    def commit_dir(self, prefix: str) -> None: ...
    def list_names(self, prefix_uri: str) -> list[str]: ...
    def delete_prefix(self, prefix: str) -> None: ...


class LocalStorage:
    scheme = "local://"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if p != self.root and self.root not in p.parents:
            raise StorageError(f"key escapes storage root: {key!r}")
        return p

    def uri(self, key: str) -> str:
        return self.scheme + key

    def save_stream(self, key: str, stream: BinaryIO, max_bytes: int) -> tuple[str, str, int]:
        """Stream to disk with a size cap → (uri, sha256, size). Partial files are removed."""
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        h, n = hashlib.sha256(), 0
        try:
            with tmp.open("wb") as f:
                while chunk := stream.read(1 << 20):
                    n += len(chunk)
                    if n > max_bytes:
                        raise StorageError(f"upload exceeds {max_bytes} bytes")
                    h.update(chunk)
                    f.write(chunk)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        return self.uri(key), h.hexdigest(), n

    def put_file(self, key: str, path: Path) -> str:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        return self.uri(key)

    def local_path(self, uri: str) -> Path:
        if not uri.startswith(self.scheme):
            raise StorageError(f"not a local storage URI: {uri}")
        return self._path(uri.removeprefix(self.scheme))

    def delete_prefix(self, prefix: str) -> None:
        p = self._path(prefix)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    def workdir(self, prefix: str) -> Path:
        """Directory to write outputs for ``prefix`` (the final location for local storage)."""
        p = self._path(prefix)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def commit_dir(self, prefix: str) -> None:  # already in place
        return None

    def list_names(self, prefix_uri: str) -> list[str]:
        p = self.local_path(prefix_uri)
        return sorted(f.name for f in p.iterdir() if f.is_file()) if p.is_dir() else []


class S3Storage:
    """S3 backend. Files are read/written through a local cache so GDAL/rasterio and PyTorch
    work on ordinary paths; ``commit_dir`` uploads a finished output directory."""

    scheme = "s3://"

    def __init__(self, bucket: str, cache_root: Path, client=None) -> None:
        import boto3

        self.bucket = bucket
        self.cache = LocalStorage(cache_root)
        self.s3 = client or boto3.client("s3")

    def uri(self, key: str) -> str:
        return f"{self.scheme}{self.bucket}/{key}"

    def _key(self, uri: str) -> str:
        prefix = f"{self.scheme}{self.bucket}/"
        if not uri.startswith(prefix):
            raise StorageError(f"not a URI in bucket {self.bucket}: {uri}")
        key = uri.removeprefix(prefix)
        self.cache._path(key)  # same traversal guard as local storage
        return key

    def save_stream(self, key: str, stream: BinaryIO, max_bytes: int) -> tuple[str, str, int]:
        local_uri, sha, n = self.cache.save_stream(key, stream, max_bytes)
        self.s3.upload_file(str(self.cache.local_path(local_uri)), self.bucket, key)
        return self.uri(key), sha, n

    def put_file(self, key: str, path: Path) -> str:
        self.cache._path(key)
        self.s3.upload_file(str(path), self.bucket, key)
        return self.uri(key)

    def local_path(self, uri: str) -> Path:
        key = self._key(uri)
        dest = self.cache._path(key)
        if dest.exists() and dest.is_dir():
            return dest
        if not dest.exists():
            names = self.list_names(uri)
            if names:  # a "directory" prefix: materialise its files
                dest.mkdir(parents=True, exist_ok=True)
                for n in names:
                    self.s3.download_file(self.bucket, f"{key}/{n}", str(dest / n))
                return dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, key, str(dest))
        return dest

    def workdir(self, prefix: str) -> Path:
        return self.cache.workdir(prefix)

    def commit_dir(self, prefix: str) -> None:
        base = self.cache.workdir(prefix)
        for f in base.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                self.s3.upload_file(str(f), self.bucket, f"{prefix}/{f.relative_to(base)}")

    def list_names(self, prefix_uri: str) -> list[str]:
        key = self._key(prefix_uri).rstrip("/") + "/"
        out = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=key, Delimiter="/"
        ):
            out += [o["Key"].removeprefix(key) for o in page.get("Contents", [])]
        return sorted(out)

    def delete_prefix(self, prefix: str) -> None:
        for page in self.s3.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix
        ):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})
        self.cache.delete_prefix(prefix)


def get_storage() -> LocalStorage | S3Storage:
    from backend.core.config import get_settings

    s = get_settings()
    if s.storage_backend == "s3":
        if not s.s3_bucket:
            raise StorageError("TS_S3_BUCKET must be set when TS_STORAGE_BACKEND=s3")
        return S3Storage(s.s3_bucket, s.storage_root / "cache")
    return LocalStorage(s.storage_root)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
