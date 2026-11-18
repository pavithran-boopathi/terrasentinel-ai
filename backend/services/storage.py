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
    def save_stream(self, key: str, stream: BinaryIO, max_bytes: int) -> tuple[str, str, int]: ...
    def put_file(self, key: str, path: Path) -> str: ...
    def local_path(self, uri: str) -> Path: ...
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


def get_storage() -> LocalStorage:
    from backend.core.config import get_settings

    return LocalStorage(get_settings().storage_root)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
