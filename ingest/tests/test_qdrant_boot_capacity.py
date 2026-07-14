from types import SimpleNamespace

import pytest

from serverless import qdrant_boot


def test_container_disk_capacity_fails_fast_when_underprovisioned(monkeypatch, tmp_path):
    monkeypatch.setattr(qdrant_boot, "STORAGE_ROOT", tmp_path / "qdrant-data")
    monkeypatch.setattr(qdrant_boot, "MIN_CONTAINER_DISK_GB", 64)
    monkeypatch.setattr(
        qdrant_boot.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=20_000_000_000, used=1, free=19_999_999_999),
    )

    with pytest.raises(RuntimeError, match="requires at least 64 GB"):
        qdrant_boot._ensure_container_disk_capacity()


def test_container_disk_capacity_accepts_provisioned_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(qdrant_boot, "STORAGE_ROOT", tmp_path / "qdrant-data")
    monkeypatch.setattr(qdrant_boot, "MIN_CONTAINER_DISK_GB", 64)
    monkeypatch.setattr(
        qdrant_boot.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=80_000_000_000, used=8, free=72_000_000_000),
    )

    qdrant_boot._ensure_container_disk_capacity()
