"""Hermetic tests for production supply-chain and release-evidence gates."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import supply_chain  # noqa: E402


def _locked_inputs(tmp_path: Path) -> dict[str, object]:
    torch_hash = "1" * 64
    other_hash = "2" * 64
    lock_path = tmp_path / "requirements.lock"
    lock_path.write_text(
        "\n".join(
            (
                "--extra-index-url https://download.pytorch.org/whl/cu999",
                f"torch==9.8.7+cu999 --hash=sha256:{torch_hash}",
                f"example-runtime==1.2.3 --hash=sha256:{other_hash}",
                "",
            )
        ),
        encoding="utf-8",
    )
    lock_digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    archive = tmp_path / "qdrant.tar.gz"
    archive.write_bytes(b"fixture qdrant archive")
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum_evidence = tmp_path / "SHA256SUMS"
    checksum_evidence.write_text(
        f"{archive_digest}  qdrant-x86_64-unknown-linux-musl.tar.gz\n",
        encoding="utf-8",
    )
    checksum_evidence_digest = hashlib.sha256(
        checksum_evidence.read_bytes()
    ).hexdigest()
    worker_artifact = tmp_path / "known-good-worker.json"
    worker_artifact.write_text(
        json.dumps({"fixture": "observed worker identity"}), encoding="utf-8"
    )
    worker_artifact_digest = hashlib.sha256(worker_artifact.read_bytes()).hexdigest()
    base_image = f"registry.example/python@sha256:{'a' * 64}"
    version = "v9.8.7"
    archive_url = supply_chain.qdrant_archive_url(version)
    checksum_source_url = (
        "https://github.com/qdrant/qdrant/releases/download/v9.8.7/SHA256SUMS"
    )
    identity = {
        "schema_version": 1,
        "status": "validated",
        "platform": "linux/amd64",
        "base_image": base_image,
        "python_version": "3.11.99",
        "torch_version": "9.8.7+cu999",
        "torch_wheel_sha256": torch_hash,
        "cuda_build": "cu999",
        "cuda_runtime_version": "99.9",
        "pytorch_index_url": "https://download.pytorch.org/whl/cu999",
        "requirements_lock_sha256": lock_digest,
        "qdrant_version": version,
        "qdrant_archive_url": archive_url,
        "qdrant_archive_sha256": archive_digest,
        "qdrant_checksum_source_url": checksum_source_url,
        "qdrant_checksum_evidence_sha256": checksum_evidence_digest,
        "known_good_worker_artifact_sha256": worker_artifact_digest,
    }
    identity_path = tmp_path / "runtime-identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    return {
        "base_image": base_image,
        "qdrant_version": version,
        "qdrant_archive_url_value": archive_url,
        "qdrant_archive_sha256": archive_digest,
        "qdrant_checksum_source_url": checksum_source_url,
        "qdrant_checksum_evidence": checksum_evidence,
        "qdrant_archive": archive,
        "requirements_lock": lock_path,
        "runtime_identity": identity_path,
        "known_good_worker_artifact": worker_artifact,
    }


@pytest.mark.parametrize(
    "reference",
    (
        "python:3.11-slim-bookworm",
        "python:latest",
        "sha256:" + "a" * 64,
        "python@sha256:" + "A" * 64,
        "python@sha256:short",
    ),
)
def test_base_image_must_be_repository_digest(reference):
    with pytest.raises(supply_chain.SupplyChainError, match="base image"):
        supply_chain.validate_base_image_reference(reference)


def test_build_inputs_bind_all_observed_evidence_before_use(tmp_path):
    inputs = _locked_inputs(tmp_path)

    result = supply_chain.validate_build_inputs(**inputs)

    assert result["base_image_digest"] == "a" * 64
    assert result["qdrant_archive_sha256"] == inputs["qdrant_archive_sha256"]
    assert result["torch_version"] == "9.8.7+cu999"
    assert result["cuda_runtime_version"] == "99.9"


def test_qdrant_archive_is_rehashed_before_acceptance(tmp_path):
    inputs = _locked_inputs(tmp_path)
    inputs["qdrant_archive"].write_bytes(b"tampered after evidence capture")

    with pytest.raises(supply_chain.SupplyChainError, match="checksum mismatch"):
        supply_chain.validate_build_inputs(**inputs)


def test_qdrant_archive_must_match_official_checksum_entry(tmp_path):
    inputs = _locked_inputs(tmp_path)
    checksum_path = inputs["qdrant_checksum_evidence"]
    checksum_path.write_text(
        f"{'f' * 64}  qdrant-x86_64-unknown-linux-musl.tar.gz\n",
        encoding="utf-8",
    )
    identity_path = inputs["runtime_identity"]
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["qdrant_checksum_evidence_sha256"] = hashlib.sha256(
        checksum_path.read_bytes()
    ).hexdigest()
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(supply_chain.SupplyChainError, match="official checksum"):
        supply_chain.validate_build_inputs(**inputs)


def test_qdrant_checksum_asset_entry_must_be_unambiguous(tmp_path):
    inputs = _locked_inputs(tmp_path)
    checksum_path = inputs["qdrant_checksum_evidence"]
    digest = inputs["qdrant_archive_sha256"]
    checksum_path.write_text(
        "".join(
            (
                f"{digest}  qdrant-x86_64-unknown-linux-musl.tar.gz\n",
                f"{digest} *qdrant-x86_64-unknown-linux-musl.tar.gz\n",
            )
        ),
        encoding="utf-8",
    )
    identity_path = inputs["runtime_identity"]
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["qdrant_checksum_evidence_sha256"] = hashlib.sha256(
        checksum_path.read_bytes()
    ).hexdigest()
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(supply_chain.SupplyChainError, match="exactly one entry"):
        supply_chain.validate_build_inputs(**inputs)


def test_qdrant_checksum_evidence_is_digest_bound(tmp_path):
    inputs = _locked_inputs(tmp_path)
    inputs["qdrant_checksum_evidence"].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(
        supply_chain.SupplyChainError, match="checksum evidence checksum mismatch"
    ):
        supply_chain.validate_build_inputs(**inputs)


def test_known_good_worker_evidence_is_rehashed_before_acceptance(tmp_path):
    inputs = _locked_inputs(tmp_path)
    inputs["known_good_worker_artifact"].write_text("tampered", encoding="utf-8")

    with pytest.raises(supply_chain.SupplyChainError, match="worker evidence checksum"):
        supply_chain.validate_build_inputs(**inputs)


def test_qdrant_checksum_evidence_must_be_official_and_version_specific(tmp_path):
    inputs = _locked_inputs(tmp_path)
    inputs["qdrant_checksum_source_url"] = "https://example.com/v9.8.7.sha256"

    with pytest.raises(supply_chain.SupplyChainError, match="checksum source"):
        supply_chain.validate_build_inputs(**inputs)


def test_floating_development_requirements_are_rejected():
    path = Path(__file__).resolve().parents[1] / "serverless" / "requirements.txt"

    with pytest.raises(supply_chain.SupplyChainError, match="exact name==version"):
        supply_chain.parse_requirements_lock(path)


def test_unconfigured_runtime_identity_is_a_release_blocker(tmp_path):
    inputs = _locked_inputs(tmp_path)
    unconfigured = (
        Path(__file__).resolve().parents[1]
        / "serverless"
        / "runtime-identity.unconfigured.json"
    )
    inputs["runtime_identity"] = unconfigured

    with pytest.raises(supply_chain.SupplyChainError, match="unconfigured"):
        supply_chain.validate_build_inputs(**inputs)


def test_runtime_identity_must_match_complete_lock_hash(tmp_path):
    inputs = _locked_inputs(tmp_path)
    inputs["requirements_lock"].write_text(
        inputs["requirements_lock"].read_text(encoding="utf-8")
        + f"late-package==4.5.6 --hash=sha256:{'3' * 64}\n",
        encoding="utf-8",
    )

    with pytest.raises(supply_chain.SupplyChainError, match="lock SHA-256"):
        supply_chain.validate_build_inputs(**inputs)


def test_torch_lock_authorizes_only_the_validated_cuda_wheel(tmp_path):
    inputs = _locked_inputs(tmp_path)
    lock_path = inputs["requirements_lock"]
    lock_path.write_text(
        lock_path.read_text(encoding="utf-8").replace(
            f"--hash=sha256:{'1' * 64}",
            f"--hash=sha256:{'1' * 64} --hash=sha256:{'4' * 64}",
            1,
        ),
        encoding="utf-8",
    )
    identity_path = inputs["runtime_identity"]
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["requirements_lock_sha256"] = hashlib.sha256(
        lock_path.read_bytes()
    ).hexdigest()
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(supply_chain.SupplyChainError, match="exactly the one"):
        supply_chain.validate_build_inputs(**inputs)


def test_installed_python_torch_cuda_and_packages_are_reverified(
    monkeypatch, tmp_path
):
    inputs = _locked_inputs(tmp_path)
    versions = {"torch": "9.8.7+cu999", "example-runtime": "1.2.3"}
    monkeypatch.setattr(supply_chain.platform, "python_version", lambda: "3.11.99")
    monkeypatch.setattr(
        supply_chain.importlib.metadata, "version", lambda name: versions[name]
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(__version__="9.8.7+cu999", version=SimpleNamespace(cuda="99.9")),
    )

    result = supply_chain.verify_installed_runtime(
        requirements_lock=inputs["requirements_lock"],
        runtime_identity=inputs["runtime_identity"],
    )

    assert result["python_version"] == "3.11.99"
    assert result["torch_version"] == "9.8.7+cu999"
    assert result["cuda_runtime_version"] == "99.9"


def test_dockerfile_wires_fail_closed_checks_before_extract_and_install():
    dockerfile = (
        Path(__file__).resolve().parents[1] / "serverless" / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert "FROM python:" not in dockerfile
    assert "ARG PYTHON_BASE_IMAGE\nFROM --platform=linux/amd64 ${PYTHON_BASE_IMAGE}" in dockerfile
    assert "curl " not in dockerfile
    assert "apt-get" not in dockerfile
    validate_at = dockerfile.index("validate-build-inputs")
    extract_at = dockerfile.index("tar xzf")
    install_at = dockerfile.index("pip install")
    assert validate_at < extract_at < install_at
    assert "--require-hashes --only-binary=:all:" in dockerfile
    assert "verify-installed-runtime" in dockerfile
    assert "--known-good-worker-artifact" in dockerfile
    assert "--qdrant-checksum-evidence" in dockerfile


def test_release_audit_is_offline_owner_only_and_checksum_bound(tmp_path):
    digest = "c" * 64
    image = f"registry.example/legal-search@sha256:{digest}"
    output = tmp_path / "evidence" / "release-1"
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_runner(command, environment, output_path):
        calls.append((list(command), dict(environment)))
        if command[:3] == ["docker", "image", "inspect"]:
            return json.dumps(
                [
                    {
                        "Id": "sha256:" + "d" * 64,
                        "RepoDigests": [image],
                    }
                ]
            )
        if command[0] == "syft" and output_path is not None:
            supply_chain._atomic_write(
                output_path, b'{"bomFormat":"CycloneDX","components":[]}'
            )
            return ""
        if command[0] == "grype" and output_path is not None:
            supply_chain._atomic_write(
                output_path,
                json.dumps(
                    {
                        "matches": [
                            {"vulnerability": {"severity": "High"}},
                            {"vulnerability": {"severity": "Critical"}},
                        ]
                    }
                ).encode(),
            )
            return ""
        return f"{command[0]} fixture-version"

    provenance = supply_chain.emit_release_audit(
        image_reference=image,
        output_dir=output,
        runner=fake_runner,
    )

    assert provenance["execution_mode"] == "offline-local-image"
    assert provenance["vulnerability_counts"] == {"critical": 1, "high": 1}
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for name in ("sbom.cdx.json", "vulnerabilities.json", "provenance.json"):
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o600
    assert provenance["artifacts"]["sbom.cdx.json"] == hashlib.sha256(
        (output / "sbom.cdx.json").read_bytes()
    ).hexdigest()
    for _, environment in calls:
        assert environment["SYFT_CHECK_FOR_APP_UPDATE"] == "false"
        assert environment["GRYPE_CHECK_FOR_APP_UPDATE"] == "false"
        assert environment["GRYPE_DB_AUTO_UPDATE"] == "false"
        assert environment["GRYPE_DB_VALIDATE_BY_HASH_ON_START"] == "true"


def test_release_audit_rejects_unpinned_image_without_running_tools(tmp_path):
    called = False

    def fail_runner(command, environment, output_path):
        nonlocal called
        called = True
        raise AssertionError((command, environment, output_path))

    with pytest.raises(supply_chain.SupplyChainError, match="base image"):
        supply_chain.emit_release_audit(
            image_reference="registry.example/legal-search:latest",
            output_dir=tmp_path / "evidence",
            runner=fail_runner,
        )
    assert called is False
