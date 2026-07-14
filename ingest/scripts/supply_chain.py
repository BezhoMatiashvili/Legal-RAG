#!/usr/bin/env python3
"""Fail-closed, offline validation for the serverless release artifact.

This module intentionally contains no project or third-party imports.  It is copied into
the worker image before dependencies are installed, so an unpinned base image, an
unverified Qdrant archive, or a floating requirements file stops the build before archive
extraction or package installation.

The release-audit command only inspects an image already present in the local Docker
daemon.  Syft update checks and Grype application/database updates are disabled; a missing
local image, scanner, or vulnerability database is an external release-gate failure rather
than permission to download anything.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

SCHEMA_VERSION = 1
TARGET_PLATFORM = "linux/amd64"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_QDRANT_VERSION_RE = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
_PYTHON_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_CUDA_BUILD_RE = re.compile(r"^cu[0-9]{3}$")
_CUDA_RUNTIME_RE = re.compile(r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?$")
_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)"
    r"(?:\[[A-Za-z0-9_,.-]+\])?=="
    r"(?P<version>[A-Za-z0-9][A-Za-z0-9._+!-]*)"
    r"(?P<hashes>(?:\s+--hash=sha256:[0-9a-f]{64})+)$"
)
_LOCK_OPTION_RE = re.compile(r"^--(?P<name>index-url|extra-index-url)\s+(?P<value>\S+)$")
_CHECKSUM_LINE_RE = re.compile(
    r"^(?P<digest>[0-9a-f]{64})\s+[ *](?P<asset>\S+)$"
)
_BSD_CHECKSUM_LINE_RE = re.compile(
    r"^SHA256 \((?P<asset>[^)]+)\) = (?P<digest>[0-9a-f]{64})$"
)


class SupplyChainError(ValueError):
    """A bounded, actionable release-input validation failure."""


@dataclass(frozen=True)
class LockedRequirement:
    name: str
    version: str
    hashes: tuple[str, ...]


@dataclass(frozen=True)
class RequirementsLock:
    requirements: dict[str, LockedRequirement]
    index_urls: tuple[str, ...]
    sha256: str


def _normalise_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise SupplyChainError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(chunk_size), b""):
                digest.update(block)
    except OSError as exc:
        raise SupplyChainError(f"cannot read release input {path}: {exc}") from exc
    return digest.hexdigest()


def validate_base_image_reference(reference: str) -> str:
    """Return the digest from an immutable OCI reference, rejecting tags and IDs."""
    if not isinstance(reference, str) or reference.count("@sha256:") != 1:
        raise SupplyChainError(
            "base image must be an explicit repository@sha256:<64 lowercase hex> reference"
        )
    repository, digest = reference.rsplit("@sha256:", maxsplit=1)
    if (
        not repository
        or repository.startswith("-")
        or any(char.isspace() for char in repository)
        or repository.startswith("sha256:")
    ):
        raise SupplyChainError("base image repository is invalid")
    return _require_sha256(digest, "base image digest")


def qdrant_archive_url(version: str) -> str:
    if not _QDRANT_VERSION_RE.fullmatch(version or ""):
        raise SupplyChainError("Qdrant version must be explicit vMAJOR.MINOR.PATCH")
    return (
        "https://github.com/qdrant/qdrant/releases/download/"
        f"{version}/qdrant-x86_64-unknown-linux-musl.tar.gz"
    )


def validate_qdrant_release(
    *, version: str, archive_url: str, archive_sha256: str, checksum_source_url: str
) -> None:
    """Require the exact official release URL plus separately recorded checksum evidence."""
    expected_url = qdrant_archive_url(version)
    if archive_url != expected_url:
        raise SupplyChainError(
            f"Qdrant archive URL must be the version-matched official release URL: {expected_url}"
        )
    _require_sha256(archive_sha256, "Qdrant archive SHA-256")

    parsed = urlparse(checksum_source_url)
    official_prefix = f"/qdrant/qdrant/releases/download/{version}/"
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or not parsed.path.startswith(official_prefix)
        or parsed.path == official_prefix
        or parsed.path == urlparse(expected_url).path
        or parsed.query
        or parsed.fragment
    ):
        raise SupplyChainError(
            "Qdrant checksum source must be a distinct version-specific HTTPS asset under "
            f"github.com{official_prefix}"
        )


def qdrant_checksum_from_evidence(path: Path, archive_url: str) -> str:
    """Extract the one exact archive checksum from captured official evidence."""
    asset = Path(urlparse(archive_url).path).name
    if not asset:
        raise SupplyChainError("Qdrant archive URL has no asset name")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SupplyChainError(f"cannot read Qdrant checksum evidence {path}: {exc}") from exc
    if not raw or len(raw) > 1024 * 1024:
        raise SupplyChainError(
            "Qdrant checksum evidence must contain between 1 byte and 1 MiB"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SupplyChainError("Qdrant checksum evidence must be UTF-8 text") from exc

    matches: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _CHECKSUM_LINE_RE.fullmatch(line) or _BSD_CHECKSUM_LINE_RE.fullmatch(
            line
        )
        if match is None:
            continue
        named_asset = match.group("asset")
        if named_asset.startswith("./"):
            named_asset = named_asset[2:]
        if named_asset == asset:
            matches.append(match.group("digest"))
    if len(matches) != 1:
        raise SupplyChainError(
            "Qdrant checksum evidence must contain exactly one entry for "
            f"{asset}; observed {len(matches)}"
        )
    return matches[0]


def verify_file_sha256(path: Path, expected_sha256: str, *, label: str) -> str:
    expected = _require_sha256(expected_sha256, f"{label} SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise SupplyChainError(
            f"{label} checksum mismatch: expected {expected}, observed {actual}"
        )
    return actual


def _logical_lock_lines(text: str) -> list[str]:
    logical: list[str] = []
    pending = ""
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        continued = stripped.endswith("\\")
        if continued:
            stripped = stripped[:-1].rstrip()
        pending = f"{pending} {stripped}".strip()
        if continued:
            continue
        if not pending:
            raise SupplyChainError(f"empty logical requirement at line {number}")
        logical.append(pending)
        pending = ""
    if pending:
        raise SupplyChainError("requirements lock ends with an unfinished continuation")
    return logical


def parse_requirements_lock(path: Path) -> RequirementsLock:
    """Parse the deliberately narrow, deterministic subset accepted for production."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SupplyChainError(f"cannot read requirements lock {path}: {exc}") from exc

    requirements: dict[str, LockedRequirement] = {}
    index_urls: list[str] = []
    for line in _logical_lock_lines(text):
        option = _LOCK_OPTION_RE.fullmatch(line)
        if option:
            value = option.group("value")
            parsed = urlparse(value)
            if parsed.scheme != "https" or not parsed.hostname:
                raise SupplyChainError("package indexes must use explicit HTTPS URLs")
            index_urls.append(value.rstrip("/"))
            continue
        if line.startswith("--"):
            raise SupplyChainError(f"unsupported requirements option: {line.split()[0]}")
        if ";" in line or " @ " in line:
            raise SupplyChainError(
                "environment markers and direct references are forbidden in the production lock"
            )
        match = _REQUIREMENT_RE.fullmatch(line)
        if not match:
            raise SupplyChainError(
                "every production requirement must be an exact name==version followed by "
                "one or more --hash=sha256:<digest> entries"
            )
        name = _normalise_name(match.group("name"))
        if name in requirements:
            raise SupplyChainError(f"duplicate locked requirement: {name}")
        hashes = tuple(re.findall(r"--hash=sha256:([0-9a-f]{64})", match.group("hashes")))
        requirements[name] = LockedRequirement(
            name=name,
            version=match.group("version"),
            hashes=hashes,
        )

    if not requirements:
        raise SupplyChainError("production requirements lock contains no packages")
    if "torch" not in requirements:
        raise SupplyChainError("production requirements lock must contain an exact torch wheel")
    if len(set(index_urls)) != len(index_urls):
        raise SupplyChainError("requirements lock contains duplicate package indexes")
    return RequirementsLock(
        requirements=requirements,
        index_urls=tuple(index_urls),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def load_runtime_identity(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupplyChainError(f"cannot read runtime identity {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SupplyChainError("runtime identity must be a JSON object")
    return value


def validate_runtime_identity(
    identity: dict[str, Any],
    lock: RequirementsLock,
    *,
    base_image: str,
    qdrant_version: str,
    qdrant_archive_url_value: str,
    qdrant_archive_sha256: str,
    qdrant_checksum_source_url: str,
) -> None:
    """Bind the lock and build inputs to read-only known-good worker evidence."""
    if identity.get("schema_version") != SCHEMA_VERSION:
        raise SupplyChainError(f"runtime identity schema_version must be {SCHEMA_VERSION}")
    if identity.get("status") != "validated":
        raise SupplyChainError(
            "runtime identity is unconfigured; supply observed known-good worker evidence"
        )
    if identity.get("platform") != TARGET_PLATFORM:
        raise SupplyChainError(f"runtime identity platform must be {TARGET_PLATFORM}")
    if identity.get("base_image") != base_image:
        raise SupplyChainError("runtime identity base_image does not match the build input")
    validate_base_image_reference(base_image)

    if not _PYTHON_VERSION_RE.fullmatch(str(identity.get("python_version", ""))):
        raise SupplyChainError("runtime identity requires an exact Python MAJOR.MINOR.PATCH")
    torch_version = identity.get("torch_version")
    if not isinstance(torch_version, str) or not torch_version:
        raise SupplyChainError("runtime identity requires an exact torch_version")
    cuda_build = identity.get("cuda_build")
    if not isinstance(cuda_build, str) or not _CUDA_BUILD_RE.fullmatch(cuda_build):
        raise SupplyChainError("runtime identity cuda_build must have the form cuNNN")
    cuda_runtime = identity.get("cuda_runtime_version")
    if not isinstance(cuda_runtime, str) or not _CUDA_RUNTIME_RE.fullmatch(cuda_runtime):
        raise SupplyChainError("runtime identity requires an exact CUDA runtime version")
    if not torch_version.endswith(f"+{cuda_build}"):
        raise SupplyChainError("torch_version local build does not match cuda_build")

    pytorch_index = identity.get("pytorch_index_url")
    expected_index = f"https://download.pytorch.org/whl/{cuda_build}"
    if pytorch_index != expected_index or expected_index not in lock.index_urls:
        raise SupplyChainError(
            "requirements lock and runtime identity must use the CUDA-specific official "
            f"PyTorch index {expected_index}"
        )
    if identity.get("requirements_lock_sha256") != lock.sha256:
        raise SupplyChainError("runtime identity does not match requirements lock SHA-256")

    torch_requirement = lock.requirements["torch"]
    if torch_requirement.version != torch_version:
        raise SupplyChainError("locked torch version does not match runtime identity")
    torch_wheel_sha256 = _require_sha256(
        identity.get("torch_wheel_sha256"), "validated torch wheel SHA-256"
    )
    if torch_requirement.hashes != (torch_wheel_sha256,):
        raise SupplyChainError(
            "torch lock must authorize exactly the one validated CUDA wheel SHA-256"
        )

    _require_sha256(
        identity.get("known_good_worker_artifact_sha256"),
        "known-good worker evidence SHA-256",
    )
    if identity.get("qdrant_version") != qdrant_version:
        raise SupplyChainError("runtime identity Qdrant version does not match build input")
    if identity.get("qdrant_archive_url") != qdrant_archive_url_value:
        raise SupplyChainError("runtime identity Qdrant archive URL does not match build input")
    if identity.get("qdrant_archive_sha256") != qdrant_archive_sha256:
        raise SupplyChainError("runtime identity Qdrant SHA-256 does not match build input")
    if identity.get("qdrant_checksum_source_url") != qdrant_checksum_source_url:
        raise SupplyChainError("runtime identity checksum source does not match build input")
    _require_sha256(
        identity.get("qdrant_checksum_evidence_sha256"),
        "Qdrant checksum evidence SHA-256",
    )
    validate_qdrant_release(
        version=qdrant_version,
        archive_url=qdrant_archive_url_value,
        archive_sha256=qdrant_archive_sha256,
        checksum_source_url=qdrant_checksum_source_url,
    )


def validate_build_inputs(
    *,
    base_image: str,
    qdrant_version: str,
    qdrant_archive_url_value: str,
    qdrant_archive_sha256: str,
    qdrant_checksum_source_url: str,
    qdrant_checksum_evidence: Path,
    qdrant_archive: Path,
    requirements_lock: Path,
    runtime_identity: Path,
    known_good_worker_artifact: Path,
) -> dict[str, Any]:
    base_digest = validate_base_image_reference(base_image)
    validate_qdrant_release(
        version=qdrant_version,
        archive_url=qdrant_archive_url_value,
        archive_sha256=qdrant_archive_sha256,
        checksum_source_url=qdrant_checksum_source_url,
    )
    lock = parse_requirements_lock(requirements_lock)
    identity = load_runtime_identity(runtime_identity)
    validate_runtime_identity(
        identity,
        lock,
        base_image=base_image,
        qdrant_version=qdrant_version,
        qdrant_archive_url_value=qdrant_archive_url_value,
        qdrant_archive_sha256=qdrant_archive_sha256,
        qdrant_checksum_source_url=qdrant_checksum_source_url,
    )
    evidence_digest = verify_file_sha256(
        known_good_worker_artifact,
        identity["known_good_worker_artifact_sha256"],
        label="known-good worker evidence",
    )
    checksum_evidence_digest = verify_file_sha256(
        qdrant_checksum_evidence,
        identity["qdrant_checksum_evidence_sha256"],
        label="Qdrant checksum evidence",
    )
    official_archive_digest = qdrant_checksum_from_evidence(
        qdrant_checksum_evidence, qdrant_archive_url_value
    )
    if official_archive_digest != qdrant_archive_sha256:
        raise SupplyChainError(
            "Qdrant archive SHA-256 does not match the captured official checksum entry"
        )
    archive_digest = verify_file_sha256(
        qdrant_archive, qdrant_archive_sha256, label="Qdrant archive"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "platform": TARGET_PLATFORM,
        "base_image_digest": base_digest,
        "qdrant_archive_sha256": archive_digest,
        "qdrant_checksum_evidence_sha256": checksum_evidence_digest,
        "requirements_lock_sha256": lock.sha256,
        "torch_version": identity["torch_version"],
        "cuda_build": identity["cuda_build"],
        "cuda_runtime_version": identity["cuda_runtime_version"],
        "known_good_worker_artifact_sha256": evidence_digest,
    }


def verify_installed_runtime(*, requirements_lock: Path, runtime_identity: Path) -> dict:
    """Verify the image's installed packages and torch/CUDA build against its identity."""
    lock = parse_requirements_lock(requirements_lock)
    identity = load_runtime_identity(runtime_identity)
    # These arguments are identity-bound during validate-build-inputs; repeat the pure
    # validation here so direct use of this subcommand also fails on malformed evidence.
    validate_runtime_identity(
        identity,
        lock,
        base_image=str(identity.get("base_image", "")),
        qdrant_version=str(identity.get("qdrant_version", "")),
        qdrant_archive_url_value=str(identity.get("qdrant_archive_url", "")),
        qdrant_archive_sha256=str(identity.get("qdrant_archive_sha256", "")),
        qdrant_checksum_source_url=str(identity.get("qdrant_checksum_source_url", "")),
    )
    observed_python = platform.python_version()
    if observed_python != identity["python_version"]:
        raise SupplyChainError(
            f"installed Python {observed_python} != identity {identity['python_version']}"
        )

    mismatches: list[str] = []
    for requirement in lock.requirements.values():
        try:
            observed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{requirement.name}=missing")
            continue
        if observed != requirement.version:
            mismatches.append(
                f"{requirement.name}={observed} expected={requirement.version}"
            )
    if mismatches:
        raise SupplyChainError(
            "installed dependency identity mismatch: " + "; ".join(mismatches[:10])
        )

    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - import failure is a release-gate failure
        raise SupplyChainError(f"cannot import locked torch runtime: {type(exc).__name__}: {exc}") from exc
    observed_torch = str(torch.__version__)
    observed_cuda = str(torch.version.cuda)
    if observed_torch != identity["torch_version"]:
        raise SupplyChainError(
            f"installed torch {observed_torch} != identity {identity['torch_version']}"
        )
    if observed_cuda != identity["cuda_runtime_version"]:
        raise SupplyChainError(
            f"installed torch CUDA {observed_cuda} != identity "
            f"{identity['cuda_runtime_version']}"
        )
    return {
        "python_version": observed_python,
        "torch_version": observed_torch,
        "cuda_runtime_version": observed_cuda,
        "requirements_lock_sha256": lock.sha256,
    }


CommandRunner = Callable[[Sequence[str], dict[str, str], Path | None], str]


def _run_command(
    command: Sequence[str], environment: dict[str, str], output_path: Path | None
) -> str:
    temporary: Path | None = None
    output_handle = None
    try:
        if output_path is not None:
            temporary = output_path.with_name(
                f".{output_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
            )
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            output_handle = os.fdopen(descriptor, "wb")
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=output_handle if output_handle is not None else subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise SupplyChainError(f"cannot run {command[0]}: {exc}") from exc
    finally:
        if output_handle is not None:
            output_handle.flush()
            os.fsync(output_handle.fileno())
            output_handle.close()
    if completed.returncode != 0:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        error = completed.stderr.decode("utf-8", errors="replace")[-1000:].strip()
        raise SupplyChainError(
            f"{' '.join(command[:2])} failed with exit {completed.returncode}: {error}"
        )
    if temporary is not None and output_path is not None:
        os.replace(temporary, output_path)
        os.chmod(output_path, 0o600)
        return ""
    return completed.stdout.decode("utf-8", errors="replace").strip()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def _severity_counts(report: object) -> dict[str, int]:
    if not isinstance(report, dict) or not isinstance(report.get("matches"), list):
        raise SupplyChainError("Grype vulnerability report has no matches array")
    counts: dict[str, int] = {}
    for match in report["matches"]:
        vulnerability = match.get("vulnerability") if isinstance(match, dict) else None
        severity = vulnerability.get("severity") if isinstance(vulnerability, dict) else None
        name = str(severity or "Unknown").lower()
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _local_image_identity(inspect_output: str, expected_digest: str) -> dict[str, Any]:
    try:
        records = json.loads(inspect_output)
    except json.JSONDecodeError as exc:
        raise SupplyChainError("docker image inspect returned malformed JSON") from exc
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise SupplyChainError("docker image inspect must return exactly one local image")
    record = records[0]
    repo_digests = record.get("RepoDigests")
    observed = {
        value.rsplit("@sha256:", maxsplit=1)[-1]
        for value in repo_digests or []
        if isinstance(value, str) and "@sha256:" in value
    }
    if expected_digest not in observed:
        raise SupplyChainError(
            "local image metadata does not contain the requested immutable repository digest"
        )
    return {"id": record.get("Id"), "repo_digests": sorted(repo_digests)}


def emit_release_audit(
    *, image_reference: str, output_dir: Path, runner: CommandRunner = _run_command
) -> dict[str, Any]:
    """Emit owner-only SBOM and vulnerability evidence without pulling or updating."""
    image_digest = validate_base_image_reference(image_reference)
    if output_dir.exists():
        raise SupplyChainError(f"release audit destination already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    environment = dict(os.environ)
    environment.update(
        {
            "SYFT_CHECK_FOR_APP_UPDATE": "false",
            "GRYPE_CHECK_FOR_APP_UPDATE": "false",
            "GRYPE_DB_AUTO_UPDATE": "false",
            "GRYPE_DB_VALIDATE_BY_HASH_ON_START": "true",
        }
    )
    inspect_output = runner(
        ["docker", "image", "inspect", image_reference], environment, None
    )
    local_image = _local_image_identity(inspect_output, image_digest)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    os.chmod(temporary, 0o700)
    try:
        sbom_path = temporary / "sbom.cdx.json"
        vulnerabilities_path = temporary / "vulnerabilities.json"
        runner(
            ["syft", f"docker:{image_reference}", "-o", "cyclonedx-json"],
            environment,
            sbom_path,
        )
        runner(
            ["grype", f"sbom:{sbom_path}", "-o", "json"],
            environment,
            vulnerabilities_path,
        )
        try:
            vulnerability_report = json.loads(
                vulnerabilities_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SupplyChainError(f"cannot parse Grype report: {exc}") from exc

        provenance = {
            "schema_version": SCHEMA_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "execution_mode": "offline-local-image",
            "image_reference": image_reference,
            "image_digest": image_digest,
            "local_image": local_image,
            "network_guards": {
                key: environment[key]
                for key in (
                    "SYFT_CHECK_FOR_APP_UPDATE",
                    "GRYPE_CHECK_FOR_APP_UPDATE",
                    "GRYPE_DB_AUTO_UPDATE",
                    "GRYPE_DB_VALIDATE_BY_HASH_ON_START",
                )
            },
            "tools": {
                "syft": runner(["syft", "version"], environment, None),
                "grype": runner(["grype", "version"], environment, None),
                "docker": runner(["docker", "version", "--format", "{{json .}}"], environment, None),
            },
            "artifacts": {
                "sbom.cdx.json": sha256_file(sbom_path),
                "vulnerabilities.json": sha256_file(vulnerabilities_path),
            },
            "vulnerability_counts": _severity_counts(vulnerability_report),
        }
        _atomic_json(temporary / "provenance.json", provenance)
        directory_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.replace(temporary, output_dir)
        parent_descriptor = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return provenance
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("validate-build-inputs")
    build.add_argument("--base-image", required=True)
    build.add_argument("--qdrant-version", required=True)
    build.add_argument("--qdrant-archive-url", required=True)
    build.add_argument("--qdrant-archive-sha256", required=True)
    build.add_argument("--qdrant-checksum-source-url", required=True)
    build.add_argument("--qdrant-checksum-evidence", required=True, type=Path)
    build.add_argument("--qdrant-archive", required=True, type=Path)
    build.add_argument("--requirements-lock", required=True, type=Path)
    build.add_argument("--runtime-identity", required=True, type=Path)
    build.add_argument("--known-good-worker-artifact", required=True, type=Path)

    installed = subparsers.add_parser("verify-installed-runtime")
    installed.add_argument("--requirements-lock", required=True, type=Path)
    installed.add_argument("--runtime-identity", required=True, type=Path)

    audit = subparsers.add_parser("emit-release-audit")
    audit.add_argument("--image", required=True)
    audit.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-build-inputs":
            result = validate_build_inputs(
                base_image=args.base_image,
                qdrant_version=args.qdrant_version,
                qdrant_archive_url_value=args.qdrant_archive_url,
                qdrant_archive_sha256=args.qdrant_archive_sha256,
                qdrant_checksum_source_url=args.qdrant_checksum_source_url,
                qdrant_checksum_evidence=args.qdrant_checksum_evidence,
                qdrant_archive=args.qdrant_archive,
                requirements_lock=args.requirements_lock,
                runtime_identity=args.runtime_identity,
                known_good_worker_artifact=args.known_good_worker_artifact,
            )
        elif args.command == "verify-installed-runtime":
            result = verify_installed_runtime(
                requirements_lock=args.requirements_lock,
                runtime_identity=args.runtime_identity,
            )
        else:
            result = emit_release_audit(
                image_reference=args.image, output_dir=args.output_dir
            )
    except SupplyChainError as exc:
        print(f"supply-chain gate failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
