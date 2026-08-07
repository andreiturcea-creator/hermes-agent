"""Fail-closed protection for terminal commands that can mutate DOCX files."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType


_VALIDATOR: ModuleType | None = None
_QUOTED = re.compile(r"['\"]([^'\"\r\n]+?\.docx)['\"]", re.IGNORECASE)
_ABSOLUTE = re.compile(r"(?<![A-Za-z0-9_])(/[^\r\n'\";|<>]+?\.docx)\b", re.IGNORECASE)
_RELATIVE = re.compile(
    r"(?<![A-Za-z0-9_./-])([.~A-Za-z0-9_][^\s'\";|<>]*?\.docx)\b",
    re.IGNORECASE,
)
MAX_CANDIDATE_PATHS = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validator() -> ModuleType:
    global _VALIDATOR
    if _VALIDATOR is not None:
        return _VALIDATOR
    explicit = os.environ.get("HERMES_DOCX_SAFETY_MODULE")
    if explicit:
        path = Path(explicit).expanduser()
    else:
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "operations" / "docx-safety" / "docx_safety.py"
    spec = importlib.util.spec_from_file_location("hermes_docx_safety_runtime", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"DOCX validator unavailable at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _VALIDATOR = module
    return module


def command_mentions_docx(command: str) -> bool:
    return ".docx" in (command or "").lower()


def candidate_paths(command: str, cwd: str) -> list[Path]:
    if not command_mentions_docx(command):
        return []
    quoted = list(_QUOTED.finditer(command))
    raw = [match.group(1).strip() for match in quoted]
    masked = list(command)
    for match in quoted:
        masked[match.start() : match.end()] = " " * (match.end() - match.start())
    remainder = "".join(masked)
    for pattern in (_ABSOLUTE, _RELATIVE):
        raw.extend(match.group(1).strip() for match in pattern.finditer(remainder))

    base = Path(cwd).expanduser().resolve()
    paths: list[Path] = []
    seen: set[str] = set()
    for value in raw:
        if any(marker in value for marker in ("$", "{", "}", "*", "?", "\n", "\r")):
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        candidate = candidate.resolve(strict=False)
        key = str(candidate)
        if candidate.suffix.lower() != ".docx" or key in seen:
            continue
        if len(paths) >= MAX_CANDIDATE_PATHS:
            raise RuntimeError(
                f"DOCX command exceeds the {MAX_CANDIDATE_PATHS}-path protection limit"
            )
        seen.add(key)
        paths.append(candidate)
    return paths


@dataclass
class _Snapshot:
    path: Path
    existed: bool
    sha256: str | None
    backup: Path | None


@dataclass
class GuardSession:
    temporary: tempfile.TemporaryDirectory
    snapshots: list[_Snapshot]


@dataclass(frozen=True)
class GuardReport:
    ok: bool
    checked: tuple[str, ...]
    restored: tuple[str, ...]
    quarantined: tuple[str, ...]
    errors: tuple[str, ...]


def _quarantine(path: Path, stamp: str) -> Path:
    rejected = path.with_name(f"{path.name}.rejected-invalid-{stamp}.bin")
    counter = 1
    while rejected.exists():
        rejected = path.with_name(
            f"{path.name}.rejected-invalid-{stamp}-{counter}.bin"
        )
        counter += 1
    os.replace(path, rejected)
    return rejected


def _restore(snapshot: _Snapshot) -> None:
    if snapshot.backup is None:
        raise RuntimeError(f"no restoration copy is available for {snapshot.path}")
    restore = snapshot.path.with_name(f".{snapshot.path.name}.restore-{os.getpid()}")
    restore.unlink(missing_ok=True)
    shutil.copy2(snapshot.backup, restore)
    os.replace(restore, snapshot.path)


def prepare_terminal_guard(command: str, cwd: str) -> GuardSession | None:
    paths = candidate_paths(command, cwd)
    if not paths:
        if command_mentions_docx(command):
            raise RuntimeError(
                "DOCX command contains no resolvable literal .docx path; "
                "pass explicit paths so Hermes can protect them"
            )
        return None
    temporary = tempfile.TemporaryDirectory(prefix="hermes-docx-guard-")
    root = Path(temporary.name)
    snapshots: list[_Snapshot] = []
    for index, path in enumerate(paths):
        existed = path.exists()
        if existed and not path.is_file():
            temporary.cleanup()
            raise RuntimeError(f"DOCX candidate is not a regular file: {path}")
        backup = root / f"{index}.docx" if existed else None
        digest = None
        if existed and backup is not None:
            shutil.copy2(path, backup)
            digest = _sha256(path)
        snapshots.append(_Snapshot(path, existed, digest, backup))
    return GuardSession(temporary, snapshots)


def finalize_terminal_guard(session: GuardSession | None) -> GuardReport:
    if session is None:
        return GuardReport(True, (), (), (), ())
    checked: list[str] = []
    restored: list[str] = []
    quarantined: list[str] = []
    errors: list[str] = []
    try:
        validator = _validator()
        stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        for snapshot in session.snapshots:
            path = snapshot.path
            if not path.exists():
                if snapshot.existed:
                    errors.append(f"{path}: existing DOCX was removed by the command")
                    _restore(snapshot)
                    restored.append(str(path))
                continue
            digest = _sha256(path) if path.is_file() else None
            if snapshot.existed and digest == snapshot.sha256:
                continue
            checked.append(str(path))
            report = validator.inspect_docx(path)
            if report.valid:
                continue
            rejected = _quarantine(path, stamp)
            quarantined.append(str(rejected))
            errors.extend(f"{path}: {message}" for message in report.errors)
            if snapshot.existed:
                _restore(snapshot)
                restored.append(str(path))
    except Exception as exc:
        errors.append(f"DOCX guard failed closed: {exc}")
        stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        for snapshot in session.snapshots:
            try:
                path = snapshot.path
                digest = _sha256(path) if path.is_file() else None
                changed = not snapshot.existed or digest != snapshot.sha256
                if path.exists() and changed:
                    quarantined.append(str(_quarantine(path, stamp)))
                if snapshot.existed and changed:
                    _restore(snapshot)
                    if str(path) not in restored:
                        restored.append(str(path))
            except Exception as restore_error:
                errors.append(f"restore/quarantine failed for {snapshot.path}: {restore_error}")
    finally:
        session.temporary.cleanup()
    return GuardReport(
        not errors,
        tuple(checked),
        tuple(restored),
        tuple(quarantined),
        tuple(errors),
    )


def format_guard_error(report: GuardReport) -> str:
    if report.ok:
        return ""
    lines = ["DOCX SAFETY BLOCK: command left an invalid .docx candidate."]
    lines.extend(f"- {error}" for error in report.errors)
    if report.restored:
        lines.append("Restored: " + ", ".join(report.restored))
    if report.quarantined:
        lines.append("Quarantined: " + ", ".join(report.quarantined))
    lines.append("Use the configured aster-docx publisher with a separately built candidate.")
    return "\n".join(lines)
