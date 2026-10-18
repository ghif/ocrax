"""Colab CLI transport and token helpers."""

from __future__ import annotations

import shutil
import subprocess
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TRANSFER_PART_BYTES = 24 * 1024 * 1024

def split_file(source: Path, directory: Path, prefix: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    with source.open("rb") as stream:
        index = 0
        while chunk := stream.read(TRANSFER_PART_BYTES):
            part = directory / f"{prefix}{index:05d}"
            part.write_bytes(chunk)
            parts.append(part)
            index += 1
    if not parts:
        raise ValueError(f"Cannot transfer empty archive: {source}")
    return parts


def join_parts(parts: list[Path], destination: Path) -> None:
    with destination.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output)


def run_colab(colab: str, session_config: Path, *args: str) -> None:
    # Keep Colab CLI auth explicit; a short-lived ADC token is passed to the VM
    # separately for its GCS API calls.
    command = [colab, "--auth=adc", "--config", str(session_config), *args]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def refresh_access_token(credentials, request) -> tuple[str, str]:
    credentials.refresh(request)
    token = credentials.token
    expiry = credentials.expiry
    if not token or expiry is None:
        raise RuntimeError("Google ADC did not provide a token with an expiry.")
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return token, expiry.astimezone(timezone.utc).isoformat()
