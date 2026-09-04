"""Shared mechanics for isolated, checksum-verified R environments."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen


def locate_r() -> tuple[str, str]:
    rscript = shutil.which('Rscript')
    r_binary = shutil.which('R')
    if rscript is None or r_binary is None:
        raise RuntimeError('R and Rscript are required but were not found on PATH')
    return rscript, r_binary


def run_r_expression(rscript: str, library: Path, expression: str) -> None:
    subprocess.run(
        [rscript, '--vanilla', '-e', expression, str(library)], check=True
    )


def download_verified(url: str, destination: Path, sha256: str) -> None:
    with urlopen(url) as response, destination.open('wb') as output:
        shutil.copyfileobj(response, output)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != sha256:
        raise RuntimeError(
            f'SHA-256 mismatch for {destination.name}: '
            f'expected {sha256}, received {digest}'
        )
