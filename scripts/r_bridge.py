"""Small, explicit CSV bridge for authoritative R metric estimators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pandas as pd


def advanced_r_library(default: str | Path, override=None) -> Path:
    """Resolve the isolated advanced-metric R library."""
    return Path(
        override or os.environ.get('ADVANCED_R_LIB', default)
    ).resolve()


def run_r_csv_exchange(
    script: str | Path,
    *,
    inputs: Mapping[str, pd.DataFrame],
    outputs: Mapping[str, dict | None],
    argument_order: Sequence[str],
    prefix: str,
    error_label: str,
    library: str | Path,
) -> dict[str, pd.DataFrame]:
    """Write named CSVs, run one R script, and read its named outputs."""
    rscript = shutil.which('Rscript')
    if rscript is None:
        raise RuntimeError(f'Rscript is required for {error_label}')
    unknown = set(argument_order) - (set(inputs) | set(outputs))
    if unknown:
        raise ValueError(f'R CSV exchange has unknown arguments: {sorted(unknown)}')
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        temporary = Path(temporary)
        paths = {
            name: temporary / f'{name}.csv'
            for name in (*inputs, *outputs)
        }
        for name, frame in inputs.items():
            frame.to_csv(paths[name], index=False)
        environment = os.environ.copy()
        environment['ADVANCED_R_LIB'] = str(Path(library).resolve())
        completed = subprocess.run(
            [
                rscript, '--vanilla', str(Path(script).resolve()),
                *(str(paths[name]) for name in argument_order),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f'{error_label} failed: {detail}')
        return {
            name: pd.read_csv(paths[name], dtype=dtype)
            for name, dtype in outputs.items()
        }
