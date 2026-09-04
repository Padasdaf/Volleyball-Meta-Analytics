"""Install the exact sbgcop release used by an independence analysis.

The primary analysis follows Franks et al. (2016) by using sbgcop 0.975.
Version 1.0 is supported only for the explicit version-sensitivity diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
from r_setup_utils import download_verified, locate_r


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_LIBRARY_ROOT = REPOSITORY_ROOT / ".r-lib"
PACKAGE_SOURCES = {
    "0.975": {
        "url": (
            "https://cran.r-project.org/src/contrib/Archive/sbgcop/"
            "sbgcop_0.975.tar.gz"
        ),
        "sha256": "c14016bbb886b491f088181195a06a5c0e3ef11d2c02fa389ca47abdf8ae8738",
    },
    "1.0": {
        "url": "https://cran.r-project.org/src/contrib/sbgcop_1.0.tar.gz",
        "sha256": "918bbb6661845112b8660a3d6864f6e3b638642f8ede81a86a0621f57d1b12d7",
    },
}


def library_for_version(version: str, library_root: Path = DEFAULT_LIBRARY_ROOT) -> Path:
    """Return the isolated R library for an approved exact sbgcop version."""
    if version not in PACKAGE_SOURCES:
        raise ValueError(f"Unsupported sbgcop version: {version}")
    return library_root.resolve() / f"sbgcop-{version}"


def _installed_version(rscript: str, library: Path) -> str | None:
    if not library.is_dir():
        return None
    expression = (
        "lib <- normalizePath(commandArgs(TRUE)[1], mustWork=TRUE); "
        "description <- packageDescription('sbgcop', lib.loc=lib); "
        "if (is.null(description)) quit(status=2); "
        "cat(description[['Version']])"
    )
    completed = subprocess.run(
        [rscript, "--vanilla", "-e", expression, str(library)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def ensure_sbgcop(
    version: str = "0.975", library_root: Path = DEFAULT_LIBRARY_ROOT
) -> Path:
    """Install and verify an exact sbgcop source release in an isolated library."""
    if version not in PACKAGE_SOURCES:
        raise ValueError(f"Unsupported sbgcop version: {version}")

    rscript, r_binary = locate_r()

    library = library_for_version(version, library_root)
    if _installed_version(rscript, library) == version:
        return library

    library.mkdir(parents=True, exist_ok=True)
    source = PACKAGE_SOURCES[version]
    archive_name = f"sbgcop_{version}.tar.gz"
    print(
        f"Installing sbgcop {version} from the official CRAN source archive...",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="sbgcop-install-") as temporary_directory:
        archive = Path(temporary_directory) / archive_name
        download_verified(source['url'], archive, source['sha256'])
        subprocess.run(
            [
                r_binary,
                "CMD",
                "INSTALL",
                f"--library={library}",
                str(archive),
            ],
            check=True,
        )

    installed = _installed_version(rscript, library)
    if installed != version:
        raise RuntimeError(
            f"Installed sbgcop version check failed: expected {version}, got {installed!r}"
        )
    return library


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install a checksum-verified exact sbgcop source release."
    )
    parser.add_argument(
        "--version",
        choices=tuple(PACKAGE_SOURCES),
        default="0.975",
        help="0.975 is primary; 1.0 is available only for sensitivity analysis",
    )
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY_ROOT)
    args = parser.parse_args()
    library = ensure_sbgcop(args.version, args.library_root)
    print(f"Verified sbgcop {args.version} in {library}")


if __name__ == "__main__":
    main()
