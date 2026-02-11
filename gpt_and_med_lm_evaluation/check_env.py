#!/usr/bin/env python3
"""Fail-fast environment check for CUPCase eval runs."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version


REQUIRED_PYTHON = (3, 12)
REQUIRED_VERSIONS = {
    "openai": "2.20.0",
    "python-dotenv": "1.2.1",
    "pandas": "3.0.0",
    "datasets": "4.5.0",
    "bert-score": "0.3.13",
    "transformers": "4.41.2",
    "tokenizers": "0.19.1",
}


def _distribution_version(distribution_name: str) -> str:
    return version(distribution_name)


def main() -> int:
    errors: list[str] = []

    if sys.version_info[:2] != REQUIRED_PYTHON:
        errors.append(
            f"Python must be {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}.x; "
            f"found {sys.version.split()[0]}"
        )

    for distribution_name, expected in REQUIRED_VERSIONS.items():
        try:
            found = _distribution_version(distribution_name)
        except PackageNotFoundError:
            errors.append(f"Missing package '{distribution_name}'")
            continue
        if found != expected:
            errors.append(
                f"{distribution_name}=={expected} required, found {distribution_name}=={found}"
            )

    if errors:
        print("Environment check failed:")
        for item in errors:
            print(f"- {item}")
        return 1

    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
