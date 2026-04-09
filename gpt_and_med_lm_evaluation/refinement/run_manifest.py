"""Run manifest utilities for experiment provenance and comparison."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union


def _repo_root_from(script_path: Union[str, Path]) -> Path:
    script_path = Path(script_path).resolve()
    for candidate in [script_path.parent, *script_path.parents]:
        if (candidate / ".git").exists():
            return candidate
    return script_path.parents[1]


def _safe_git_output(repo_root: Path, args: List[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return completed.stdout.strip()


def get_git_context(repo_root: Union[str, Path]) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    commit = _safe_git_output(repo_root, ["rev-parse", "HEAD"])
    branch = _safe_git_output(repo_root, ["branch", "--show-current"])
    status = _safe_git_output(repo_root, ["status", "--porcelain"])
    return {
        "repo_root": str(repo_root),
        "commit_sha": commit,
        "branch": branch,
        "dirty_worktree": bool(status) if status is not None else None,
    }


def create_run_manifest(
    *,
    script_path: Union[str, Path],
    output_paths: Mapping[str, Union[str, Path, None]],
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    task: str,
    provider: str,
    model: str,
    variant: Optional[str] = None,
    runtime_seconds: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    script_path = Path(script_path).resolve()
    repo_root = _repo_root_from(script_path)
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "script_name": script_path.name,
        "script_path": str(script_path),
        "task": task,
        "provider": provider,
        "model": model,
        "variant": variant,
        "runtime_seconds": runtime_seconds,
        "dataset": dict(dataset),
        "config": dict(config),
        "output_paths": {
            key: (str(value) if value is not None else None)
            for key, value in output_paths.items()
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "git": get_git_context(repo_root),
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def save_run_manifest(
    manifest: Mapping[str, Any],
    output_path: Union[str, Path],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def load_run_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_run_manifests(
    baseline_manifest: Mapping[str, Any],
    refined_manifest: Mapping[str, Any],
    *,
    fields: Iterable[Tuple[str, str]] = (
        ("git.commit_sha", "git commit"),
        ("provider", "provider"),
        ("model", "model"),
        ("dataset.input_path", "dataset input_path"),
        ("dataset.dataset_preset", "dataset preset"),
        ("config.max_iterations", "max_iterations"),
        ("config.clinical_threshold", "clinical_threshold"),
        ("config.similarity_threshold", "similarity_threshold"),
        ("config.disclosure_fraction", "disclosure_fraction"),
        ("config.early_confidence_threshold", "early_confidence_threshold"),
        ("config.revision_instability_threshold", "revision_instability_threshold"),
        ("config.curiosity_threshold", "curiosity_threshold"),
        ("config.humility_threshold", "humility_threshold"),
        ("config.random_seed", "random seed"),
        ("config.sampling_mode", "sampling mode"),
        ("config.n_batches", "n_batches"),
        ("config.batch_size", "batch_size"),
    ),
) -> Dict[str, Any]:
    mismatches: List[Dict[str, Any]] = []
    matches: List[str] = []

    def resolve(data: Mapping[str, Any], dotted_key: str) -> Any:
        current: Any = data
        for key in dotted_key.split("."):
            if not isinstance(current, Mapping) or key not in current:
                return None
            current = current[key]
        return current

    for dotted_key, label in fields:
        baseline_value = resolve(baseline_manifest, dotted_key)
        refined_value = resolve(refined_manifest, dotted_key)
        if baseline_value == refined_value:
            matches.append(label)
            continue
        mismatches.append(
            {
                "field": dotted_key,
                "label": label,
                "baseline": baseline_value,
                "refined": refined_value,
            }
        )

    return {
        "is_comparable": len(mismatches) == 0,
        "matches": matches,
        "mismatches": mismatches,
    }
