"""Helpers for building evaluation batches without accidental case duplication."""

from __future__ import annotations

import logging
from typing import List

import pandas as pd


logger = logging.getLogger(__name__)


def build_eval_batches(
    ds: pd.DataFrame,
    n_batches: int,
    batch_size: int,
    random_seed: int,
    sampling_mode: str = "unique",
) -> List[pd.DataFrame]:
    """Build evaluation batches.

    Modes:
    - ``unique``: shuffle once and evaluate each case at most once.
    - ``bootstrap``: sample each batch independently, allowing repeats.
    """
    if ds.empty:
        return []
    if n_batches <= 0:
        raise ValueError("n_batches must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    effective_batch = min(batch_size, len(ds))
    if effective_batch < batch_size:
        logger.warning(
            "Requested batch size (%s) exceeds dataset size (%s); using %s rows per batch",
            batch_size,
            len(ds),
            effective_batch,
        )

    if sampling_mode == "bootstrap":
        batches = []
        for batch_num in range(n_batches):
            batch = ds.sample(n=effective_batch, random_state=random_seed + batch_num)
            batches.append(batch)
        return batches

    if sampling_mode != "unique":
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

    shuffled = ds.sample(frac=1.0, random_state=random_seed)
    max_rows = min(len(ds), n_batches * effective_batch)
    selected = shuffled.iloc[:max_rows]

    if max_rows < len(ds):
        logger.warning(
            "Unique sampling only covers %s/%s rows because n_batches * batch_size is limited",
            max_rows,
            len(ds),
        )

    batches = []
    for start in range(0, len(selected), effective_batch):
        batches.append(selected.iloc[start : start + effective_batch])
    return batches
