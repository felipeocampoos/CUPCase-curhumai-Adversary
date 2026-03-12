import pandas as pd

from eval_batching import build_eval_batches


def test_unique_sampling_covers_each_row_once():
    ds = pd.DataFrame({"value": list(range(6))})

    batches = build_eval_batches(
        ds=ds,
        n_batches=4,
        batch_size=10,
        random_seed=42,
        sampling_mode="unique",
    )

    seen = []
    for batch in batches:
        seen.extend(batch["value"].tolist())

    assert len(batches) == 1
    assert sorted(seen) == list(range(6))
    assert len(seen) == len(set(seen))


def test_unique_sampling_respects_batch_size_without_overlap():
    ds = pd.DataFrame({"value": list(range(7))})

    batches = build_eval_batches(
        ds=ds,
        n_batches=3,
        batch_size=3,
        random_seed=7,
        sampling_mode="unique",
    )

    seen = []
    for batch in batches:
        assert len(batch) <= 3
        seen.extend(batch["value"].tolist())

    assert len(seen) == 7
    assert len(seen) == len(set(seen))


def test_bootstrap_sampling_can_repeat_rows():
    ds = pd.DataFrame({"value": list(range(3))})

    batches = build_eval_batches(
        ds=ds,
        n_batches=2,
        batch_size=3,
        random_seed=1,
        sampling_mode="bootstrap",
    )

    seen = []
    for batch in batches:
        seen.extend(batch["value"].tolist())

    assert len(seen) == 6
    assert len(set(seen)) <= 3
