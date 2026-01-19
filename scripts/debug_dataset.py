"""Sanity check script for ChurnDataset.from_kkbox."""

import argparse
from pathlib import Path

import torch

from src.data.dataset import ChurnDataset


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify ChurnDataset.from_kkbox produces correct data."
    )
    parser.add_argument(
        "--user-logs",
        type=Path,
        required=True,
        help="Path to user_logs.csv file.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to train_v2.csv file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset sanity checks."""
    args = parse_args()

    print("=" * 60)
    print("Loading TRAIN split with aggregated=False...")
    print("=" * 60)

    ds_seq = ChurnDataset.from_kkbox(
        user_logs_path=str(args.user_logs),
        labels_path=str(args.labels),
        split="train",
        aggregated=False,
    )

    # Check 1: Dataset length > 0
    n_seq = len(ds_seq)
    print(f"Dataset length: {n_seq}")
    assert n_seq > 0, "Dataset length must be > 0"
    print("  [PASS] Dataset length > 0")

    # Check 2: time_series shape == (N, 30, 6)
    ts_shape = ds_seq.time_series_data.shape
    print(f"time_series_data shape: {ts_shape}")
    assert ts_shape == (n_seq, 30, 6), f"Expected ({n_seq}, 30, 6), got {ts_shape}"
    print("  [PASS] time_series shape == (N, 30, 6)")

    # Check 3: labels shape == (N,)
    labels_shape = ds_seq.labels.shape
    print(f"labels shape: {labels_shape}")
    assert labels_shape == (n_seq,), f"Expected ({n_seq},), got {labels_shape}"
    print("  [PASS] labels shape == (N,)")

    # Check 4: At least one user has is_active=0 on some day
    # is_active is feature index 5
    is_active_all = ds_seq.time_series_data[:, :, 5]
    has_inactive_day = (is_active_all == 0).any().item()
    print(f"Has at least one inactive day across all users: {has_inactive_day}")
    assert has_inactive_day, "Expected at least one user with is_active=0 on some day"
    print("  [PASS] At least one user has is_active=0 on some day")

    # Check 5: All labels are 0 or 1
    unique_labels = torch.unique(ds_seq.labels).tolist()
    print(f"Unique labels: {unique_labels}")
    assert set(unique_labels).issubset({0, 1}), f"Labels must be 0 or 1, got {unique_labels}"
    print("  [PASS] All labels are 0 or 1")

    # Check 6: __getitem__ returns correct keys
    sample_seq = ds_seq[0]
    expected_keys_seq = {"time_series", "text", "label"}
    actual_keys_seq = set(sample_seq.keys())
    print(f"__getitem__ keys (aggregated=False): {actual_keys_seq}")
    assert actual_keys_seq == expected_keys_seq, f"Expected {expected_keys_seq}, got {actual_keys_seq}"
    print("  [PASS] __getitem__ returns correct keys for aggregated=False")

    print()
    print("=" * 60)
    print("Loading TRAIN split with aggregated=True...")
    print("=" * 60)

    ds_tab = ChurnDataset.from_kkbox(
        user_logs_path=str(args.user_logs),
        labels_path=str(args.labels),
        split="train",
        aggregated=True,
    )

    # Check 7: Dataset length matches
    n_tab = len(ds_tab)
    print(f"Dataset length: {n_tab}")
    assert n_tab == n_seq, f"Expected {n_seq}, got {n_tab}"
    print("  [PASS] Dataset length matches aggregated=False")

    # Check 8: tabular_features shape == (N, 7)
    tab_shape = ds_tab.tabular_features.shape
    print(f"tabular_features shape: {tab_shape}")
    assert tab_shape == (n_tab, 7), f"Expected ({n_tab}, 7), got {tab_shape}"
    print("  [PASS] tabular_features shape == (N, 7)")

    # Check 9: __getitem__ returns correct keys
    sample_tab = ds_tab[0]
    expected_keys_tab = {"features", "label"}
    actual_keys_tab = set(sample_tab.keys())
    print(f"__getitem__ keys (aggregated=True): {actual_keys_tab}")
    assert actual_keys_tab == expected_keys_tab, f"Expected {expected_keys_tab}, got {actual_keys_tab}"
    print("  [PASS] __getitem__ returns correct keys for aggregated=True")

    # Check 10: features tensor shape == (7,)
    features_shape = sample_tab["features"].shape
    print(f"features shape from __getitem__: {features_shape}")
    assert features_shape == (7,), f"Expected (7,), got {features_shape}"
    print("  [PASS] features shape == (7,)")

    # Check 11: No NaN values in tabular_features
    has_nan = torch.isnan(ds_tab.tabular_features).any().item()
    print(f"tabular_features has NaN: {has_nan}")
    assert not has_nan, "tabular_features contains NaN values"
    print("  [PASS] No NaN values in tabular_features")

    # Check 12: active_days (feature index 5) >= 14 for all users
    active_days = ds_tab.tabular_features[:, 5]
    min_active = active_days.min().item()
    print(f"Minimum active_days: {min_active}")
    assert min_active >= 14, f"Expected min active_days >= 14, got {min_active}"
    print("  [PASS] active_days >= 14 for all users")

    # Check 13: recency (feature index 6) >= 0 for all users
    recency = ds_tab.tabular_features[:, 6]
    min_recency = recency.min().item()
    print(f"Minimum recency: {min_recency}")
    assert min_recency >= 0, f"Expected min recency >= 0, got {min_recency}"
    print("  [PASS] recency >= 0 for all users")

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)

    # Print summary statistics
    print()
    print("Summary Statistics:")
    print(f"  Train set size: {n_seq}")
    print(f"  Churn rate: {ds_seq.labels.float().mean().item():.4f}")
    print(f"  Mean active_days: {active_days.mean().item():.2f}")
    print(f"  Mean recency: {recency.mean().item():.2f}")


if __name__ == "__main__":
    main()
