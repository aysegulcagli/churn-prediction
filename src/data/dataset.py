"""Dataset classes for multi-modal churn prediction."""

import os
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class ChurnDataset(Dataset):
    """Multi-modal dataset for churn prediction.

    Handles time series behavioral data, text data, and churn labels.

    Attributes:
        time_series_data: Tensor of shape (num_samples, sequence_length, num_features).
        text_data: List of text strings or tokenized text tensors.
        labels: Tensor of binary churn labels (0 = retained, 1 = churned).
        aggregated: If True, return aggregated features instead of raw time series.
    """

    def __init__(
        self,
        time_series_data: torch.Tensor,
        text_data: list[str],
        labels: torch.Tensor,
        aggregated: bool = False,
        tabular_features: torch.Tensor | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            time_series_data: Time series features for each user.
            text_data: Text data (support tickets, feedback) for each user.
            labels: Binary churn labels.
            aggregated: If True, return aggregated tabular features.
            tabular_features: Precomputed tabular features for baseline model.

        Raises:
            ValueError: If data lengths do not match.
        """
        num_samples = time_series_data.shape[0]

        if len(text_data) != num_samples:
            raise ValueError(
                f"Time series has {num_samples} samples but text has {len(text_data)}"
            )

        if labels.shape[0] != num_samples:
            raise ValueError(
                f"Time series has {num_samples} samples but labels has {labels.shape[0]}"
            )

        self.time_series_data = time_series_data
        self.text_data = text_data
        self.labels = labels
        self.num_samples = num_samples
        self.aggregated = aggregated
        self.tabular_features = tabular_features

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing either:
                - time_series, text, label (when aggregated=False)
                - features, label (when aggregated=True)
        """
        if self.aggregated and self.tabular_features is not None:
            return {
                "features": self.tabular_features[idx],
                "label": self.labels[idx],
            }

        if self.aggregated:
            time_series = self.time_series_data[idx]

            mean = time_series.mean(dim=0)
            std = time_series.std(dim=0)
            min_val = time_series.min(dim=0).values
            max_val = time_series.max(dim=0).values
            last_val = time_series[-1]

            features = torch.cat([mean, std, min_val, max_val, last_val])

            return {
                "features": features,
                "label": self.labels[idx],
            }

        return {
            "time_series": self.time_series_data[idx],
            "text": self.text_data[idx],
            "label": self.labels[idx],
        }

    @classmethod
    def from_kkbox(
        cls,
        user_logs_path: str,
        labels_path: str,
        split: str,
        aggregated: bool = False,
        chunksize: int = 5_000_000,
        cache_dir: str = "data/cache",
    ) -> "ChurnDataset":
        """Load KKBOX dataset and create ChurnDataset instance.

        Args:
            user_logs_path: Path to user_logs.csv file.
            labels_path: Path to train_v2.csv file.
            split: One of "train", "val", "test".
            aggregated: If True, return aggregated tabular features.
            chunksize: Number of rows to read per chunk from user_logs.csv.
            cache_dir: Directory for caching processed tensors.

        Returns:
            ChurnDataset instance for the requested split.

        Raises:
            ValueError: If split is not one of train/val/test.
        """
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got {split}")

        # Check cache first
        cache_file = os.path.join(cache_dir, f"kkbox_{split}.pt")
        if os.path.exists(cache_file):
            print(f"[CACHE HIT] Loading cached dataset: {cache_file}", flush=True)
            cached = torch.load(cache_file)
            time_series_tensor = cached["time_series"]
            labels_tensor = cached["labels"]
            tabular_features_tensor = cached.get("tabular")
            num_users = time_series_tensor.shape[0]
            text_data = [""] * num_users
            print(f"[CACHE HIT] Loaded {num_users} users from cache", flush=True)
            return cls(
                time_series_tensor,
                text_data,
                labels_tensor,
                aggregated=aggregated,
                tabular_features=tabular_features_tensor if aggregated else None,
            )

        # Estimate total chunks for progress bar
        print(f"[CACHE MISS] Processing {user_logs_path} from scratch...", flush=True)
        print("Counting total rows (this may take a moment)...", flush=True)
        with open(user_logs_path, "r", encoding="utf-8") as f:
            total_rows = sum(1 for _ in f) - 1  # subtract header
        total_chunks = ceil(total_rows / chunksize)
        print(f"Total rows: {total_rows:,}, chunks: {total_chunks}", flush=True)

        # Load labels (small file, fits in memory)
        labels_df = pd.read_csv(labels_path)
        valid_users = set(labels_df["msno"].unique())

        # Date boundaries
        cutoff_date = pd.Timestamp("2017-01-31")
        train_cutoff = pd.Timestamp("2016-12-31")
        val_start = pd.Timestamp("2017-01-01")
        val_end = pd.Timestamp("2017-01-15")
        test_start = pd.Timestamp("2017-01-16")
        test_end = pd.Timestamp("2017-01-31")

        numeric_cols = [
            "num_25", "num_50", "num_75", "num_985",
            "num_100", "num_unq", "total_secs"
        ]

        # =====================================================================
        # PASS 1: Determine window_end per user (chunked)
        # =====================================================================
        user_max_dates: dict[str, pd.Timestamp] = {}

        pbar1 = tqdm(total=total_chunks, desc="PASS 1: computing window_end")
        for chunk in pd.read_csv(user_logs_path, chunksize=chunksize):
            pbar1.update(1)

            # Keep only valid users
            chunk = chunk[chunk["msno"].isin(valid_users)]
            if chunk.empty:
                continue

            # Convert date
            chunk["date"] = pd.to_datetime(chunk["date"], format="%Y%m%d")

            # Filter to dates <= cutoff
            chunk = chunk[chunk["date"] <= cutoff_date]
            if chunk.empty:
                continue

            # Update max dates per user
            chunk_max = chunk.groupby("msno")["date"].max()
            for msno, max_date in chunk_max.items():
                if msno not in user_max_dates or max_date > user_max_dates[msno]:
                    user_max_dates[msno] = max_date
        pbar1.close()

        # Build window_end DataFrame
        window_end_df = pd.DataFrame([
            {"msno": msno, "window_end": window_end}
            for msno, window_end in user_max_dates.items()
        ])

        # Assign splits
        def assign_split(window_end: pd.Timestamp) -> str:
            if window_end <= train_cutoff:
                return "train"
            elif val_start <= window_end <= val_end:
                return "val"
            elif test_start <= window_end <= test_end:
                return "test"
            return "unknown"

        window_end_df["split_label"] = window_end_df["window_end"].apply(assign_split)

        # Keep only users in requested split
        split_users_df = window_end_df[window_end_df["split_label"] == split].copy()
        split_users_df["window_start"] = split_users_df["window_end"] - pd.Timedelta(days=29)

        # Build lookup dictionaries for fast filtering
        split_user_set = set(split_users_df["msno"])
        user_window_start = dict(zip(split_users_df["msno"], split_users_df["window_start"]))
        user_window_end = dict(zip(split_users_df["msno"], split_users_df["window_end"]))

        # =====================================================================
        # PASS 2: Read logs again, filter to split users and windows (chunked)
        # =====================================================================
        print(f"Split '{split}' has {len(split_user_set)} users", flush=True)
        accumulated_chunks = []

        pbar2 = tqdm(total=total_chunks, desc="PASS 2: collecting windowed logs")
        for chunk in pd.read_csv(user_logs_path, chunksize=chunksize):
            pbar2.update(1)

            # Keep only split users
            chunk = chunk[chunk["msno"].isin(split_user_set)]
            if chunk.empty:
                continue

            # Convert date
            chunk["date"] = pd.to_datetime(chunk["date"], format="%Y%m%d")

            # Filter each row to its user's window
            chunk["window_start"] = chunk["msno"].map(user_window_start)
            chunk["window_end"] = chunk["msno"].map(user_window_end)
            chunk = chunk[
                (chunk["date"] >= chunk["window_start"]) &
                (chunk["date"] <= chunk["window_end"])
            ]
            if chunk.empty:
                continue

            # Keep only needed columns
            chunk = chunk[["msno", "date"] + numeric_cols]
            accumulated_chunks.append(chunk)
        pbar2.close()

        # Combine all chunks
        if accumulated_chunks:
            windowed_df = pd.concat(accumulated_chunks, ignore_index=True)
        else:
            windowed_df = pd.DataFrame(columns=["msno", "date"] + numeric_cols)

        # Deduplicate by grouping on (msno, date) and summing
        windowed_df = windowed_df.groupby(["msno", "date"], as_index=False)[numeric_cols].sum()

        # =====================================================================
        # POST-PROCESSING (unchanged logic)
        # =====================================================================

        # Create complete 30-day grid per user
        grid_rows = []
        for _, row in split_users_df.iterrows():
            dates = pd.date_range(row["window_start"], row["window_end"], freq="D")
            for d in dates:
                grid_rows.append({"msno": row["msno"], "date": d})
        grid_df = pd.DataFrame(grid_rows)

        # Left-join logs onto grid and zero-fill missing days
        filled_df = grid_df.merge(
            windowed_df[["msno", "date"] + numeric_cols],
            on=["msno", "date"],
            how="left"
        )
        filled_df[numeric_cols] = filled_df[numeric_cols].fillna(0)

        # Remove users with < 14 active days
        active_days = filled_df[filled_df["total_secs"] > 0].groupby("msno").size()
        active_days = active_days.reset_index(name="active_count")
        valid_users_14 = set(active_days[active_days["active_count"] >= 14]["msno"])
        filled_df = filled_df[filled_df["msno"].isin(valid_users_14)]

        # Compute per-day features (6 total)
        filled_df["log_total_secs"] = np.log1p(
            np.clip(filled_df["total_secs"].values, a_min=0.0, a_max=None)
        )
        filled_df["completion_denom"] = (
            filled_df["num_25"] + filled_df["num_50"] +
            filled_df["num_75"] + filled_df["num_985"] + filled_df["num_100"]
        )
        filled_df["completion_rate"] = np.where(
            filled_df["completion_denom"] > 0,
            filled_df["num_100"] / filled_df["completion_denom"],
            0.0
        )
        filled_df["is_active"] = (filled_df["total_secs"] > 0).astype(float)

        # Log-transform count features for numerical stability
        filled_df["log_num_unq"] = np.log1p(
            np.clip(filled_df["num_unq"].values, a_min=0.0, a_max=None)
        )
        filled_df["log_num_100"] = np.log1p(
            np.clip(filled_df["num_100"].values, a_min=0.0, a_max=None)
        )
        filled_df["log_num_25"] = np.log1p(
            np.clip(filled_df["num_25"].values, a_min=0.0, a_max=None)
        )

        # Pivot to sequence tensor
        feature_cols = [
            "log_total_secs", "log_num_unq", "log_num_100",
            "log_num_25", "completion_rate", "is_active"
        ]
        filled_df = filled_df.sort_values(["msno", "date"])
        user_ids = filled_df["msno"].unique()
        num_users = len(user_ids)

        time_series_list = []
        for uid in user_ids:
            user_data = filled_df[filled_df["msno"] == uid][feature_cols].values
            time_series_list.append(user_data)

        time_series_array = np.stack(time_series_list, axis=0)
        # Replace NaN/Inf with 0 for numerical stability
        time_series_array = np.nan_to_num(time_series_array, nan=0.0, posinf=0.0, neginf=0.0)
        time_series_tensor = torch.tensor(time_series_array, dtype=torch.float32)

        # Extract labels in same order
        user_id_to_label = dict(zip(labels_df["msno"], labels_df["is_churn"]))
        labels_list = [user_id_to_label[uid] for uid in user_ids]
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        # Create placeholder text data
        text_data = [""] * num_users

        # Compute tabular features if aggregated=True
        tabular_features_tensor = None
        if aggregated:
            tabular_list = []
            for uid in user_ids:
                user_df = filled_df[filled_df["msno"] == uid]

                # 1. total_secs_mean
                total_secs_mean = user_df["total_secs"].mean()

                # 2. total_secs_std
                total_secs_std = user_df["total_secs"].std()
                if pd.isna(total_secs_std):
                    total_secs_std = 0.0

                # 3. total_secs_last (last day of window)
                total_secs_last = user_df.iloc[-1]["total_secs"]

                # 4. num_unq_mean
                num_unq_mean = user_df["num_unq"].mean()

                # 5. completion_rate_mean
                completion_rate_mean = user_df["completion_rate"].mean()

                # 6. active_days
                active_days_count = (user_df["total_secs"] > 0).sum()

                # 7. recency (days between last active day and final day)
                active_rows = user_df[user_df["total_secs"] > 0]
                if len(active_rows) > 0:
                    last_active_date = active_rows["date"].max()
                    final_date = user_df["date"].max()
                    recency = (final_date - last_active_date).days
                else:
                    recency = 30

                tabular_list.append([
                    total_secs_mean,
                    total_secs_std,
                    total_secs_last,
                    num_unq_mean,
                    completion_rate_mean,
                    float(active_days_count),
                    float(recency),
                ])

            tabular_array = np.array(tabular_list, dtype=np.float32)
            tabular_features_tensor = torch.tensor(tabular_array, dtype=torch.float32)

        # Save to cache
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(
            {
                "time_series": time_series_tensor,
                "tabular": tabular_features_tensor,
                "labels": labels_tensor,
            },
            cache_file,
        )
        print(f"[CACHE WRITE] Saved processed dataset to {cache_file}", flush=True)

        return cls(
            time_series_tensor,
            text_data,
            labels_tensor,
            aggregated=aggregated,
            tabular_features=tabular_features_tensor if aggregated else None,
        )
