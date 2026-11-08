"""
data.py
Prepares the BANKING77 dataset for the 12 supported categories:
- Assigns traceable source_ids linked to original train.csv / test.csv rows.
- Detects case/spacing duplicate groups in training data.
- Excludes training duplicate groups with conflicting recorded categories.
- Splits retained training data into ~70% fit, ~15% selection, ~15% calibration
  with random seed 42, keeping duplicate groups together and stratifying by category.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SPLIT_SEED = 42
ALLOWED_DEV_SPLITS = ("fit", "selection", "calibration")
ALL_SPLITS = ("fit", "selection", "calibration", "test")


def load_intents(intents_path: str | Path = "intents.json") -> Dict[str, Any]:
    """Load intents.json and return the full configuration dictionary."""
    path = Path(intents_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing intents file at: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    category_order = data.get("category_order", [])
    intents_list = data.get("intents", [])
    if len(category_order) != 12 or len(intents_list) != 12:
        raise ValueError("intents.json must define exactly 12 categories.")
    if "card_arrival" not in category_order:
        raise ValueError("intents.json must include 'card_arrival' ('Card delivery').")

    label_to_display = {item["label"]: item["display_name"] for item in intents_list}
    label_to_definition = {item["label"]: item["definition"] for item in intents_list}
    data["label_to_display"] = label_to_display
    data["label_to_definition"] = label_to_definition
    return data


def normalize_text(text: str) -> str:
    """
    Normalize text for duplicate detection:
    Identical apart from capital letters and spacing (e.g. 'My card is lost' vs 'MY CARD IS LOST ').
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def make_dup_group_id(norm_text: str) -> str:
    """Create a deterministic short hash ID for a normalized text group."""
    digest = hashlib.sha1(norm_text.encode("utf-8")).hexdigest()[:12]
    return f"grp_{digest}"


def _assign_stratified_group_splits(
    df_retained: pd.DataFrame,
    category_order: List[str],
    seed: int = SPLIT_SEED,
) -> pd.Series:
    """
    Divide retained training rows into ~70% fit, ~15% selection, ~15% calibration
    using seed 42:
    - Keeps every duplicate group in a single subset.
    - Stratifies within each of the 12 categories to hit ~70/15/15 row counts.
    - Guarantees every category is present in every subset.
    """
    rng = np.random.default_rng(seed)
    split_assignment: Dict[str, str] = {}

    for cat in category_order:
        cat_df = df_retained[df_retained["category"] == cat]
        n_total = len(cat_df)
        if n_total < 3:
            raise ValueError(
                f"Category '{cat}' has only {n_total} rows; need at least 3 to populate all splits."
            )

        # Target row counts for this category: ~15% selection, ~15% calibration, ~70% fit
        target_sel = max(1, int(round(0.15 * n_total)))
        target_cal = max(1, int(round(0.15 * n_total)))
        target_fit = n_total - target_sel - target_cal

        # Group by duplicate group ID within this category
        grp_sizes = (
            cat_df.groupby("dup_group_id", sort=True)
            .size()
            .reset_index(name="group_size")
        )
        # Sort deterministically by dup_group_id before shuffling with seed 42
        grp_sizes = grp_sizes.sort_values("dup_group_id").reset_index(drop=True)
        perm = rng.permutation(len(grp_sizes))
        grp_sizes = grp_sizes.iloc[perm].reset_index(drop=True)

        counts = {"fit": 0, "selection": 0, "calibration": 0}
        targets = {"fit": target_fit, "selection": target_sel, "calibration": target_cal}

        for _, row in grp_sizes.iterrows():
            gid = row["dup_group_id"]
            gsize = int(row["group_size"])

            # Ensure every subset gets at least 1 group first, then fill proportionally to remaining deficit
            empty_splits = [s for s in ALLOWED_DEV_SPLITS if counts[s] == 0]
            if empty_splits:
                chosen_split = empty_splits[0]
            else:
                # Choose split with largest remaining capacity (target - current)
                deficits = {s: targets[s] - counts[s] for s in ALLOWED_DEV_SPLITS}
                chosen_split = max(ALLOWED_DEV_SPLITS, key=lambda s: (deficits[s], s == "fit"))

            split_assignment[gid] = chosen_split
            counts[chosen_split] += gsize

    return df_retained["dup_group_id"].map(split_assignment)


def prepare_datasets(
    raw_train_path: str | Path = "banking_data/train.csv",
    raw_test_path: str | Path = "banking_data/test.csv",
    intents_path: str | Path = "intents.json",
    output_dir: str | Path = "data",
    seed: int = SPLIT_SEED,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load raw BANKING77 train.csv and test.csv, filter to 12 categories, handle duplicates,
    create fit/selection/calibration/test splits, and save traceable artifacts to output_dir.
    """
    intents_cfg = load_intents(intents_path)
    category_order: List[str] = intents_cfg["category_order"]
    label_to_display: Dict[str, str] = intents_cfg["label_to_display"]

    train_raw = pd.read_csv(raw_train_path)
    test_raw = pd.read_csv(raw_test_path)

    # 1. Assign stable, traceable source IDs from the raw files before filtering
    train_raw = train_raw.reset_index(drop=True)
    train_raw["raw_row_index"] = train_raw.index
    train_raw["source_id"] = train_raw["raw_row_index"].apply(lambda idx: f"train_{idx:05d}")
    train_raw["source_file"] = "banking_data/train.csv"

    test_raw = test_raw.reset_index(drop=True)
    test_raw["raw_row_index"] = test_raw.index
    test_raw["source_id"] = test_raw["raw_row_index"].apply(lambda idx: f"test_{idx:05d}")
    test_raw["source_file"] = "banking_data/test.csv"

    # 2. Filter to the 12 supported categories
    train_12 = train_raw[train_raw["category"].isin(category_order)].copy()
    test_12 = test_raw[test_raw["category"].isin(category_order)].copy()

    # 3. Normalize text and identify duplicate groups in training data
    train_12["norm_text"] = train_12["text"].apply(normalize_text)
    train_12["dup_group_id"] = train_12["norm_text"].apply(make_dup_group_id)

    test_12["norm_text"] = test_12["text"].apply(normalize_text)
    test_12["dup_group_id"] = test_12["norm_text"].apply(make_dup_group_id)

    # Check for conflicting categories within the same normalized training text group
    grp_cat_counts = train_12.groupby("dup_group_id")["category"].nunique()
    conflicting_groups = set(grp_cat_counts[grp_cat_counts > 1].index)

    excluded_conflict_df = train_12[train_12["dup_group_id"].isin(conflicting_groups)].copy()
    train_retained = train_12[~train_12["dup_group_id"].isin(conflicting_groups)].copy()

    # Calculate duplicate statistics on retained training data
    retained_grp_sizes = train_retained.groupby("dup_group_id").size()
    multi_row_groups = retained_grp_sizes[retained_grp_sizes > 1]
    num_duplicate_groups_retained = int(len(multi_row_groups))
    num_duplicate_rows_total_in_groups = int(multi_row_groups.sum())
    num_extra_duplicate_messages = int((multi_row_groups - 1).sum())

    #Mark whether each row belongs to a multi-message duplicate group
    dup_group_set = set(multi_row_groups.index)
    train_retained["is_duplicate_group"] = train_retained["dup_group_id"].isin(dup_group_set)
    test_12["is_duplicate_group"] = False

    # 4. Divide retained training data into ~70% fit, ~15% selection, ~15% calibration (seed 42)
    train_retained["split"] = _assign_stratified_group_splits(
        train_retained, category_order=category_order, seed=seed
    )
    test_12["split"] = "test"

    # Combine into one master dataframe with original text & category preserved
    combined_df = pd.concat([train_retained, test_12], ignore_index=True)
    combined_df["display_name"] = combined_df["category"].map(label_to_display)
    combined_df["char_length"] = combined_df["text"].astype(str).str.len()
    combined_df["word_count"] = (
        combined_df["text"].astype(str).str.split().apply(len)
    )

    # 5. Integrity verification checks
    for split_name in ALL_SPLITS:
        cats_present = set(combined_df[combined_df["split"] == split_name]["category"].unique())
        missing = set(category_order) - cats_present
        if missing:
            raise AssertionError(f"Split '{split_name}' is missing categories: {missing}")

    # Verify each training duplicate group only appears in 1 development split
    splits_per_group = train_retained.groupby("dup_group_id")["split"].nunique()
    if (splits_per_group > 1).any():
        raise AssertionError("Duplicate group leakage detected across development splits!")

    # 6. Build summary metadata for Data Overview, Agent tools, and the Report
    split_counts_by_cat: Dict[str, Dict[str, int]] = {}
    for split_name in ALL_SPLITS:
        sub = combined_df[combined_df["split"] == split_name]
        vc = sub["category"].value_counts()
        split_counts_by_cat[split_name] = {cat: int(vc.get(cat, 0)) for cat in category_order}

    summary_metadata: Dict[str, Any] = {
        "seed": seed,
        "raw_train_total_rows": int(len(train_raw)),
        "raw_test_total_rows": int(len(test_raw)),
        "selected_12_train_rows_before_exclusion": int(len(train_12)),
        "selected_12_train_rows_retained": int(len(train_retained)),
        "selected_12_test_rows": int(len(test_12)),
        "deduplication_and_exclusions": {
            "conflicting_duplicate_groups_excluded": int(len(conflicting_groups)),
            "conflicting_duplicate_rows_excluded": int(len(excluded_conflict_df)),
            "excluded_source_ids": excluded_conflict_df["source_id"].tolist(),
            "retained_duplicate_groups_count": num_duplicate_groups_retained,
            "retained_messages_in_duplicate_groups": num_duplicate_rows_total_in_groups,
            "retained_extra_duplicate_instances": num_extra_duplicate_messages,
        },
        "split_totals": {
            s: int((combined_df["split"] == s).sum()) for s in ALL_SPLITS
        },
        "split_proportions_dev": {
            s: round(
                float((train_retained["split"] == s).sum()) / len(train_retained), 4
            )
            for s in ALLOWED_DEV_SPLITS
        },
        "category_counts_by_split": split_counts_by_cat,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_path / "processed_splits.csv", index=False)
    with (out_path / "data_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_metadata, f, indent=2)

    return combined_df, summary_metadata


def load_processed_data(
    output_dir: str | Path = "data",
    intents_path: str | Path = "intents.json",
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Load saved processed_splits.csv, data_summary.json, and intents.json."""
    out_path = Path(output_dir)
    splits_csv = out_path / "processed_splits.csv"
    summary_json = out_path / "data_summary.json"

    if not splits_csv.exists() or not summary_json.exists():
        df, summary = prepare_datasets(intents_path=intents_path, output_dir=output_dir)
    else:
        df = pd.read_csv(splits_csv)
        with summary_json.open("r", encoding="utf-8") as f:
            summary = json.load(f)

    intents_cfg = load_intents(intents_path)
    return df, summary, intents_cfg


if __name__ == "__main__":
    df, summary = prepare_datasets()
    print("=== STEP 1: DATA PREPARATION SUMMARY ===")
    print(f"Raw BANKING77 train rows: {summary['raw_train_total_rows']}")
    print(f"Raw BANKING77 test rows : {summary['raw_test_total_rows']}")
    print(f"Train rows in 12 categories (before exclusion): {summary['selected_12_train_rows_before_exclusion']}")
    print(f"Conflicting duplicate groups excluded: {summary['deduplication_and_exclusions']['conflicting_duplicate_groups_excluded']} "
          f"({summary['deduplication_and_exclusions']['conflicting_duplicate_rows_excluded']} rows)")
    print(f"Retained duplicate groups (same category, kept together in split): "
          f"{summary['deduplication_and_exclusions']['retained_duplicate_groups_count']} groups "
          f"({summary['deduplication_and_exclusions']['retained_messages_in_duplicate_groups']} total rows, "
          f"{summary['deduplication_and_exclusions']['retained_extra_duplicate_instances']} extra duplicates)")
    print("\nSplit Totals:", summary["split_totals"])
    print("Development Split Proportions:", summary["split_proportions_dev"])

    # Print per-category table for the report (Page 4 requirement)
    cat_table = pd.DataFrame(summary["category_counts_by_split"])
    cat_table["dev_total"] = cat_table["fit"] + cat_table["selection"] + cat_table["calibration"]
    print("\n=== MESSAGES PER CATEGORY AND SUBSET (Save for Report Section 2) ===")
    print(cat_table.to_string())