"""Extend the starter data module without changing its IDs or split algorithm.

Training reads train.csv only. Official test is opened after the freeze manifest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import ROOT, digest, read_json, save_csv, write_json
from data import _assign_stratified_group_splits, load_intents, make_dup_group_id, normalize_text


def source_rows(path, prefix, intents):
    df = pd.read_csv(path)
    df["raw_row_index"] = np.arange(len(df))
    df["source_id"] = [f"{prefix}_{i:05d}" for i in range(len(df))]
    df["source_file"] = f"banking_data/{prefix}.csv"
    df["source_csv_line"] = df.raw_row_index + 2
    df["source_sha256"] = digest(path)
    df["norm_text"] = df.text.map(normalize_text)
    df["dup_group_id"] = df.norm_text.map(make_dup_group_id)
    df = df[df.category.isin(intents["category_order"])].copy()
    df["display_name"] = df.category.map(intents["label_to_display"])
    df["char_length"] = df.text.str.len()
    df["word_count"] = df.text.str.split().str.len()
    return df


def prepare_development(root=ROOT):
    intents = load_intents(root / "intents.json")
    train = source_rows(root / "banking_data/train.csv", "train", intents)
    conflict_ids = train.groupby("dup_group_id").category.nunique().loc[lambda s: s > 1].index
    excluded = train[train.dup_group_id.isin(conflict_ids)].copy()
    retained = train[~train.dup_group_id.isin(conflict_ids)].copy()
    sizes = retained.groupby("dup_group_id").size()
    repeated = sizes[sizes > 1]
    retained["is_duplicate_group"] = retained.dup_group_id.isin(repeated.index)
    retained["split"] = _assign_stratified_group_splits(retained, intents["category_order"], seed=42)
    audit = {
        "seed": 42, "training_file_sha256": digest(root / "banking_data/train.csv"),
        "selected_train_rows": len(train), "retained_train_rows": len(retained),
        "conflicting_groups_excluded": len(conflict_ids), "conflicting_rows_excluded": len(excluded),
        "excluded_source_ids": excluded.source_id.tolist(), "duplicate_groups_retained": len(repeated),
        "messages_in_duplicate_groups": int(repeated.sum()), "extra_duplicate_instances": int((repeated - 1).sum()),
        "counts": pd.crosstab(retained.category, retained.split).reindex(intents["category_order"]).to_dict(),
        "test_access": "Not read in preparation, model selection, calibration, or clustering"}
    write_json(root / "data/development_summary.json", audit)
    save_csv(root / "data/development.csv", retained)
    save_csv(root / "data/excluded.csv", excluded)
    for split in ("fit", "selection", "calibration"):
        save_csv(root / f"data/development/{split}.csv", retained[retained.split.eq(split)])
    return retained, audit


def prepare_test(root=ROOT):
    if not (root / "artifacts/freeze_manifest.json").exists():
        raise RuntimeError("Freeze all fitted settings before opening official test")
    intents = load_intents(root / "intents.json")
    test = source_rows(root / "banking_data/test.csv", "test", intents)
    test["split"] = "test"
    test["is_duplicate_group"] = False
    save_csv(root / "data/official_test.csv", test)
    dev = pd.read_csv(root / "data/development.csv")
    write_json(root / "results/test_overlap_audit.json", {
        "normalized_text_overlap_count": int(test.norm_text.isin(dev.norm_text).sum()),
        "source_ids": test.loc[test.norm_text.isin(dev.norm_text), "source_id"].tolist(),
        "action": "Official rows retained, no model/settings changes after inspection"})
    return test
