"""Read-only table transformations shared by the interface and its checks."""
from __future__ import annotations

import pandas as pd


def csv_bytes(frame):
    """Include a UTF-8 BOM so downloaded enquiry text also opens in Windows Excel."""
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def category_counts(rows, order, names):
    counts = rows.category.value_counts().reindex(order, fill_value=0)
    return pd.DataFrame({
        "category": order,
        "display_name": [names[label] for label in order],
        "count": counts.to_numpy(),
    })


def representatives(fit, assignments, method, k, cluster_id):
    selected = assignments[
        assignments.method.eq(method) & assignments.k.eq(k)
        & assignments.seed.eq(0) & assignments.cluster_id.eq(cluster_id)
    ]
    selected = selected.sort_values(
        ["representative_score", "source_id"], ascending=[False, True], kind="stable"
    ).head(5)
    return selected.merge(
        fit[["source_id", "text", "source_file", "source_csv_line"]],
        on="source_id", how="left", validate="one_to_one",
    )


def probability_table(result, names):
    return pd.DataFrame([
        {"category": label, "display_name": names[label], "probability": value}
        for label, value in result["probabilities"].items()
    ])


def route_export(result):
    row = {key: value for key, value in result.items()
           if key not in {"probabilities", "top_three"}}
    row.update({f"p__{label}": value for label, value in result["probabilities"].items()})
    for rank, entry in enumerate(result["top_three"], start=1):
        row[f"top_{rank}_category"] = entry["category"]
        row[f"top_{rank}_probability"] = entry["probability"]
    return pd.DataFrame([row])


def threshold_table(calibration, split, route_id, saved_threshold):
    rows = calibration[
        calibration.kind.eq("threshold_trial") & calibration.split.eq(split)
        & calibration.model_id.eq(route_id)
    ][["threshold", "n", "accepted", "wrong_accepted", "coverage", "error",
       "model_id", "split", "run_id"]].copy()
    rows["saved_rule"] = rows.threshold.eq(saved_threshold)
    if split == "calibration":
        rows["qualifies"] = rows.accepted.ge(20) & rows.error.le(.05)
    return rows.sort_values("threshold").reset_index(drop=True)
