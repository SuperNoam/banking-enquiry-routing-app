"""Separate reproducible training, then frozen evaluation. Never called by the UI."""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from clustering import fit_groups
from common import ROOT, digest, read_json, save_csv, write_json
from models import (choose_review_rule, encode, fit_temperature, losses, probabilities,
                    reliability_bins, review_outcome, score_probabilities)
from preparation import prepare_development, prepare_test

DESCRIPTIONS = {"frequency": "Category-frequency baseline", "tfidf": "TF-IDF classifier",
                "encoder": "Encoder classifier", "route": "Route classifier"}


def evaluate(root, df, features, bundle, cfg):
    labels = cfg["category_order"]
    y = df.category.map({v: i for i, v in enumerate(labels)}).to_numpy()
    split = df.split.iloc[0]
    directory = root / "results" / (f"development/{split}" if split != "test" else "official_test")
    directory.mkdir(parents=True, exist_ok=True)
    outputs, metric_rows, per_rows, confusion_rows, calibration_rows = [], [], [], [], []
    raw_logits = {
        "frequency": np.tile(np.log(bundle["frequencies"]), (len(df), 1)),
        "tfidf": bundle["tfidf"].decision_function(bundle["tfidf_vectorizer"].transform(df.text)),
        "encoder": bundle["encoder"].decision_function(features)}
    raw_logits["route"] = raw_logits[cfg["chosen_method"]]
    for method in ["frequency", "tfidf", "encoder", "route"]:
        model_id = cfg["model_ids"][method]
        run_id = f"{cfg['run_id']}:{model_id}:{split}"
        temperature = cfg["temperature"] if method == "route" else 1.0
        p = probabilities(raw_logits[method], temperature)
        scores = score_probabilities(p, y)
        # Report exact stable log-softmax NLL; clipping applies only to generic helper.
        scores["mean_log_loss"] = float(losses(raw_logits[method], y, temperature).mean())
        bins = reliability_bins(p, y) if method in ["route", cfg["chosen_method"]] else []
        trials = [review_outcome(p, y, t) for t in cfg["thresholds"]] if method == "route" else []
        fixed = review_outcome(p, y, cfg["review_threshold"]) if method == "route" else None
        data = {"model_id": model_id, "description": DESCRIPTIONS[method], "split": split,
                "run_id": run_id, "category_order": labels, **scores,
                "calibration_bins": bins, "review_trials": trials, "saved_rule_outcome": fixed}
        write_json(directory / f"{model_id}.json", data)
        metric_rows.append({k: v for k, v in data.items()
                            if k in ["model_id", "description", "split", "run_id", "n", "accuracy", "macro_f1", "mean_log_loss"]})
        for label, row in zip(labels, scores["per_category"]):
            per_rows.append({"model_id": model_id, "split": split, "run_id": run_id, "category": label, **row})
        for i, true in enumerate(labels):
            for j, predicted in enumerate(labels):
                confusion_rows.append({"model_id": model_id, "split": split, "run_id": run_id,
                                       "true_intent": true, "predicted_intent": predicted,
                                       "count": scores["confusion_matrix"][i][j]})
        for row in bins:
            calibration_rows.append({"kind": "confidence_bin", "probability_version": "adjusted" if method == "route" else "original",
                                     "model_id": model_id, "split": split, "run_id": run_id, **row})
        for row in trials:
            calibration_rows.append({"kind": "threshold_trial", "probability_version": "adjusted", "model_id": model_id,
                                     "split": split, "run_id": run_id, "chosen": row["threshold"] == cfg["review_threshold"], **row})
        if fixed is not None:
            calibration_rows.append({"kind": "saved_rule", "probability_version": "adjusted", "model_id": model_id,
                                     "split": split, "run_id": run_id, **fixed})
        predictions = df[["source_id", "source_file", "source_csv_line", "raw_row_index", "text", "category", "split"]].copy()
        predictions = predictions.rename(columns={"category": "true_intent"})
        predictions["model_id"] = model_id
        predictions["run_id"] = run_id
        predictions["predicted_intent"] = np.asarray(labels)[p.argmax(axis=1)]
        predictions["confidence"] = p.max(axis=1)
        predictions["correct"] = predictions.true_intent.eq(predictions.predicted_intent)
        predictions["log_loss"] = losses(raw_logits[method], y, temperature)
        predictions["status"] = ["automatic route" if method == "route" and cfg["review_threshold"] is not None
                                  and c >= cfg["review_threshold"] else "needs human review" if method == "route" else "not applicable"
                                  for c in p.max(axis=1)]
        for i, label in enumerate(labels):
            predictions[f"p__{label}"] = p[:, i]
        outputs.append(predictions)
    all_predictions = pd.concat(outputs, ignore_index=True)
    save_csv(directory / "predictions.csv", all_predictions)
    return metric_rows, per_rows, confusion_rows, calibration_rows, all_predictions


def train(root=ROOT, config_path=None):
    cfg_path = config_path or root / "config.json"
    cfg = read_json(cfg_path)
    (root / "artifacts").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    labels = read_json(root / "intents.json")["category_order"]
    cfg["category_order"] = labels
    dev, audit = prepare_development(root)
    fit = dev[dev.split.eq("fit")]
    selection = dev[dev.split.eq("selection")]
    cal = dev[dev.split.eq("calibration")]
    index = {v: i for i, v in enumerate(labels)}
    yfit = fit.category.map(index).to_numpy()
    yselection = selection.category.map(index).to_numpy()
    ycal = cal.category.map(index).to_numpy()
    print("Encoding development messages with frozen MiniLM", flush=True)
    embeddings = encode(dev.text, root)
    np.save(root / "artifacts/development_embeddings.npy", embeddings)
    efit = embeddings[dev.split.eq("fit")]
    eselection = embeddings[dev.split.eq("selection")]
    ecal = embeddings[dev.split.eq("calibration")]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, dtype=np.float64)
    xfit = vectorizer.fit_transform(fit.text)
    xselection = vectorizer.transform(selection.text)
    bundle = {"tfidf_vectorizer": vectorizer,
              "frequencies": np.bincount(yfit, minlength=12) / len(yfit)}
    selections = []
    for method, xf, xs in [("tfidf", xfit, xselection), ("encoder", efit, eselection)]:
        candidates = []
        for c in cfg["C_values"]:
            # lbfgs fits multinomial loss for >=3 classes. L2 is its default penalty.
            model = LogisticRegression(C=c, solver="lbfgs", max_iter=1500, random_state=42, tol=1e-7).fit(xf, yfit)
            loss = float(losses(model.decision_function(xs), yselection).mean())
            candidates.append((loss, c, model))
            selections.append({"method": method, "C": c, "selection_mean_log_loss": loss})
            print(f"{method} C={c}: selection loss {loss:.6f}", flush=True)
        best = min(candidates, key=lambda x: (round(x[0], 12), x[1]))
        bundle[method] = best[2]
        cfg.setdefault("selected_C", {})[method] = best[1]
    chosen = min(["tfidf", "encoder"], key=lambda m: (
        round(next(x["selection_mean_log_loss"] for x in selections if x["method"] == m and x["C"] == cfg["selected_C"][m]), 12),
        0 if m == "tfidf" else 1))
    cfg["chosen_method"] = chosen
    cfg["selection_ties"] = "Loss rounded to 12 decimals, smaller C within a method, TF-IDF across methods"
    cfg["model_ids"] = {"frequency": "frequency_fit", "tfidf": f"tfidf_C{cfg['selected_C']['tfidf']:g}",
                        "encoder": f"encoder_C{cfg['selected_C']['encoder']:g}", "route": f"route_{chosen}_temperature"}
    cfg["route_model_id"] = cfg["model_ids"]["route"]
    cfg["run_id"] = "banking77_" + audit["training_file_sha256"][:10] + "_seed42_v1"
    for row in selections:
        row["best_for_method"] = row["C"] == cfg["selected_C"][row["method"]]
        row["chosen_for_route"] = row["best_for_method"] and row["method"] == chosen
    save_csv(root / "results/selection_results.csv", selections)
    logits_cal = bundle[chosen].decision_function(vectorizer.transform(cal.text) if chosen == "tfidf" else ecal)
    cfg["temperature"] = fit_temperature(logits_cal, ycal, cfg["temperature_log_bounds"])
    cfg["review_threshold"], trials = choose_review_rule(probabilities(logits_cal, cfg["temperature"]), ycal)
    cfg["review_rule"] = "review every enquiry" if cfg["review_threshold"] is None else f"automatic if confidence >= {cfg['review_threshold']}"
    print(f"Route: {chosen}; T={cfg['temperature']:.6f}; {cfg['review_rule']}", flush=True)
    joblib.dump(bundle, root / "artifacts/classifiers.joblib", compress=3)
    cfg["clustering"]["default"] = fit_groups(root, efit, fit, yfit, cfg["clustering"])
    write_json(cfg_path, cfg)
    freeze = {"config_sha256": digest(cfg_path), "models_sha256": digest(root / "artifacts/classifiers.joblib"),
              "pca_sha256": digest(root / "artifacts/pca.joblib"), "intents_sha256": digest(root / "intents.json"),
              "training_sha256": audit["training_file_sha256"], "test_read": False,
              "fit_source_ids": fit.source_id.tolist(), "selection_source_ids": selection.source_id.tolist(),
              "calibration_source_ids": cal.source_id.tolist(), "calibration_threshold_trials": trials}
    write_json(root / "artifacts/freeze_manifest.json", freeze)
    metric_rows, per_rows, confusion_rows, calibration_rows, predictions = [], [], [], [], []
    for split, df, features in [(s, dev[dev.split.eq(s)], embeddings[dev.split.eq(s)])
                                for s in ["fit", "selection", "calibration"]]:
        outputs = evaluate(root, df, features, bundle, cfg)
        for accumulator, rows in zip([metric_rows, per_rows, confusion_rows, calibration_rows], outputs[:4]):
            accumulator.extend(rows)
        predictions.append(outputs[4])
    # First read of official test within this pipeline: all decisions already saved.
    test = prepare_test(root)
    test_embeddings = encode(test.text, root)
    outputs = evaluate(root, test, test_embeddings, bundle, cfg)
    for accumulator, rows in zip([metric_rows, per_rows, confusion_rows, calibration_rows], outputs[:4]):
        accumulator.extend(rows)
    predictions.append(outputs[4])
    for name, rows in [("metrics", metric_rows), ("per_category", per_rows), ("confusion", confusion_rows),
                       ("calibration", calibration_rows), ("predictions", pd.concat(predictions, ignore_index=True))]:
        save_csv(root / f"results/{name}.csv", rows)
    pred_test = outputs[4]
    mistakes = pred_test[pred_test.model_id.eq(cfg["route_model_id"]) & ~pred_test.correct].head(10)
    save_csv(root / "results/test_mistakes.csv", mistakes)
    a = pred_test[pred_test.model_id.eq(cfg["model_ids"]["tfidf"])].sort_values("source_id")
    b = pred_test[pred_test.model_id.eq(cfg["model_ids"]["encoder"])].sort_values("source_id")
    assert a.source_id.tolist() == b.source_id.tolist()
    differences = a.log_loss.to_numpy() - b.log_loss.to_numpy()
    rng = np.random.default_rng(42)
    means = np.array([differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(1000)])
    low, high = np.percentile(means, [2.5, 97.5])
    save_csv(root / "results/bootstrap.csv", [{"tfidf_model_id": cfg["model_ids"]["tfidf"],
        "encoder_model_id": cfg["model_ids"]["encoder"], "tfidf_C": cfg["selected_C"]["tfidf"],
        "encoder_C": cfg["selected_C"]["encoder"], "n": len(a), "tfidf_loss": a.log_loss.mean(),
        "encoder_loss": b.log_loss.mean(), "delta": differences.mean(), "ci_low": low, "ci_high": high,
        "resamples": 1000, "seed": 42, "sign_convention": "TF-IDF loss minus encoder loss"}])
    save_csv(root / "results/bootstrap_samples.csv", {"iteration": np.arange(1000), "delta": means})
    assert digest(cfg_path) == freeze["config_sha256"]
    assert digest(root / "artifacts/classifiers.joblib") == freeze["models_sha256"]
    print("Frozen official test evaluation complete", flush=True)
    print(pd.DataFrame(metric_rows).query("split == 'test'").to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        train(ROOT, ROOT / args.config)
