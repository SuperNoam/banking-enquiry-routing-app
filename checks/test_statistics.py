import math

import joblib
import numpy as np
import pandas as pd
import pytest

from clustering import entropy_mi
from common import ROOT, digest, read_json
from models import (Router, choose_review_rule, losses, probabilities, reliability_bins,
                    review_outcome, score_probabilities)


def test_saved_splits_match_starter_and_keep_groups():
    dev = pd.read_csv(ROOT / "data/development.csv")
    starter = pd.read_csv(ROOT / "data/processed_splits.csv")
    a = dev.set_index("source_id").split
    b = starter[starter.split.ne("test")].set_index("source_id").split
    assert a.sort_index().equals(b.sort_index())
    assert dev.groupby("dup_group_id").split.nunique().max() == 1
    assert pd.crosstab(dev.category, dev.split).shape == (12, 3)
    assert (pd.crosstab(dev.category, dev.split) > 0).all().all()
    assert dev.source_id.is_unique
    assert set(dev.source_id).isdisjoint(pd.read_csv(ROOT / "data/official_test.csv").source_id)


def test_all_saved_probabilities_and_order():
    cfg = read_json(ROOT / "config.json")
    df = pd.read_csv(ROOT / "results/predictions.csv")
    cols = [f"p__{label}" for label in cfg["category_order"]]
    p = df[cols].to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
    predicted = np.asarray(cfg["category_order"])[p.argmax(axis=1)]
    assert (predicted == df.predicted_intent.to_numpy()).all()
    original = df[df.model_id.eq(cfg["model_ids"][cfg["chosen_method"]])].set_index("source_id")
    route = df[df.model_id.eq(cfg["route_model_id"])].set_index("source_id")
    assert (original.predicted_intent == route.predicted_intent).all()


def test_hand_calculated_scores_and_softmax():
    p = np.array([[.8, .2], [.4, .6], [.7, .3]])
    y = np.array([0, 1, 1])
    s = score_probabilities(p, y)
    assert s["accuracy"] == 2 / 3
    assert s["mean_log_loss"] == pytest.approx(-(math.log(.8) + math.log(.6) + math.log(.3)) / 3)
    assert s["confusion_matrix"] == [[1, 0], [1, 1]]
    assert s["macro_f1"] == pytest.approx(2 / 3)
    z = np.array([[4] + [0] * 11, [10000] + [-10000] * 11])
    assert probabilities(z)[0, 0] == pytest.approx(.8323, abs=.0001)
    assert probabilities(z, 2)[0, 0] == pytest.approx(.4018, abs=.0002)
    assert np.isfinite(losses(z, [0, 1])).all()
    np.testing.assert_array_equal(probabilities(z).argmax(1), probabilities(z, 2).argmax(1))
    with pytest.raises(ValueError):
        probabilities(z, 0)


def test_review_selection_and_empty_acceptance():
    p = np.array([[.8, .2]] * 20 + [[.2, .8]] * 2)
    y = np.array([0] * 20 + [0] * 2)
    threshold, trials = choose_review_rule(p, y)
    assert threshold is None  # 2/22 exceeds 5%, higher thresholds accept none.
    assert len(trials) == 5
    assert review_outcome(p, y, None)["error"] is None
    p[20:] = [.4, .6]
    threshold, _ = choose_review_rule(p, y)
    assert threshold == .7


def test_bins_boundaries_and_entropy():
    p = np.array([[1, 0], [.5, .5], [.9, .1]])
    bins = reliability_bins(p, np.array([0, 1, 0]))
    assert len(bins) == 10 and bins[9]["count"] == 2 and bins[5]["count"] == 1
    assert bins[0]["mean_confidence"] is None
    h, mi = entropy_mi(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), 2, 2)
    assert h == pytest.approx(math.log(2)) and mi == pytest.approx(h)


def test_frozen_settings_and_fit_only_vocabulary():
    freeze = read_json(ROOT / "artifacts/freeze_manifest.json")
    assert freeze["config_sha256"] == digest(ROOT / "config.json")
    assert freeze["models_sha256"] == digest(ROOT / "artifacts/classifiers.joblib")
    bundle = joblib.load(ROOT / "artifacts/classifiers.joblib")
    fit = pd.read_csv(ROOT / "data/development/fit.csv")
    analyzer = bundle["tfidf_vectorizer"].build_analyzer()
    fit_tokens = {token for text in fit.text for token in analyzer(text)}
    assert set(bundle["tfidf_vectorizer"].vocabulary_).issubset(fit_tokens)
    assert set(freeze["fit_source_ids"]) == set(fit.source_id)


def test_saved_metrics_recalculate():
    cfg = read_json(ROOT / "config.json")
    pred = pd.read_csv(ROOT / "results/predictions.csv")
    metrics = pd.read_csv(ROOT / "results/metrics.csv")
    for row in metrics.itertuples():
        sub = pred[pred.model_id.eq(row.model_id) & pred.split.eq(row.split)]
        y = sub.true_intent.map({v: i for i, v in enumerate(cfg["category_order"])}).to_numpy()
        result = score_probabilities(sub[[f"p__{v}" for v in cfg["category_order"]]].to_numpy(), y)
        assert row.n == result["n"]
        assert row.accuracy == pytest.approx(result["accuracy"])
        assert row.macro_f1 == pytest.approx(result["macro_f1"])
        assert row.mean_log_loss == pytest.approx(result["mean_log_loss"], abs=1e-12)


def test_clustering_runs_and_representative_membership():
    clusters = pd.read_csv(ROOT / "results/clusters.csv")
    fit = pd.read_csv(ROOT / "data/development/fit.csv")
    assert clusters.run_id.nunique() == 18
    assert clusters.groupby("run_id").size().eq(len(fit)).all()
    assert (clusters.cluster_id >= 0).all() and (clusters.cluster_id < clusters.k).all()
    quality = pd.read_csv(ROOT / "results/clustering_results.csv")
    assert len(quality) == 6 and quality.shuffled_high.lt(quality.mutual_information).all()


def test_router_handles_empty_input():
    router = Router()
    for text in ["", " \n ", None]:
        with pytest.raises(ValueError):
            router.predict(text)
