"""Exercise the Streamlit entry point, saved-result selectors and CSV exports."""
from io import BytesIO
import json

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from streamlit.testing.v1 import AppTest

from app_data import csv_bytes, representatives, route_export, threshold_table
from common import ROOT, digest, read_json
from models import Router


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("The interface must load saved models without fitting")

    for model in [LogisticRegression, PCA, KMeans, GaussianMixture]:
        monkeypatch.setattr(model, "fit", fail)


@pytest.fixture
def downloads(monkeypatch):
    frames = {}
    original = st.download_button

    def capture(label, data, *args, **kwargs):
        frames[kwargs["key"]] = pd.read_csv(BytesIO(data))
        return original(label, data, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", capture)
    return frames


@pytest.fixture
def app():
    return AppTest.from_file(ROOT / "app.py", default_timeout=30).run()


@pytest.mark.parametrize("screen", [
    "Data overview", "Route an enquiry", "Explore request groups", "Compare models",
    "Ask the analysis agent",
])
def test_all_screens_open_without_fitting(app, screen):
    app.sidebar.radio(key="screen").set_value(screen).run()
    assert not app.exception
    assert not app.error
    assert app.title[0].value == screen


@pytest.mark.parametrize("split", ["fit", "selection", "calibration"])
def test_data_filter_and_download_agree(app, downloads, split):
    category = "card_arrival"
    app.selectbox(key="data_split").set_value(split)
    app.selectbox(key="data_category").set_value(category).run()
    assert not app.exception
    source = pd.read_csv(ROOT / f"data/development/{split}.csv")
    expected = source[source.category.eq(category)].reset_index(drop=True)
    pd.testing.assert_frame_equal(downloads["data_download"], expected)
    shown = next(element.value for element in app.dataframe if "source_id" in element.value)
    assert shown.source_id.tolist() == expected.source_id.tolist()
    assert shown.text.tolist() == expected.text.tolist()
    assert app.metric[0].value == f"{len(expected):,}"


@pytest.mark.parametrize("method", ["frequency", "tfidf", "encoder", "route"])
@pytest.mark.parametrize("split", ["fit", "selection", "calibration", "test"])
def test_model_subset_scores_and_exports(app, downloads, method, split):
    cfg = read_json(ROOT / "config.json")
    app.sidebar.radio(key="screen").set_value("Compare models").run()
    app.selectbox(key="compare_model").set_value(method)
    app.selectbox(key="compare_split").set_value(split).run()
    assert not app.exception
    assert not app.error
    metrics = pd.read_csv(ROOT / "results/metrics.csv")
    expected = metrics[metrics.model_id.eq(cfg["model_ids"][method]) & metrics.split.eq(split)]
    pd.testing.assert_frame_equal(downloads["scores_download"], expected.reset_index(drop=True))
    row = expected.iloc[0]
    displayed = {metric.label: metric.value for metric in app.metric}
    assert displayed["Accuracy"] == f"{row.accuracy:.2%}"
    assert displayed["Mean log-loss"] == f"{row.mean_log_loss:.6f}"
    predictions = downloads["predictions_download"]
    assert len(predictions) == row.n
    assert predictions.model_id.eq(row.model_id).all()
    assert predictions.split.eq(split).all()
    assert downloads["confusion_download"]["count"].sum() == row.n
    mistakes = downloads["mistakes_download"]
    assert len(mistakes) <= 10
    assert mistakes.true_intent.ne(mistakes.predicted_intent).all()
    assert downloads["bins_download"].groupby("model_id").size().eq(10).all()
    assert downloads["bins_download"].split.eq(split).all()
    assert downloads["evidence_download"].split.eq("calibration").all()
    assert downloads["bootstrap_download"].iloc[0].n == 480


@pytest.mark.parametrize("method", ["kmeans", "gmm"])
@pytest.mark.parametrize("k", [6, 12, 18])
def test_group_representatives_are_ranked_members(app, downloads, method, k):
    app.sidebar.radio(key="screen").set_value("Explore request groups").run()
    app.selectbox(key="group_method").set_value(method)
    app.selectbox(key="group_k").set_value(k).run()
    app.selectbox(key="group_id").set_value(k - 1).run()
    assert not app.exception
    assignments = pd.read_csv(ROOT / "results/clusters.csv")
    selected = assignments[assignments.method.eq(method) & assignments.k.eq(k)
                           & assignments.seed.eq(0) & assignments.cluster_id.eq(k - 1)]
    expected = selected.sort_values(["representative_score", "source_id"], ascending=[False, True]).head(5)
    shown = downloads["reps_download"]
    assert shown.source_id.tolist() == expected.source_id.tolist()
    fit = pd.read_csv(ROOT / "data/development/fit.csv").set_index("source_id")
    assert shown.text.tolist() == fit.loc[shown.source_id, "text"].tolist()
    assert len(downloads["shuffles_download"]) == 100
    histogram = json.loads(app.get("plotly_chart")[-1].proto.spec)
    assert histogram["data"][0]["type"] == "bar"
    assert sum(histogram["data"][0]["y"]) == 100
    assert downloads["assignments_download"].seed.eq(0).all()
    assert downloads["assignments_download"].k.eq(k).all()


def test_representatives_handle_small_and_empty_groups():
    fit = pd.DataFrame({"source_id": ["a", "b"], "text": ["A", "B"],
                        "source_file": ["train.csv"] * 2, "source_csv_line": [2, 3]})
    assignments = pd.DataFrame({"source_id": ["a", "b"], "method": ["gmm"] * 2,
                                "k": [6] * 2, "seed": [0] * 2, "cluster_id": [0] * 2,
                                "representative_score": [.5, .9]})
    assert representatives(fit, assignments, "gmm", 6, 0).source_id.tolist() == ["b", "a"]
    assert representatives(fit, assignments, "gmm", 6, 1).empty


def test_review_all_and_undefined_error():
    frame = pd.DataFrame({"kind": ["threshold_trial"], "split": ["calibration"],
                          "model_id": ["route"], "threshold": [.9], "n": [10],
                          "accepted": [0], "wrong_accepted": [0], "coverage": [0],
                          "error": [np.nan], "run_id": ["run"]})
    result = threshold_table(frame, "calibration", "route", None)
    assert not result.saved_rule.any()
    assert not result.qualifies.any()
    assert pd.isna(result.iloc[0].error)


def test_csv_preserves_enquiry_text_and_probability_fields():
    cfg = read_json(ROOT / "config.json")
    labels = cfg["category_order"]
    result = {"text": 'My card, "please"\nWhere is it? café', "subset": "new_enquiry",
              "model_id": cfg["route_model_id"], "run_id": cfg["run_id"],
              "suggested_category": labels[0], "confidence": .75,
              "temperature": cfg["temperature"], "review_threshold": .5,
              "review_rule": "automatic if confidence >= 0.5", "status": "automatic route",
              "probabilities": {label: (.75 if index == 0 else .25 / 11)
                                for index, label in enumerate(labels)},
              "top_three": [{"category": label, "probability": .75 if index == 0 else .25 / 11}
                            for index, label in enumerate(labels[:3])]}
    restored = pd.read_csv(BytesIO(csv_bytes(route_export(result)))).iloc[0]
    assert restored.text == result["text"]
    assert restored.model_id == result["model_id"]
    assert restored.temperature == pytest.approx(result["temperature"])
    assert sum(restored[f"p__{label}"] for label in labels) == pytest.approx(1)
    assert restored.top_1_category == result["suggested_category"]


@pytest.mark.skipif(not (ROOT / "artifacts/encoder/model.safetensors").exists(),
                    reason="Download the pinned MiniLM encoder to run inference checks")
def test_real_routing_and_invalid_input_clears_previous_result(app, downloads):
    hashes = {name: digest(ROOT / name) for name in ["config.json", "artifacts/classifiers.joblib"]}
    app.sidebar.radio(key="screen").set_value("Route an enquiry").run()
    app.text_area(key="route_text").set_value("My card has not arrived")
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert not app.error
    result = downloads["route_download"].iloc[0]
    assert result.suggested_category == "card_arrival"
    assert result.status in ["automatic route", "needs human review"]
    probabilities = result.filter(like="p__").to_numpy(dtype=float)
    assert len(probabilities) == 12
    assert probabilities.sum() == pytest.approx(1)
    assert result.confidence == pytest.approx(probabilities.max())
    assert len(app.dataframe[0].value) == 12
    app.text_area(key="route_text").set_value(" \n ")
    app.button[0].click().run()
    assert app.error[0].value == "Enter a nonempty enquiry"
    assert "route_result" not in app.session_state
    assert not app.success
    assert hashes == {name: digest(ROOT / name) for name in hashes}


def test_missing_encoder_is_visible(app, monkeypatch):
    def unavailable(self, text):
        raise FileNotFoundError("MiniLM is missing. Run: python download_models.py")

    monkeypatch.setattr(Router, "predict", unavailable)
    app.sidebar.radio(key="screen").set_value("Route an enquiry").run()
    app.text_area(key="route_text").set_value("Where is my card?")
    app.button[0].click().run()
    assert not app.exception
    assert "download_models.py" in app.error[0].value


def test_agent_screen_is_explicitly_pending(app):
    app.sidebar.radio(key="screen").set_value("Ask the analysis agent").run()
    assert app.button[0].disabled
    assert "checkpoint 3" in app.info[0].value
