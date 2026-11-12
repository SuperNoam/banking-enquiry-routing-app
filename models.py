"""Fit-only feature models, stable probabilities, metrics and routing inference."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from common import ROOT, read_json


def probabilities(logits, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    return softmax(np.asarray(logits, dtype=float) / temperature, axis=1)


def losses(logits, targets, temperature=1.0):
    logp = log_softmax(np.asarray(logits, dtype=float) / temperature, axis=1)
    return -logp[np.arange(len(targets)), np.asarray(targets, dtype=int)]


def fit_temperature(logits, targets, bounds=(-4, 4)):
    result = minimize_scalar(lambda u: losses(logits, targets, np.exp(u)).mean(),
                             method="bounded", bounds=bounds, options={"xatol": 1e-10})
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def score_probabilities(p, targets):
    """Zero-denominator precision/recall/F1 = 0; clipping only for log(p)."""
    targets = np.asarray(targets, dtype=int)
    predicted = p.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predicted, labels=np.arange(p.shape[1]), zero_division=0)
    return {
        "n": len(targets), "accuracy": float(np.mean(predicted == targets)),
        "macro_f1": float(f1.mean()),
        "mean_log_loss": float(-np.log(np.clip(p[np.arange(len(targets)), targets], 1e-15, 1)).mean()),
        "confusion_matrix": confusion_matrix(targets, predicted, labels=np.arange(p.shape[1])).tolist(),
        "per_category": [{"precision": float(a), "recall": float(b), "f1": float(c), "support": int(d)}
                         for a, b, c, d in zip(precision, recall, f1, support)]}


def reliability_bins(p, targets):
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == targets
    bins = np.minimum(np.floor(confidence * 10).astype(int), 9)
    return [{"bin": i, "lower": i / 10, "upper": (i + 1) / 10,
             "upper_inclusive": i == 9, "count": int((bins == i).sum()),
             "mean_confidence": float(confidence[bins == i].mean()) if (bins == i).any() else None,
             "fraction_correct": float(correct[bins == i].mean()) if (bins == i).any() else None}
            for i in range(10)]


def review_outcome(p, targets, threshold):
    accepted = np.zeros(len(p), dtype=bool) if threshold is None else p.max(axis=1) >= threshold
    wrong = accepted & (p.argmax(axis=1) != targets)
    n_accept = int(accepted.sum())
    return {"threshold": threshold, "n": len(p), "accepted": n_accept,
            "review_required": len(p) - n_accept, "wrong_accepted": int(wrong.sum()),
            "coverage": n_accept / len(p), "error": int(wrong.sum()) / n_accept if n_accept else None}


def choose_review_rule(p, targets):
    rows = [review_outcome(p, targets, t) for t in [0.5, 0.6, 0.7, 0.8, 0.9]]
    for row in rows:
        row["qualifies"] = row["accepted"] >= 20 and row["error"] <= .05
    qualifying = [x["threshold"] for x in rows if x["qualifies"]]
    return (min(qualifying) if qualifying else None), rows


@lru_cache(maxsize=2)
def load_encoder(root_string=str(ROOT)):
    from sentence_transformers import SentenceTransformer
    root = ROOT.__class__(root_string)
    cfg = read_json(root / "config.json")["encoder"]
    path = root / cfg["path"]
    if not (path / "model.safetensors").exists():
        raise FileNotFoundError("MiniLM is missing. Run: python download_models.py")
    encoder = SentenceTransformer(str(path), device="cpu", local_files_only=True)
    encoder.eval()
    return encoder


def encode(texts, root=ROOT):
    encoder = load_encoder(str(root))
    return encoder.encode(list(texts), normalize_embeddings=True, batch_size=64,
                          show_progress_bar=False, convert_to_numpy=True)


class Router:
    """Read saved artifacts only. Inference never fits any model."""
    def __init__(self, root=ROOT):
        self.root = root
        self.config = read_json(root / "config.json")
        self.labels = self.config["category_order"]
        self.bundle = joblib.load(root / "artifacts/classifiers.joblib")

    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a nonempty enquiry")
        if len(text) > 20000:
            raise ValueError("Enquiry exceeds 20,000 characters")
        chosen = self.config["chosen_method"]
        features = (self.bundle["tfidf_vectorizer"].transform([text]) if chosen == "tfidf"
                    else encode([text], self.root))
        model = self.bundle[chosen]
        p = probabilities(model.decision_function(features), self.config["temperature"])[0]
        order = np.argsort(-p, kind="stable")
        t = self.config["review_threshold"]
        return {"text": text, "subset": "new_enquiry", "model_id": self.config["route_model_id"],
                "run_id": self.config["run_id"], "suggested_category": self.labels[order[0]],
                "confidence": float(p[order[0]]),
                "probabilities": {label: float(value) for label, value in zip(self.labels, p)},
                "top_three": [{"category": self.labels[i], "probability": float(p[i])} for i in order[:3]],
                "temperature": self.config["temperature"], "review_threshold": t,
                "review_rule": "review every enquiry" if t is None else f"automatic if confidence >= {t}",
                "status": "automatic route" if t is not None and p.max() >= t else "needs human review"}
