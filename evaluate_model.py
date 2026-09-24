#!/usr/bin/env python3
"""
HandsToVoice — Model evaluation report.

Runs stratified k-fold cross-validation using the app's own training pipeline
(src/training.py), so every recorded sequence is scored by a model that never
saw it. Also measures accuracy when the face isn't detected, and live
per-prediction latency of the deployed model. Does not touch models/.

Writes reports/evaluation_report.md, reports/confusion_matrix.png and
reports/evaluation_results.json.

Usage:  venv/bin/python3 evaluate_model.py [--folds 5] [--epochs 60]
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src import training
from src.training import SEQUENCE_LENGTH, FEATURE_LENGTH, HAND_FEATURE_LENGTH

SEQUENCE_DIR = "data/sequences"
REPORT_DIR = "reports"


def load_dataset():
    X, y = [], []
    with open(os.path.join(SEQUENCE_DIR, "labels.csv"), newline="") as f:
        for row in csv.DictReader(f):
            path = os.path.join(SEQUENCE_DIR, row["file"])
            if not os.path.exists(path):
                continue
            seq = np.load(path)
            if seq.shape == (SEQUENCE_LENGTH, FEATURE_LENGTH):
                X.append(seq)
                y.append(row["label"])
    return np.array(X, dtype=np.float32), np.array(y)


def cross_validate(X, y_enc, num_classes, folds, epochs, seed=42):
    from sklearn.model_selection import StratifiedKFold
    y_pred = np.empty_like(y_enc)
    y_pred_noface = np.empty_like(y_enc)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (dev_idx, test_idx) in enumerate(skf.split(X, y_enc), 1):
        t0 = time.time()
        rng = np.random.default_rng(seed + k)
        tr_rel, val_rel = training.split_per_class(y_enc[dev_idx], rng)
        tr_idx, val_idx = dev_idx[tr_rel], dev_idx[val_rel]
        X_train, y_train = training.augment(X[tr_idx], y_enc[tr_idx], rng)
        model = training.build_model(num_classes)
        training.fit(model, X_train, y_train, X[val_idx], y_enc[val_idx], epochs=epochs)

        X_test = X[test_idx]
        y_pred[test_idx] = model.predict(X_test, verbose=0).argmax(axis=1)
        X_noface = X_test.copy()
        X_noface[:, :, HAND_FEATURE_LENGTH:] = 0.0
        y_pred_noface[test_idx] = model.predict(X_noface, verbose=0).argmax(axis=1)

        acc = (y_pred[test_idx] == y_enc[test_idx]).mean()
        print(f"  fold {k}/{folds}: accuracy {acc:.1%} on {len(test_idx)} held-out "
              f"sequences ({time.time() - t0:.0f}s)", flush=True)
    return y_pred, y_pred_noface


def measure_latency(runs=200):
    """Per-prediction time of the deployed model, the way the app calls it."""
    from src.classifier import LSTMClassifier
    clf = LSTMClassifier()
    if not clf.is_ready():
        return None
    x = np.random.default_rng(0).normal(0, 0.1, (1, clf.SEQUENCE_LENGTH, clf.FEATURE_LENGTH))
    x = x.astype(np.float32)
    for _ in range(10):
        clf._predict_fn(x)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        clf._predict_fn(x).numpy()
        times.append((time.perf_counter() - t) * 1000)
    return {"mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95))}


def save_confusion_png(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("Predicted sign")
    ax.set_ylabel("Actual sign")
    ax.set_title("Cross-validated confusion matrix (held-out sequences only)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(res, path):
    lines = [
        "# HandsToVoice — Sign Recognition Evaluation",
        "",
        f"Generated {res['generated']} by `evaluate_model.py`.",
        "",
        "## Method",
        "",
        f"- **{res['n_sequences']} recorded sequences** of **{res['n_classes']} signs** "
        f"({res['min_per_class']}–{res['max_per_class']} per sign).",
        f"- **{res['folds']}-fold stratified cross-validation**: each sequence is predicted "
        "by a model trained without it, using the app's own training pipeline "
        "(`src/training.py`: same augmentation, architecture and early stopping).",
        "- **Face-missing test**: the same held-out sequences with the face-position "
        "features removed, as happens live when the face detector misses.",
        "- **Latency**: time for one prediction by the deployed model, as the app calls it.",
        "",
        "## Results",
        "",
        "| Measure | Result |",
        "|---|---|",
        f"| Accuracy (held-out) | **{res['accuracy']:.1%}** |",
        f"| Macro-average F1 | {res['macro_f1']:.3f} |",
        f"| Accuracy with face not detected | {res['accuracy_no_face']:.1%} |",
    ]
    if res["latency"]:
        lines.append(f"| Prediction latency | {res['latency']['mean_ms']:.1f} ms mean, "
                     f"{res['latency']['p95_ms']:.1f} ms 95th percentile |")
    lines += ["", "Confusion matrix: `confusion_matrix.png`.", "",
              "## Per-sign results", "",
              "| Sign | Precision | Recall | F1 | Samples |", "|---|---|---|---|---|"]
    for name, m in sorted(res["per_class"].items(), key=lambda kv: kv[1]["f1"]):
        lines.append(f"| {name} | {m['precision']:.2f} | {m['recall']:.2f} | "
                     f"{m['f1']:.2f} | {m['support']} |")
    lines += ["", "## Most confused signs", ""]
    if res["confusions"]:
        lines += ["| Actual | Predicted as | Times |", "|---|---|---|"]
        lines += [f"| {c['actual']} | {c['predicted']} | {c['count']} |"
                  for c in res["confusions"]]
    else:
        lines.append("No confusions between signs.")
    lines += [
        "",
        "## Limitations",
        "",
        "- **All recordings are from one signer**, so these numbers measure how well the "
        "model recognizes new attempts by the *same* person. Accuracy on a different "
        "signer is not measured here and is expected to be lower; that requires "
        "recordings from additional signers, kept entirely out of training.",
        "- Signs are recognized one at a time (isolated-sign recognition), not as "
        "continuous signed sentences.",
        "- Recordings were made in a limited set of rooms and lighting conditions.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from sklearn.preprocessing import LabelEncoder

    X, y = load_dataset()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)
    counts = np.bincount(y_enc)
    print(f"Loaded {len(X)} sequences, {len(classes)} signs. "
          f"Running {args.folds}-fold cross-validation...", flush=True)

    y_pred, y_pred_noface = cross_validate(X, y_enc, len(classes), args.folds, args.epochs)

    p, r, f1, support = precision_recall_fscore_support(
        y_enc, y_pred, labels=range(len(classes)), zero_division=0)
    cm = confusion_matrix(y_enc, y_pred, labels=range(len(classes)))
    off = [(cm[i, j], classes[i], classes[j])
           for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j]]
    off.sort(reverse=True)

    print("Measuring prediction latency of the deployed model...", flush=True)
    res = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_sequences": int(len(X)),
        "n_classes": len(classes),
        "min_per_class": int(counts.min()),
        "max_per_class": int(counts.max()),
        "folds": args.folds,
        "accuracy": float((y_pred == y_enc).mean()),
        "accuracy_no_face": float((y_pred_noface == y_enc).mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {classes[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                   "f1": float(f1[i]), "support": int(support[i])}
                      for i in range(len(classes))},
        "confusions": [{"actual": a, "predicted": b, "count": int(n)} for n, a, b in off[:10]],
        "latency": measure_latency(),
    }

    os.makedirs(REPORT_DIR, exist_ok=True)
    save_confusion_png(cm, classes, os.path.join(REPORT_DIR, "confusion_matrix.png"))
    with open(os.path.join(REPORT_DIR, "evaluation_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    write_report(res, os.path.join(REPORT_DIR, "evaluation_report.md"))

    print(f"\nAccuracy (held-out): {res['accuracy']:.1%}   "
          f"face not detected: {res['accuracy_no_face']:.1%}   macro F1: {res['macro_f1']:.3f}")
    print(f"Report written to {REPORT_DIR}/evaluation_report.md")


if __name__ == "__main__":
    main()
