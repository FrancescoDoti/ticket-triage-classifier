"""python -m triage.experiment [--models tfidf emb ft]

Train on Banking77 train minus a stratified 10% validation split. Temperature and
auto-handling thresholds are fit on validation; every reported number is on the
official 3,080-query test set.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from .calibration import ece, fit_temperature, risk_coverage, softmax, threshold_for_accuracy
from .models import EmbeddingLR, FineTuned, TfidfLR

TARGETS = [0.95, 0.98, 0.99]


def load(data: Path):
    tr, te = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    classes = sorted(tr["category"].unique())
    idx = {c: i for i, c in enumerate(classes)}
    return tr["text"].to_numpy(), tr["category"].map(idx).to_numpy(), te["text"].to_numpy(), te["category"].map(idx).to_numpy(), classes


def evaluate(model, Xv, yv, Xt, yt, classes):
    lv = model.logits(Xv)
    t0 = time.perf_counter()
    lt = model.logits(Xt)
    ms = (time.perf_counter() - t0) / len(Xt) * 1000
    T = fit_temperature(lv, yv)
    pv, pt_raw, pt = softmax(lv, T), softmax(lt, 1.0), softmax(lt, T)
    pred = pt.argmax(1)
    correct = pred == yt
    nll = lambda p: float(-np.mean(np.log(p[np.arange(len(yt)), yt] + 1e-12)))
    sel = {}
    for target in TARGETS:
        th = threshold_for_accuracy(pv.max(1), pv.argmax(1) == yv, target)
        auto = pt.max(1) >= th
        sel[str(target)] = {"threshold": th, "test_automation_rate": float(auto.mean()),
                            "test_accuracy_on_automated": float(correct[auto].mean()) if auto.any() else None}
    cm = confusion_matrix(yt, pred)
    np.fill_diagonal(cm, 0)
    top = np.dstack(np.unravel_index(np.argsort(-cm.ravel())[:8], cm.shape))[0]
    return {
        "val_accuracy": float((lv.argmax(1) == yv).mean()),
        "accuracy": float(accuracy_score(yt, pred)), "macro_f1": float(f1_score(yt, pred, average="macro")),
        "temperature": T, "nll_raw": nll(pt_raw), "nll_calibrated": nll(pt),
        "ece_raw": ece(pt_raw, yt), "ece_calibrated": ece(pt, yt), "ms_per_ticket": ms,
        "selective": sel,
        "top_confusions": [{"true": classes[i], "predicted": classes[j], "count": int(cm[i, j])} for i, j in top],
    }, pt, correct


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", nargs="+", default=["tfidf", "emb", "ft"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    X, y, Xt, yt, classes = load(Path(a.data))
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.1, stratify=y, random_state=0)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    prev = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    curves = {}
    for key in a.models:
        model = {"tfidf": TfidfLR, "emb": EmbeddingLR, "ft": FineTuned,
                 "ft_minilm": lambda: FineTuned("microsoft/MiniLM-L12-H384-uncased", epochs=6, lr=8e-5, head_lr=8e-5),
                 "ft_samelr": lambda: FineTuned(epochs=5, head_lr=5e-5)}[key]()
        print(f"== {model.name}", flush=True)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        train_s = time.perf_counter() - t0
        res, pt, correct = evaluate(model, Xv, yv, Xt, yt, classes)
        res["train_seconds"] = train_s
        res["name"] = model.name
        prev[key] = res
        cov, acc = risk_coverage(pt.max(1), correct)
        curves[key] = (model.name, cov, acc)
        np.save(out / f"curve_{key}.npy", np.stack([cov, acc]))
        print(json.dumps({k: v for k, v in res.items() if k != "top_confusions"}, indent=1), flush=True)
        if key == "tfidf":
            Path("models").mkdir(exist_ok=True)
            joblib.dump({"pipe": model.pipe, "classes": classes, "temperature": res["temperature"],
                         "threshold": res["selective"]["0.98"]["threshold"]}, "models/tfidf.joblib")
    (out / "results.json").write_text(json.dumps(prev, indent=2))
    plot(out)


def plot(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads((out / "results.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, color in [("tfidf", "#9ca3af"), ("emb", "#f59e0b"), ("ft", "#2563eb")]:
        f = out / f"curve_{key}.npy"
        if f.exists():
            cov, acc = np.load(f)
            ax.plot(cov, acc, c=color, label=f"{res[key]['name']} (acc {res[key]['accuracy']:.3f})")
    ax.set_xlabel("share of tickets handled automatically (most confident first)")
    ax.set_ylabel("accuracy on the automated share")
    ax.set_ylim(0.85, 1.001)
    ax.axhline(0.98, ls="--", c="k", lw=0.8)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Accuracy vs automation rate, Banking77 test set")
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=140)


if __name__ == "__main__":
    main()
