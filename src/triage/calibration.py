"""Confidence calibration and selective classification (decide when to hand off to a human)."""
import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T minimising validation NLL. Argmax is unchanged."""
    def nll(logT):
        p = softmax(logits, np.exp(logT))
        return -np.mean(np.log(p[np.arange(len(labels)), labels] + 1e-12))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: |accuracy - confidence| averaged over equal-width confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = pred == labels
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def threshold_for_accuracy(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Lowest confidence threshold whose auto-handled tickets reach `target` accuracy on validation data.

    Sort by confidence descending; accuracy of the top-k must stay >= target. Picking the threshold
    on validation and reporting on test keeps the automation rate honest.
    """
    order = np.argsort(-conf)
    acc = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    ok = np.flatnonzero(acc >= target)
    if not ok.size:
        return 1.01  # nothing can be automated at this target
    k = ok.max()
    return float(conf[order][k])


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-conf)
    cov = np.arange(1, len(conf) + 1) / len(conf)
    acc = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    return cov, acc
