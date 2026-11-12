import joblib
import numpy as np
import pytest

from triage.calibration import ece, fit_temperature, risk_coverage, softmax, threshold_for_accuracy
from triage.models import TfidfLR


def test_softmax_rows_sum_to_one_and_temperature_keeps_argmax():
    z = np.array([[2.0, 1.0, 0.0], [0.0, 5.0, 1.0]])
    for T in (0.5, 1.0, 3.0):
        p = softmax(z, T)
        assert np.allclose(p.sum(1), 1) and (p.argmax(1) == z.argmax(1)).all()


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 4000, 5
    labels = rng.integers(0, k, n)
    true_logits = rng.normal(0, 1, (n, k))
    true_logits[np.arange(n), labels] += 1.5
    # sample labels from the true model so it is calibrated at T=1, then sharpen the logits 3x
    p = softmax(true_logits)
    y = np.array([rng.choice(k, p=row) for row in p])
    T = fit_temperature(true_logits * 3, y)
    assert T == pytest.approx(3, rel=0.15)
    assert ece(softmax(true_logits * 3, T), y) < ece(softmax(true_logits * 3), y)


def test_threshold_meets_target_on_the_data_it_was_fit_on():
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.6, 0.5])
    correct = np.array([1, 1, 1, 0, 1, 0], dtype=bool)
    th = threshold_for_accuracy(conf, correct, 1.0)
    assert th == 0.9
    assert threshold_for_accuracy(conf, np.zeros(6, bool), 0.9) > 1.0
    cov, acc = risk_coverage(conf, correct)
    assert cov[-1] == 1.0 and acc[2] == 1.0


def test_api_routes_by_threshold(tmp_path, monkeypatch):
    texts = ["card not arrived", "where is my card", "card still missing", "refund please", "i want my money back", "reverse the payment"] * 5
    y = [0, 0, 0, 1, 1, 1] * 5
    m = TfidfLR().fit(texts, y)
    path = tmp_path / "m.joblib"
    joblib.dump({"pipe": m.pipe, "classes": ["card_arrival", "refund"], "temperature": 1.0, "threshold": 0.5}, path)
    monkeypatch.setenv("TRIAGE_MODEL", str(path))
    from fastapi.testclient import TestClient
    from triage import api
    api.bundle.cache_clear()
    c = TestClient(api.app)
    r = c.post("/classify", json={"text": "my card has not arrived"}).json()
    assert r["intent"] == "card_arrival" and r["route"] == "auto"
    assert c.post("/classify", json={"text": ""}).status_code == 422
