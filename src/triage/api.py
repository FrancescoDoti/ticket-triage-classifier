"""POST /classify -> intent, calibrated confidence, and whether a human should take it.

Serves the TF-IDF model (0.03 ms/ticket on CPU, no GPU or model download) produced by
`python -m triage.experiment`. TRIAGE_MODEL points at the bundle (default models/tfidf.joblib).
"""
import os
from functools import lru_cache

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .calibration import softmax
from .models import as_logits

app = FastAPI(title="ticket triage")


@lru_cache(maxsize=1)
def bundle():
    return joblib.load(os.environ.get("TRIAGE_MODEL", "models/tfidf.joblib"))


class Ticket(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.post("/classify")
def classify(t: Ticket):
    b = bundle()
    p = softmax(as_logits(b["pipe"].decision_function([t.text])), b["temperature"])[0]
    top = np.argsort(-p)[:3]
    conf = float(p[top[0]])
    return {
        "intent": b["classes"][top[0]], "confidence": round(conf, 4),
        "route": "auto" if conf >= b["threshold"] else "human",
        "alternatives": [{"intent": b["classes"][i], "confidence": round(float(p[i]), 4)} for i in top[1:]],
    }
