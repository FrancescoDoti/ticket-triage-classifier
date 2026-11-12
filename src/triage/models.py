"""Three classifiers behind one interface: fit(texts, y) and logits(texts) -> (n, classes)."""
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline


def as_logits(scores: np.ndarray) -> np.ndarray:
    """sklearn returns one column for binary problems; softmax([0, z]) == sigmoid(z) restores two."""
    return np.column_stack([np.zeros_like(scores), scores]) if scores.ndim == 1 else scores


class TfidfLR:
    name = "TF-IDF + logistic regression"

    def fit(self, texts, y):
        feats = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                              ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2))])
        self.pipe = make_pipeline(feats, LogisticRegression(C=20, max_iter=3000))
        self.pipe.fit(texts, y)
        return self

    def logits(self, texts):
        return as_logits(self.pipe.decision_function(texts))


class EmbeddingLR:
    def __init__(self, model="BAAI/bge-small-en-v1.5"):
        self.model_name = model
        self.name = f"{model.split('/')[-1]} embeddings + logistic regression"

    def _enc(self, texts):
        if not hasattr(self, "enc"):
            from sentence_transformers import SentenceTransformer
            self.enc = SentenceTransformer(self.model_name)
        return self.enc.encode(list(texts), batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    def fit(self, texts, y):
        self.lr = LogisticRegression(C=10, max_iter=3000).fit(self._enc(texts), y)
        return self

    def logits(self, texts):
        return as_logits(self.lr.decision_function(self._enc(texts)))


class FineTuned:
    """Full fine-tune of a small encoder with a classification head."""

    def __init__(self, model="BAAI/bge-small-en-v1.5", epochs=6, lr=5e-5, head_lr=2e-3, batch=32, seed=0):
        self.model_name, self.epochs, self.lr, self.head_lr, self.batch, self.seed = model, epochs, lr, head_lr, batch, seed
        self.name = f"fine-tuned {model.split('/')[-1]}"

    def fit(self, texts, y, n_classes=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        torch.manual_seed(self.seed)
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        n_classes = n_classes or int(max(y)) + 1
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=n_classes).to(self.device)
        # The pooler and classifier start random; at the encoder's small learning rate they barely move,
        # so they get their own (much larger) one.
        head = [p for n, p in self.model.named_parameters() if n.startswith(("classifier", "bert.pooler"))]
        body = [p for n, p in self.model.named_parameters() if not n.startswith(("classifier", "bert.pooler"))]
        opt = torch.optim.AdamW([{"params": body, "lr": self.lr}, {"params": head, "lr": self.head_lr}], weight_decay=0.01)
        steps = self.epochs * int(np.ceil(len(texts) / self.batch))
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        texts, y = list(texts), np.asarray(y)
        rng = np.random.default_rng(self.seed)
        self.model.train()
        for ep in range(self.epochs):
            t0, order, tot = time.time(), rng.permutation(len(texts)), 0.0
            for i in range(0, len(order), self.batch):
                idx = order[i:i + self.batch]
                b = self.tok([texts[j] for j in idx], padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
                out = self.model(**b, labels=torch.tensor(y[idx], device=self.device))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                tot += out.loss.item() * len(idx)
            print(f"  epoch {ep + 1}/{self.epochs} loss {tot / len(texts):.4f} ({time.time() - t0:.0f}s)", flush=True)
        self.model.eval()
        return self

    def logits(self, texts):
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                b = self.tok(list(texts[i:i + 256]), padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
                out.append(self.model(**b).logits.float().cpu().numpy())
        return np.concatenate(out)
