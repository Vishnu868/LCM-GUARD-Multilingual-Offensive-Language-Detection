"""LCM-GUARD feature modalities: semantic, phonetic, context and cultural, fused by concatenation."""
import json
import re

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")


class ContextFeatures(BaseEstimator, TransformerMixin):
    """Handcrafted writing-style features for short, noisy text."""

    names = ["length", "n_words", "emoji", "exclaim", "question", "punct",
             "repeated_chars", "caps_ratio", "special_symbols", "digits"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            letters = [c for c in t if c.isalpha()]
            rows.append([
                len(t), len(t.split()), len(EMOJI_RE.findall(t)), t.count("!"), t.count("?"),
                sum(c in ".,;:!?" for c in t), len(re.findall(r"(.)\1{2,}", t)),
                sum(c.isupper() for c in letters) / max(len(letters), 1),
                len(re.findall(r"[@#$%^&*~]", t)), sum(c.isdigit() for c in t),
            ])
        return sparse.csr_matrix(np.log1p(np.array(rows, dtype=float)))


class CulturalFeatures(BaseEstimator, TransformerMixin):
    """Counts of lexicon-category hits (religion, caste, politics, slurs, threats, ...)."""

    def __init__(self, lexicon_path: str = "lexicons/categories.json"):
        self.lexicon_path = lexicon_path

    def fit(self, X, y=None):
        lex = json.load(open(self.lexicon_path, encoding="utf-8"))
        self.categories_ = {k: [w.lower() for w in v] for k, v in lex.items() if not k.startswith("_")}
        return self

    def transform(self, X):
        rows = []
        for t in X:
            low = t.lower()
            rows.append([sum(low.count(w) for w in words) for words in self.categories_.values()])
        return sparse.csr_matrix(np.array(rows, dtype=float))


def build_fusion_features(lexicon_path: str = "lexicons/categories.json") -> Pipeline:
    union = FeatureUnion([
        ("semantic", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                     max_features=50000, sublinear_tf=True)),
        ("phonetic", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                     max_features=100000, sublinear_tf=True)),
        ("context", Pipeline([("f", ContextFeatures()), ("s", MaxAbsScaler())])),
        ("cultural", Pipeline([("f", CulturalFeatures(lexicon_path)), ("s", MaxAbsScaler())])),
    ])
    return union
