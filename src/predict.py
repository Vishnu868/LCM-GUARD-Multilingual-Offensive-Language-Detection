"""Classify new text with the trained LCM-GUARD fusion model.

Usage:  python src/predict.py "Nuvvu idiot ra" "Have a nice day"
"""
import os
import sys

import joblib

sys.path.append(os.path.dirname(__file__))  # so pickled feature classes resolve


def main():
    texts = sys.argv[1:] or ["Nuvvu idiot ra", "Have a great day everyone"]
    model = joblib.load("models/lcm_guard.joblib")
    scores = model.decision_function(texts)
    for t, s in zip(texts, scores):
        label = "Offensive" if s > 0 else "Non-Offensive"
        print(f"{label:<14} (margin {s:+.3f})  {t}")


if __name__ == "__main__":
    main()
