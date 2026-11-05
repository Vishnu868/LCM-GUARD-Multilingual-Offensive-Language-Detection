"""Train the LCM-GUARD 4-modal fusion model (Linear SVM) and compare classical ML baselines."""
import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from features import build_fusion_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--lexicon", default="lexicons/categories.json")
    ap.add_argument("--compare", action="store_true", help="also train baseline classifiers")
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["text", "label"])
    langs = df["language"] if "language" in df.columns else pd.Series("all", index=df.index)
    X_tr, X_te, y_tr, y_te, lang_tr, lang_te = train_test_split(
        df["text"].astype(str), df["label"].astype(int), langs,
        test_size=0.2, stratify=df["label"], random_state=42)

    classifiers = {"LCM-GUARD (Linear SVM)": LinearSVC(C=0.5, class_weight="balanced")}
    if args.compare:
        from xgboost import XGBClassifier
        classifiers.update({
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "SVM (RBF)": SVC(kernel="rbf", C=2.0, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                     eval_metric="logloss"),
        })

    rows, best = [], None
    for name, clf in classifiers.items():
        pipe = Pipeline([("fusion", build_fusion_features(args.lexicon)), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({"model": name, "accuracy": accuracy_score(y_te, pred),
                     "macro_f1": f1_score(y_te, pred, average="macro")})
        if best is None:
            best = (pipe, pred)

    print(pd.DataFrame(rows).round(4).to_string(index=False))
    pipe, pred = best
    print("\nLCM-GUARD classification report:\n",
          classification_report(y_te, pred, target_names=["Non-Offensive", "Offensive"]))
    if lang_te.nunique() > 1:
        per_lang = pd.DataFrame({"lang": lang_te.values, "y": y_te.values, "p": pred})
        print("Per-language macro F1:")
        print(per_lang.groupby("lang").apply(
            lambda g: f1_score(g["y"], g["p"], average="macro")).round(4))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/lcm_guard.joblib")
    print("Saved models/lcm_guard.joblib")


if __name__ == "__main__":
    main()
