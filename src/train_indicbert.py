"""Fine-tune IndicBERT (or mBERT) for offensive / non-offensive classification."""
import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments)

MODELS = {"indicbert": "ai4bharat/indic-bert", "mbert": "bert-base-multilingual-cased"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--model", choices=MODELS, default="indicbert")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    tr, te = train_test_split(df[["text", "label"]], test_size=0.2, stratify=df["label"],
                              random_state=42)

    name = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    enc = lambda b: tok(b["text"], truncation=True, max_length=args.max_len)  # noqa: E731
    ds_tr = Dataset.from_pandas(tr, preserve_index=False).map(enc, batched=True)
    ds_te = Dataset.from_pandas(te, preserve_index=False).map(enc, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=2, id2label={0: "Non-Offensive", 1: "Offensive"},
        label2id={"Non-Offensive": 0, "Offensive": 1})

    def metrics(p):
        pred = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, pred),
                "macro_f1": f1_score(p.label_ids, pred, average="macro")}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=f"models/{args.model}", num_train_epochs=args.epochs,
                               per_device_train_batch_size=16, per_device_eval_batch_size=32,
                               learning_rate=2e-5, weight_decay=0.01, eval_strategy="epoch",
                               save_strategy="epoch", load_best_model_at_end=True,
                               metric_for_best_model="macro_f1", report_to="none"),
        train_dataset=ds_tr, eval_dataset=ds_te, tokenizer=tok, compute_metrics=metrics)
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(f"models/{args.model}/best")


if __name__ == "__main__":
    main()
