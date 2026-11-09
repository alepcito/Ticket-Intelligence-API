"""Train and evaluate a Spanish ticket classifier: python -m training.train."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def train(data_path: Path, output_dir: Path) -> dict:
    with data_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    texts = [row["text"].strip() for row in rows]
    labels = [row["category"].strip() for row in rows]
    if any(not text or not label for text, label in zip(texts, labels)):
        raise ValueError("Text and category must not be empty")
    if len(set(texts)) != len(texts):
        raise ValueError("Duplicate texts found; remove them before splitting")
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True)),
        ("classifier", LogisticRegression(C=4.0, max_iter=1000, random_state=42)),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    categories = sorted(set(labels))
    report = {
        "model_version": "demo-v1",
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "dataset_type": "synthetic_demo",
        "random_state": 42,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "baseline_macro_f1": f1_score(y_test, baseline.predict(x_test), average="macro", zero_division=0),
        "categories": categories,
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=categories).tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Persist exactly the evaluated model; the holdout is never used for fitting.
    joblib.dump({"pipeline": model, "metadata": report}, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/tickets.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output), indent=2, ensure_ascii=False))
