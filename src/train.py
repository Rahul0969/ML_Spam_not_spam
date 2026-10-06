"""
Task 2 - Spam SMS Classifier (training script)

Run from the project root:
    python src/train.py
    python src/train.py --data path/to/sms.tsv     # use your own copy of the dataset

Steps (numbered the same way as the notebook):
 1. Load the dataset (downloads it if data/sms.tsv is missing)
 2. Clean the text
 3. Stratified train/test split (80/20)
 4. Compare 2 models with 5-fold cross-validation on TRAIN only
 5. Train the best one, evaluate on the untouched TEST set
 6. Save metrics, confusion matrix, example predictions and the model
"""
import argparse, json, os, re, urllib.request
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

SEED = 42
DATA_URL = ("https://raw.githubusercontent.com/justmarkham/"
            "pycon-2016-tutorial/master/data/sms.tsv")   # mirror of UCI SMS Spam Collection


def clean_text(text: str) -> str:
    """Lowercase, replace urls/numbers with tokens, remove punctuation."""
    text = str(text).lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", " urltoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"Downloading dataset to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["target"] = (df["label"] == "spam").astype(int)      # ham=0, spam=1
    return df


def build_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")),
        ("clf", model),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sms.tsv")
    args = ap.parse_args()
    for d in ("models", "results"):
        os.makedirs(d, exist_ok=True)

    # 1. load
    df = load_data(args.data)
    print("Rows after removing duplicates:", len(df))
    print(df["label"].value_counts())

    # 2. clean
    df["clean"] = df["message"].apply(clean_text)

    # 3. split
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        df["clean"], df["target"], df["message"],
        test_size=0.2, stratify=df["target"], random_state=SEED)
    print("Train:", len(X_train), "Test:", len(X_test))

    # 4. compare models using cross-validation on the training set only
    candidates = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }
    cv_scores = {}
    for name, m in candidates.items():
        s = cross_val_score(build_pipeline(m), X_train, y_train, cv=5, scoring="f1")
        cv_scores[name] = float(s.mean())
        print(f"{name}: CV F1 = {s.mean():.4f}")
    best_name = max(cv_scores, key=cv_scores.get)
    print("Best model:", best_name)

    # 5. train best model + evaluate on test set
    model = build_pipeline(candidates[best_name]).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": best_name,
        "cv_f1": cv_scores,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # 6. save everything
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {best_name} (test set)")
    plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5, 3.5))
    vals = [metrics[k] for k in ("accuracy", "precision", "recall", "f1")]
    plt.bar(["Accuracy", "Precision", "Recall", "F1"], vals, color="#3b82f6")
    plt.ylim(0, 1.05)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.title("Test-set metrics")
    plt.savefig("results/metrics_chart.png", dpi=150, bbox_inches="tight")
    plt.close()

    # example predictions: 8 real test messages (4 spam, 4 ham) + 4 custom ones
    test_df = pd.DataFrame({"message": raw_test.values, "actual": y_test.values,
                            "predicted": y_pred, "spam_probability": y_prob})
    real = pd.concat([test_df[test_df.actual == 1].head(4), test_df[test_df.actual == 0].head(4)])
    real["actual"] = real["actual"].map({0: "ham", 1: "spam"})
    custom_msgs = [
        "Congratulations! You won a FREE iPhone. Click http://claim-prize.com now",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: your account is suspended, call 09061234567 to verify",
        "Can you send me the notes from today's class?",
    ]
    cprob = model.predict_proba([clean_text(m) for m in custom_msgs])[:, 1]
    custom = pd.DataFrame({"message": custom_msgs, "actual": "(custom)",
                           "predicted": (cprob >= 0.5).astype(int), "spam_probability": cprob})
    ex = pd.concat([real, custom], ignore_index=True)
    ex["predicted_label"] = ex["predicted"].map({0: "ham", 1: "spam"})
    ex["confidence"] = np.where(ex["predicted"] == 1, ex["spam_probability"], 1 - ex["spam_probability"])
    ex = ex[["message", "actual", "predicted_label", "confidence", "spam_probability"]].round(4)
    ex.to_csv("results/example_predictions.csv", index=False)
    print(ex[["message", "actual", "predicted_label", "confidence"]].to_string())

    joblib.dump(model, "models/spam_classifier.joblib")
    print("Saved models/spam_classifier.joblib and results/*")


if __name__ == "__main__":
    main()
