"""
Inference for the Spam classifier.

Usage (from project root):
    python src/predict.py "Win a free prize now! Click http://x.com"
    python src/predict.py "msg one" "msg two"
"""
import re, sys
import joblib


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", " urltoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python src/predict.py "your message" ["another message" ...]')
    model = joblib.load("models/spam_classifier.joblib")
    for msg in sys.argv[1:]:
        p_spam = model.predict_proba([clean_text(msg)])[0, 1]
        label = "SPAM" if p_spam >= 0.5 else "HAM (not spam)"
        conf = p_spam if p_spam >= 0.5 else 1 - p_spam
        print(f"[{label}] confidence={conf:.2%} | {msg}")


if __name__ == "__main__":
    main()
