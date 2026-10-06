# ML_2_SpamClassifier — Spam SMS Classifier

AVIP 2026 · AI/ML Engineering · Task 2.
Classifies an SMS as **spam** or **ham** using TF-IDF features + a classical ML model (scikit-learn).

## Dataset
UCI **SMS Spam Collection** (free, public): https://archive.ics.uci.edu/dataset/228/sms+spam+collection
The code downloads a tab-separated mirror automatically into `data/sms.tsv`
(format: `label<TAB>message`). If you're offline, download it manually and save it as `data/sms.tsv`.

## Preprocessing
1. Remove empty rows and duplicate messages
2. Lowercase
3. Replace URLs -> `urltoken`, numbers -> `numtoken`
4. Remove punctuation, collapse spaces
5. TF-IDF (1-2 word phrases, English stop-words removed, `min_df=2`)
6. Stratified 80/20 train/test split (`random_state=42`)

## Model selection
Naive Bayes vs Logistic Regression, compared by 5-fold cross-validated F1 on the **training set only**.
The best one is trained and evaluated once on the untouched test set.

## Project structure
```
ML_2_SpamClassifier/
├── notebooks/spam_classifier.ipynb   # full walkthrough (run top to bottom)
├── src/train.py                      # same pipeline as a script
├── src/predict.py                    # inference from the command line
├── models/spam_classifier.joblib     # saved model (created after training)
├── results/                          # metrics.json, confusion_matrix.png,
│                                     # metrics_chart.png, example_predictions.csv
├── data/sms.tsv                      # dataset (auto-downloaded)
├── TASK2_SUMMARY.md
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt

# Option A - notebook
jupyter notebook          # open notebooks/spam_classifier.ipynb -> Kernel > Restart & Run All

# Option B - script (from the project root)
python src/train.py
```

## Run inference
```bash
python src/predict.py "WINNER! You won a free prize. Click http://claim.com now"
python src/predict.py "Are we meeting for lunch tomorrow?"
```
Output: label + confidence for each message.

## Results
See `results/metrics.json`, `results/confusion_matrix.png` and `results/example_predictions.csv`
(generated when you run the notebook/script — numbers are not hard-coded).

## Author
<your name> · Arithmatrix Virtual Internship Program (AVIP) 2026 · #AVIP2026
