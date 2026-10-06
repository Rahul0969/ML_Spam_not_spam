# Task 2 Summary — Spam SMS Classifier

**Problem.** Classify SMS messages as spam or ham.

**Data.** UCI SMS Spam Collection; <N> unique messages after removing duplicates, <X>% spam (imbalanced).

**Method.** Cleaned text (lowercase, URL/number tokens, no punctuation) -> TF-IDF with 1-2 grams -> compared Naive Bayes and Logistic Regression with 5-fold CV on the training set. Chosen model: <best model>.

**Results (20% held-out test set).** Accuracy <..>, Precision <..>, Recall <..>, F1 <..>. Confusion matrix: <TN> ham correct, <TP> spam caught, <FP> ham wrongly flagged, <FN> spam missed.

**Observations.** <Which errors did it make? e.g. short spam without keywords>. Because spam is the minority class, precision/recall/F1 matter more than accuracy. In practice a false positive (blocking a real message) is costlier than a missed spam.

**Limitations / next steps.** Small, English-only dataset from older SMS; could try SVM, word embeddings or a transformer, and add a simple web demo.

> Fill the <...> values from results/metrics.json after you run the notebook. Keep this file under 300 words.
