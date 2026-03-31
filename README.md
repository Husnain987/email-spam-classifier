# Email Spam Classifier

Binary text classifier that distinguishes spam from ham (legitimate) emails using feature engineering and logistic regression.

Built as part of **Data 100 (Principles and Techniques of Data Science)** at UC Berkeley.

---

## Overview

Given a dataset of 8,348 real-world labeled emails from [SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/), the goal is to build a classifier that predicts whether an email is spam (1) or ham (0).

The project covers the full ML pipeline: exploratory data analysis, feature engineering on raw text, model training, and evaluation using precision, recall, and false positive rate.

---

## Key Steps

### 1. Data Cleaning
- Loaded and lowercased all email text
- Imputed missing values in subject/email fields with empty strings
- Applied a 90/10 train-validation split (`random_state=42`)

### 2. Feature Engineering
- Implemented `words_in_texts()` — a vectorized function that converts a list of target words and a Series of emails into a binary NumPy matrix
- Selected discriminative words by comparing spam vs. ham proportions visually
- Words like `free`, `click`, `offer`, and `remove` appeared significantly more in spam than ham

### 3. Model Training
- Built feature matrix `X_train` and label vector `Y_train` from 5 keyword features
- Trained a `LogisticRegression` classifier via scikit-learn
- Achieved ~76% training accuracy

### 4. Evaluation

| Metric | Zero Predictor | Logistic Regression |
|--------|---------------|---------------------|
| Accuracy | ~74% | ~76% |
| Recall (spam) | 0.0 | > 0 |
| False Positive Rate | 0.0 | Low |

---

## Tech Stack

- Python, Jupyter Notebook
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn

---

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook projB1.ipynb
```

Unzip `spam_ham_data.zip` in the same directory before running.
