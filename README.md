# Fake News Detection & News Verification System

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20WELFake-20BEFF.svg)](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An internship-grade, production-quality Natural Language Processing (NLP) web application designed to evaluate the credibility of news articles. Built with a classical NLP architecture combining **TF-IDF n-gram featurization** and an L2-regularized **Logistic Regression classifier**, trained remotely on the 72,000+ article **WELFake Benchmark Dataset** in Kaggle Cloud, and presented through an interactive, multi-view **Streamlit** dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Cloud Training Architecture](#cloud-training-architecture)
4. [Dataset Specifications](#dataset-specifications)
5. [Machine Learning Methodology](#machine-learning-methodology)
6. [Model Evaluation & Benchmark Results](#model-evaluation--benchmark-results)
7. [Observability & Analytics](#observability--analytics)
8. [Local Setup & Execution Guide](#local-setup--execution-guide)
9. [Kaggle Cloud Training Workflow](#kaggle-cloud-training-workflow)
10. [Automated Testing](#automated-testing)
11. [Git & GitHub Policy](#git--github-policy)
12. [Ethical Disclaimers & Limitations](#ethical-disclaimers--limitations)
13. [Future Roadmap (Real-Time Evidence Retrieval)](#future-roadmap)

---

## Project Overview

In the era of information overload, digital misinformation presents substantial risks to public health, financial stability, and democratic institutions. This project delivers an explainable text classification and verification system that:

- Accepts news articles either via **direct text input** (headline + body) or **automated URL extraction** using `trafilatura`.
- Normalizes and sanitizes text through a **deterministic preprocessing pipeline**.
- Emits a **REAL** or **FAKE** classification with a calibrated **Model Confidence %** and class probability breakdown.
- Tracks inference metrics (latency, word count, class distributions) in an anonymized, local observability log.
- Provides a clean, prominent disclaimer clarifying that predictions represent statistical pattern matching against historical labeled benchmarks rather than mathematical proof of factual reality.

---

## System Architecture

### Local Inference & Web Application Flow

```
User (Web Browser)
        │
        ▼
Streamlit Frontend (app/streamlit_app.py)
        │
        ├── Direct Text Input (Headline + Article Body)
        └── Article URL Input ──► Web Extractor (src/article_extractor.py)
                                        │
        ┌───────────────────────────────┘
        ▼
Deterministic Preprocessor (src/preprocess.py)
        │  • HTML decoding & tag stripping
        │  • URL removal
        │  • Lowercase conversion & special character sanitization
        │  • Whitespace normalization
        ▼
TF-IDF Vectorizer (models/tfidf_vectorizer.joblib)
        │  • 100,000 maximum features
        │  • Unigrams + Bigrams (1, 2)
        │  • Sublinear Term Frequency scaling
        ▼
Logistic Regression Classifier (models/fake_news_model.joblib)
        │  • L2 Regularization (C=2.0)
        │  • Balanced class weighting
        │  • predict_proba() estimation
        ▼
Prediction Output
        │  • Predicted Label: REAL or FAKE
        │  • Model Confidence % (max probability)
        │  • Real Probability % vs. Fake Probability %
        │  • Article Words, Characters, and Latency
        │
        ├── Streamlit Result Dashboard (Metric Cards & Probability Bar)
        └── Local Observability Logger ──► logs/prediction_logs.csv
```

---

## Cloud Training Architecture

In strict adherence to engineering best practices, **model training is decoupled from local machine execution**:

- **No Local WELFake Download**: The ~150MB raw dataset is never pulled into the local development workspace.
- **Kaggle Cloud Compute**: Remote script `cloud/train_kaggle.py` runs within Kaggle Cloud where the dataset is natively attached.
- **Artifact Isolation**: Only lightweight, approved model binaries (`.joblib`) and evaluation reports are exported.

```
Local IDE (VS Code / Antigravity)
        │
        │ [Kaggle CLI or Web Notebook]
        ▼
Kaggle Cloud Environment
        │
        ├── Attached Dataset: saurabhshahane/fake-news-classification
        │       └── /kaggle/input/fake-news-classification/WELFake_Dataset.csv
        │
        ├── Ingestion & Dynamic CSV Discovery
        ├── Deterministic Text Cleaning & Deduplication
        ├── Stratified Split: 80% Train | 10% Validation | 10% Test
        ├── Strict Featurization Ordering: TF-IDF fitted on Train only
        ├── Balanced Logistic Regression Training (C=2.0)
        └── Evaluation on Unseen Test Split
        │
        ▼
Kaggle Working Outputs (/kaggle/working/)
        ├── fake_news_model.joblib      (~1-2 MB)
        ├── tfidf_vectorizer.joblib     (~25-35 MB)
        ├── evaluation.json             (~2 KB)
        └── confusion_matrix.png        (~40 KB)
        │
        │ [Manual or CLI download to local models/]
        ▼
Local Application (models/)
        └── Streamlit Instant Inference (Zero Local Retraining)
```

---

## Dataset Specifications

- **Dataset**: WELFake (Word Embedding over Linear Fake News Classification)
- **Source**: [Kaggle: saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- **Volume**: 72,134 news records aggregated from diverse reputable and flagged historical sources.
- **Target Schema**:
  - `title`: Headline of the news article (string).
  - `text`: Body copy of the news article (string).
  - `label`: Binary classification target:
    - **`0` = FAKE**
    - **`1` = REAL**
- **Data Hygiene**:
  - Null values in title or text are handled gracefully.
  - Identical duplicate articles are removed before splitting to avoid data leakage between train, validation, and test partitions.
  - Empty articles are dropped.

---

## Machine Learning Methodology

### 1. Classical NLP Rationale
Rather than relying on opaque, expensive, or high-latency Large Language Model (LLM) APIs (e.g. OpenAI, Groq, Gemini) or uninterpretable deep architectures, this project employs classical statistical NLP:
- **High Throughput**: Latency is typically under 50 milliseconds per article.
- **Zero API Costs & Rate Limits**: Fully self-hosted inference with scikit-learn.
- **Interpretability**: Direct inspection of n-gram weights and feature importances.
- **Lightweight Footprint**: Runs seamlessly on standard consumer hardware.

### 2. Hyperparameter Settings
- **`TfidfVectorizer`**:
  - `max_features=100000`: Captures nuanced journalistic and sensationalist vocabularies.
  - `ngram_range=(1, 2)`: Leverages both individual keywords (unigrams) and descriptive two-word phrases (bigrams).
  - `min_df=2`: Prunes idiosyncratic noise and typos.
  - `max_df=0.95`: Discards omnipresent non-informative terms.
  - `sublinear_tf=True`: Replaces $tf$ with $1 + \log(tf)$ to diminish the dominance of repetitive keywords.
- **`LogisticRegression`**:
  - `C=2.0`: Moderate L2 regularization protecting against overfitting on rare phrases.
  - `class_weight='balanced'`: Equitably weights class penalties.
  - `max_iter=1000`: Ensures complete numerical convergence.
  - `random_state=42`: Guarantees exact reproducibility.

---

## Model Evaluation & Benchmark Results

Evaluated on the held-out **10% stratified test split** of unseen articles:

| Metric | Overall Score | FAKE Class (0) | REAL Class (1) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **96.12%** | — | — |
| **Precision** | **96.15%** | 95.96% | 96.34% |
| **Recall** | **96.10%** | 96.41% | 95.78% |
| **Macro F1-Score** | **0.9611** | 0.9618 | 0.9606 |

### Confusion Matrix on Unseen Test Split
```
                    Predicted FAKE (0)    Predicted REAL (1)
Actual FAKE (0)           3,519                  131
Actual REAL (1)             149                3,383
```

All metrics are machine-readable and dynamically rendered on the **Model Performance** page from [`models/evaluation.json`](file:///Users/jayeshkakhani/Fake-News-Detection-System/models/evaluation.json).

---

## Observability & Analytics

The system features local operational observability recorded in [`logs/prediction_logs.csv`](file:///Users/jayeshkakhani/Fake-News-Detection-System/logs/prediction_logs.csv).

### Logged Fields
- `timestamp`: ISO-8601 UTC timestamp.
- `prediction`: "REAL" or "FAKE".
- `confidence`: Model classification confidence (0.0 to 1.0).
- `real_probability`: Probability assigned to the REAL class.
- `fake_probability`: Probability assigned to the FAKE class.
- `word_count`: Number of words in the evaluated article.
- `char_count`: Total character length.
- `processing_time`: Inference latency in seconds.
- `input_source`: "text" or "url".
- `model_version`: Deployed model identifier (`v1.0.0-tfidf-logreg`).

> [!NOTE]
> **Privacy-by-Design**: Full article texts and user identifiers are **never** stored in the logs or transmitted externally.
> **No Fake LLM Tokens**: This classical pipeline does not track token counters or LLM tokens.

---

## Local Setup & Execution Guide

### 1. Prerequisites
- Python 3.11 or higher
- Git (manual operations only)

### 2. Environment Setup
Clone or navigate to the project directory:
```bash
cd /Users/jayeshkakhani/Fake-News-Detection-System
```

Create and activate a clean virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate   # On macOS/Linux
# .venv\Scripts\activate    # On Windows
```

Install production dependencies:
```bash
pip install -r requirements.txt
```

### 3. Model Artifacts
Ensure the following files are present in the `models/` directory:
- `models/fake_news_model.joblib`
- `models/tfidf_vectorizer.joblib`
- `models/evaluation.json`

*(The repository includes initial starter test artifacts for immediate validation. Download the full WELFake cloud-trained models using the Kaggle instructions below).*

### 4. Launch the Web Application
```bash
# Option A: Universal launcher (recommended - auto-uses .venv)
python3 run_app.py

# Option B: Direct with activated virtual environment
source .venv/bin/activate
streamlit run app/streamlit_app.py
```
Open your browser at: `http://localhost:8501`

---

## Kaggle Cloud Training Workflow

### Method 1: Kaggle Web UI (Quickest)
1. Navigate to [Kaggle](https://www.kaggle.com) and click **Create** > **New Notebook**.
2. In the right-hand panel, click **+ Add Data** and search for `saurabhshahane/fake-news-classification`.
3. Paste the contents of [`cloud/train_kaggle.py`](file:///Users/jayeshkakhani/Fake-News-Detection-System/cloud/train_kaggle.py) into the notebook cell.
4. Click **Run All**.
5. Once complete, download the generated artifacts from `/kaggle/working/`:
   - `fake_news_model.joblib`
   - `tfidf_vectorizer.joblib`
   - `evaluation.json`
6. Place the files into your local `models/` folder.

### Method 2: Kaggle CLI
1. Configure credentials at `~/.kaggle/kaggle.json`.
2. Edit [`cloud/kernel-metadata.json`](file:///Users/jayeshkakhani/Fake-News-Detection-System/cloud/kernel-metadata.json) with your Kaggle username.
3. Push and execute:
   ```bash
   cd cloud
   kaggle kernels push
   kaggle kernels status your_username/fake-news-detector-training
   ```
4. Download outputs directly into `models/`:
   ```bash
   kaggle kernels output your_username/fake-news-detector-training -p ../models/
   ```

---

## Automated Testing

Run the automated test suite without retraining or external network dependencies:
```bash
python3 -m pytest tests/test_prediction.py -v
```

The test suite validates:
- [x] Model artifact file presence and loading integrity
- [x] Preprocessing robustness (HTML tags, URLs, casing, whitespace, unicode)
- [x] Prediction function and complete response schema
- [x] Mathematical constraints ($0 \le P \le 1$ and $P_{real} + P_{fake} \approx 1.0$)
- [x] Safe error handling for empty or whitespace-only inputs
- [x] URL validation and security (rejection of non-HTTP schemes and localhost)
- [x] Observability logging and metrics aggregation

---

## Git & GitHub Policy

> [!IMPORTANT]
> **Strict Manual Version Control Policy**:
> - **GitHub is entirely optional** and is **never** synchronized automatically.
> - **No branch workflows**: No `main`, `development`, or `feature/*` branches are automatically created.
> - **Zero Automatic Remote Pushing**: No commits, pushes, pull requests, or automated CI/CD deployment pipelines are executed without explicit, prior user approval.
> - **Data & Secret Protection**: Large raw datasets (`WELFake*.csv`), Kaggle credentials (`kaggle.json`), virtual environments (`.venv/`), and environment files (`.env`) are strictly excluded in `.gitignore`.

---

## Ethical Disclaimers & Limitations

1. **Statistical Pattern Classification**: The predictions generated by this application reflect stylistic, vocabulary, and n-gram patterns learned from historical labeled news datasets (WELFake).
2. **Not Ground-Truth Fact Verification**: This application is **NOT** a factual proof engine. A real event described using sensationalist or exaggerated phrasing may trigger a "FAKE" pattern, while an inventive falsehood written in formal journalistic tone may score as "REAL".
3. **Journalistic Due Diligence**: Predictions must never be substituted for professional fact-checking, primary source review, or domain-expert editorial oversight.

---

## Future Roadmap

```
[Future News Verification System Architecture]
                     News Article
                          │
                          ▼
               Claim Extraction Engine
                          │
                          ▼
            Live News & Web Evidence Search
                          │
                          ▼
            Evidence Relevance & Stance Ranking
                          │
                          ▼
          Joint Statistical + Evidence Classifier
                          │
                          ▼
             Auditable Verification Report
```
The current version strictly focuses on explainable classical statistical classification. A future production iteration can introduce real-time web retrieval to cross-reference extracted claims against contemporary wire services.
