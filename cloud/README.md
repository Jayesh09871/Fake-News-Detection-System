# Remote & Cloud Training Guide

This directory contains the training pipelines for the **Fake News Detection System**.

You can train the model using **Google Colab** (recommended for quick execution & automatic downloads), locally with your own dataset placed in the **`data/`** folder, or via **Kaggle Cloud**.

---

## Dataset Reference

- **Name**: WELFake Dataset (Word Embedding over Linear Fake News Classification)
- **Source**: [Kaggle WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- **Total Articles**: ~72,134 news records
- **Schema**:
  - `title`: News headline
  - `text`: Article body content
  - `label`: `0` = **FAKE**, `1` = **REAL**

---

## Method 1: Google Colab Training (Recommended)

Google Colab provides free cloud compute (CPU/GPU) with fast training and one-click artifact downloading.

### Steps:
1. Open **[Google Colab](https://colab.research.google.com)** in your browser.
2. Click **File** > **Upload Notebook** and select:
   ```
   cloud/Fake_News_Training_Colab.ipynb
   ```
3. Upload `WELFake_Dataset.csv` when prompted by the upload cell (or drag and drop it into the `/content/data/` folder on the left panel).
4. Run all cells:
   - Cleans the text, deduplicates rows, and splits 80% train / 10% validation / 10% test.
   - Extracts 100,000 TF-IDF features and fits the balanced Logistic Regression model.
   - Evaluates on the held-out test split and generates the confusion matrix.
5. The final cell in the notebook will **automatically download** the 4 production artifacts to your computer:
   - `fake_news_model.joblib`
   - `tfidf_vectorizer.joblib`
   - `evaluation.json`
   - `confusion_matrix.png`
6. Move these 4 files into your local **`Fake-News-Detection-System/models/`** directory!

---

## Method 2: Training with Local `data/` Folder

If you have downloaded `WELFake_Dataset.csv` onto your computer and want to train using the script:

1. Place `WELFake_Dataset.csv` into the **`data/`** directory:
   ```
   Fake-News-Detection-System/
   └── data/
       └── WELFake_Dataset.csv
   ```
   *(Note: The `data/` directory is git-ignored, so the dataset will never be committed to Git).*

2. Run the training script:
   ```bash
   python cloud/train_model.py --data-path data/WELFake_Dataset.csv --output-dir models
   ```

3. The script will automatically clean, split, train, evaluate, and write the model artifacts directly to `models/`!

---

## Method 3: Kaggle Cloud Training

If you prefer Kaggle:

1. Go to [kaggle.com](https://www.kaggle.com) and click **Create** > **New Notebook**.
2. Click **+ Add Data** on the right panel and search for `saurabhshahane/fake-news-classification`.
3. Paste the contents of [`cloud/train_model.py`](file:///Users/jayeshkakhani/Fake-News-Detection-System/cloud/train_model.py) into the notebook cell.
4. Click **Run All**.
5. Download `fake_news_model.joblib`, `tfidf_vectorizer.joblib`, and `evaluation.json` from `/kaggle/working/` and place them into your local `models/` directory.

---

## What Happens After Artifacts Are in `models/`

Once `models/fake_news_model.joblib`, `models/tfidf_vectorizer.joblib`, and `models/evaluation.json` are placed in the `models/` directory:

1. Launch Streamlit:
   ```bash
   python3 run_app.py
   ```
2. The application will instantly load the full 100,000-feature model trained on the complete 72,000-article dataset.
3. The **Model Performance** page will display the full benchmark accuracy (~96.1%), precision, recall, and confusion matrix.
4. Both real and fake news articles from any source or URL will be differentiated with high accuracy and calibrated confidence scores!
