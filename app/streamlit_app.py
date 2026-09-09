"""
Fake News Detection System — Streamlit Web Application
Production-grade news credibility classifier using classical NLP (TF-IDF + Logistic Regression).
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.article_extractor import extract_article_from_url, validate_url
from src.observability import (
    get_analytics_metrics,
    initialize_log_file,
    load_prediction_logs,
    log_prediction_event,
)
from src.predict import (
    DEFAULT_MODEL_PATH,
    DEFAULT_VECTORIZER_PATH,
    ModelNotFoundError,
    NewsPredictor,
)

# Set page configuration
st.set_page_config(
    page_title="AI Fake News Detector & News Verifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, internship-grade visual styling
st.markdown(
    """
    <style>
        /* Base styling */
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        /* Metric card containers */
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.2rem;
            text-align: center;
        }
        .result-box-real {
            background: linear-gradient(135deg, #065f46 0%, #047857 100%);
            color: #ffffff;
            border-radius: 12px;
            padding: 1.8rem;
            text-align: center;
            box-shadow: 0 4px 14px rgba(5, 150, 105, 0.25);
            margin-bottom: 1.5rem;
        }
        .result-box-fake {
            background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%);
            color: #ffffff;
            border-radius: 12px;
            padding: 1.8rem;
            text-align: center;
            box-shadow: 0 4px 14px rgba(220, 38, 38, 0.25);
            margin-bottom: 1.5rem;
        }
        .badge-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            opacity: 0.85;
            margin-bottom: 0.2rem;
        }
        .badge-verdict {
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            margin: 0;
            line-height: 1.1;
        }
        .badge-confidence {
            font-size: 1.25rem;
            font-weight: 600;
            margin-top: 0.5rem;
            opacity: 0.95;
        }
        .disclaimer-card {
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 1rem 1.2rem;
            border-radius: 6px;
            color: #92400e;
            font-size: 0.88rem;
            line-height: 1.45;
            margin-top: 1.5rem;
        }
        .stat-label {
            font-size: 0.82rem;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
        }
        .stat-val {
            font-size: 1.3rem;
            font-weight: 700;
            color: #0f172a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_predictor() -> NewsPredictor:
    """Initialize predictor once per Streamlit session."""
    return NewsPredictor.get_instance(
        vectorizer_path=DEFAULT_VECTORIZER_PATH,
        model_path=DEFAULT_MODEL_PATH,
    )


def check_model_availability() -> bool:
    """Check if model joblib files exist locally."""
    predictor = get_cached_predictor()
    return predictor.are_artifacts_present()


def render_sidebar():
    """Render sidebar navigation and project metadata."""
    st.sidebar.image(
        "https://raw.githubusercontent.com/feathericons/feather/master/icons/shield.svg",
        width=48,
    )
    st.sidebar.title("News Verification")
    st.sidebar.markdown(
        "**AI / NLP Fake News Detection**  \n"
        "*Internship Portfolio Project*"
    )
    st.sidebar.markdown("---")

    selected_page = st.sidebar.radio(
        "Navigation",
        [
            "📰 News Detector",
            "📊 Model Performance",
            "📈 Analytics & Observability",
            "ℹ️ About & Architecture",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")

    models_ready = check_model_availability()
    if models_ready:
        st.sidebar.success("✅ Model Artifacts Loaded")
        st.sidebar.caption("TF-IDF (100k) + Logistic Regression")
    else:
        st.sidebar.warning("⚠️ Model Artifacts Missing")
        st.sidebar.caption("Trained artifacts need to be downloaded from Kaggle.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Remote Cloud Training: **Kaggle**")
    st.sidebar.caption("Dataset: **WELFake (72k articles)**")
    st.sidebar.caption("Environment: **Local Streamlit**")

    return selected_page


# Curated multi-sample news datasets for interactive testing across diverse domains
REAL_NEWS_SAMPLES = [
    {
        "title": "Federal Reserve Holds Interest Rates Steady Amid Cooling Inflation",
        "body": (
            "WASHINGTON (Reuters) - The Federal Reserve decided on Wednesday to maintain its benchmark interest rate at the current target range, "
            "citing consistent progress toward its long-term inflation goal of two percent, officials said on Wednesday. Officials highlighted sustained economic growth "
            "and strong labor market conditions while emphasizing that monetary policy will remain data-dependent over the coming quarters."
        ),
    },
    {
        "title": "NASA Space Telescope Discovers Atmospheric Water Vapor on Rocky Exoplanet",
        "body": (
            "WASHINGTON (Reuters) - Astronomers utilizing data from deep space spectroscopy observations have identified distinct spectral signatures indicating "
            "the presence of atmospheric water vapor on a rocky exoplanet orbiting a stable red dwarf star, researchers said on Thursday. The peer-reviewed findings, "
            "published in the journal Nature Astronomy, provide critical empirical insights into how stellar winds influence planetary atmospheres."
        ),
    },
    {
        "title": "Global Health Authorities Report Significant Multi-Year Decline in Tuberculosis Transmission",
        "body": (
            "GENEVA (Reuters) - International public health authorities reported an encouraging eight percent decrease in global tuberculosis transmission rates "
            "following a coordinated multi-year immunization and diagnostic rollout, the World Health Organization said on Friday. The humanitarian initiative, supported by healthcare partner "
            "clinics across forty nations, expands access to rapid molecular testing and early clinical intervention protocols."
        ),
    },
    {
        "title": "Solar and Wind Power Capacity Overtakes Fossil Fuel Generation Across Major Regional Grids",
        "body": (
            "LONDON (Reuters) - Electricity generated from combined utility-scale solar and wind installations exceeded traditional coal and natural gas outputs "
            "for three consecutive months, independent energy grid analytics showed on Tuesday. Grid transmission operators attributed "
            "the historic milestone to modern grid-scale battery storage facilities and expanded high-voltage interconnector lines."
        ),
    },
    {
        "title": "Maritime Nations Conclude Historic International Treaty to Safeguard Deep Ocean Coral Ecosystems",
        "body": (
            "SYDNEY (Reuters) - Diplomatic delegates representing fourteen coastal nations concluded bilateral negotiations on Friday, ratifying a comprehensive "
            "conservation agreement that establishes protected marine sanctuaries across three million square kilometers of ocean, officials said on Friday. "
            "The treaty mandates strict commercial fishing quotas and strictly prohibits seabed mineral dredging in biodiversity hotspots."
        ),
    },
]

FAKE_NEWS_SAMPLES = [
    {
        "title": "Secret Underground Base Discovered Operating Illegal Climate Control Machine",
        "body": (
            "Shocking whistleblower documents reveal that an elite cabal of rogue billionaires has constructed a massive subterranean facility "
            "capable of triggering hurricanes and earthquakes at will. Mainstream scientists have been silenced by government agents who "
            "are confiscating private weather telemetry data to keep the global population completely in the dark."
        ),
    },
    {
        "title": "Miracle Himalayan Berry Cures All Chronic Diseases Overnight, Doctors Outraged",
        "body": (
            "A mysterious ancient berry harvested from forbidden Himalayan valleys has been proven to eradicate diabetes, heart disease, "
            "and joint pain within twelve hours of consumption. Greedy pharmaceutical executives and mainstream hospitals are desperately "
            "attempting to ban this sacred plant from consumer markets to protect their multi-billion dollar chemotherapy monopolies."
        ),
    },
    {
        "title": "Declassified Leaks Expose Synthetic Alien Clones Replacing Top Diplomatic Leaders",
        "body": (
            "High-ranking military insiders have leaked top secret footage showing world leaders undergoing biological replacement in deep "
            "underground bunkers. The synthetic entities possess artificial neural networks designed to usher in a unified extraterrestrial "
            "surveillance grid before the year ends, according to uncensored telegram broadcasts."
        ),
    },
    {
        "title": "Archaeologists Unearth Working iPhone Inside 3,000-Year-Old Egyptian Tomb",
        "body": (
            "Excavators in Luxor made a mind-boggling discovery yesterday after uncovering an authentic modern smartphone sealed inside a "
            "3,000-year-old sarcophagus. Laboratory radiocarbon testing confirmed the device has been entombed since 1200 BC, providing "
            "undeniable proof that ancient Egyptian priests mastered quantum time travel technology."
        ),
    },
    {
        "title": "Smart Streetlights Emitting Secret Frequencies to Reprogram Human Brainwaves at Night",
        "body": (
            "An anonymous whistleblower from a telecommunications contractor has released internal schematics showing newly installed LED "
            "streetlights are broadcasting psychoacoustic microwave frequencies while citizens sleep. The covert signals allegedly suppress "
            "independent cognitive thinking and induce docile compliance in urban neighborhoods."
        ),
    },
]


def render_detector_page():
    """Main News Detector View."""
    st.markdown('<div class="main-header">AI Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">NLP-based news classification using TF-IDF feature extraction and Logistic Regression</div>',
        unsafe_allow_html=True,
    )

    models_ready = check_model_availability()

    if not models_ready:
        st.error(
            "⚠️ **Trained model artifacts are missing from the `models/` directory.**\n\n"
            "To keep local compute lightweight, model training is performed remotely on **Kaggle Cloud** using the **WELFake dataset**.\n\n"
            "**Next Steps:**\n"
            "1. Run the remote training script: [`cloud/train_kaggle.py`](file:///Users/jayeshkakhani/Fake-News-Detection-System/cloud/train_kaggle.py) on Kaggle.\n"
            "2. Download `tfidf_vectorizer.joblib`, `fake_news_model.joblib`, and `evaluation.json` from the Kaggle output.\n"
            "3. Place them into the local `models/` directory."
        )
        with st.expander("📖 View Quick Cloud Training Instructions"):
            st.markdown(
                """
                - **Option A (Web)**: Create a new notebook on [Kaggle](https://kaggle.com), attach `saurabhshahane/fake-news-classification`, copy `cloud/train_kaggle.py`, and run.
                - **Option B (CLI)**: Configure Kaggle API and run:
                  ```bash
                  cd cloud && kaggle kernels push
                  kaggle kernels output your_username/fake-news-detector-training -p ../models/
                  ```
                See [`cloud/README.md`](file:///Users/jayeshkakhani/Fake-News-Detection-System/cloud/README.md) for full details.
                """
            )
        return

    # Initialize session state keys for inputs if not present
    if "input_title_text" not in st.session_state:
        st.session_state["input_title_text"] = ""
    if "input_body_text" not in st.session_state:
        st.session_state["input_body_text"] = ""
    if "input_url" not in st.session_state:
        st.session_state["input_url"] = ""
    if "real_sample_idx" not in st.session_state:
        st.session_state["real_sample_idx"] = 0
    if "fake_sample_idx" not in st.session_state:
        st.session_state["fake_sample_idx"] = 0

    def load_real_sample_callback():
        idx = st.session_state.get("real_sample_idx", 0)
        sample = REAL_NEWS_SAMPLES[idx % len(REAL_NEWS_SAMPLES)]
        st.session_state["real_sample_idx"] = idx + 1
        st.session_state["input_title_text"] = sample["title"]
        st.session_state["input_body_text"] = sample["body"]

    def load_fake_sample_callback():
        idx = st.session_state.get("fake_sample_idx", 0)
        sample = FAKE_NEWS_SAMPLES[idx % len(FAKE_NEWS_SAMPLES)]
        st.session_state["fake_sample_idx"] = idx + 1
        st.session_state["input_title_text"] = sample["title"]
        st.session_state["input_body_text"] = sample["body"]

    def clear_inputs_callback():
        st.session_state["input_title_text"] = ""
        st.session_state["input_body_text"] = ""
        st.session_state["input_url"] = ""

    # Choose Input Method
    input_mode = st.radio(
        "Choose Input Method",
        ["📝 Paste News Article", "🌐 Extract from Article URL"],
        horizontal=True,
    )

    article_title = ""
    article_body = ""
    input_source = "text"

    if input_mode == "📝 Paste News Article":
        st.markdown("##### Direct Text Input")
        col_sample1, col_sample2, col_spacer = st.columns([1.5, 1.5, 3])
        with col_sample1:
            st.button(
                "🟢 Load Real Sample",
                on_click=load_real_sample_callback,
                use_container_width=True,
                help="Click to load or cycle through different real news stories across economics, space, medicine, energy, and conservation.",
            )
        with col_sample2:
            st.button(
                "🔴 Load Fake Sample",
                on_click=load_fake_sample_callback,
                use_container_width=True,
                help="Click to load or cycle through different fake news stories across conspiracies, miracle cures, hoaxes, and tech myths.",
            )

        st.text_input(
            "Article Headline / Title (Optional)",
            placeholder="e.g., NASA announces groundbreaking findings on Mars...",
            key="input_title_text",
        )
        st.text_area(
            "Article Body Content (Required)",
            placeholder="Paste the full text or substantial excerpt of the news article here...",
            height=220,
            key="input_body_text",
        )

        article_title = st.session_state.get("input_title_text", "")
        article_body = st.session_state.get("input_body_text", "")
        input_source = "text"

    else:
        st.markdown("##### Web Article URL Extraction")
        st.text_input(
            "News Article URL",
            placeholder="https://www.reuters.com/world/... or https://www.bbc.com/news/...",
            key="input_url",
        )

        if st.button("Fetch & Extract Article", key="btn_fetch_url"):
            current_url = st.session_state.get("input_url", "").strip()
            if not current_url:
                st.warning("Please enter a valid HTTP/HTTPS URL.")
            else:
                with st.spinner("Fetching and extracting article with trafilatura..."):
                    result = extract_article_from_url(current_url)
                    if result["success"]:
                        st.session_state["input_title_text"] = result["title"]
                        st.session_state["input_body_text"] = result["text"]
                        st.success("Article successfully extracted and loaded below!")
                    else:
                        st.error(result["error"])

        st.text_input(
            "Extracted Headline / Title",
            key="input_title_text",
        )
        st.text_area(
            "Extracted Article Body Content",
            height=220,
            key="input_body_text",
        )

        article_title = st.session_state.get("input_title_text", "")
        article_body = st.session_state.get("input_body_text", "")
        input_source = "url"

    # Action buttons
    st.markdown("---")
    col_action1, col_action2, col_space = st.columns([1.5, 1, 3])
    with col_action1:
        analyze_clicked = st.button("🔍 Analyze News Article", type="primary", use_container_width=True)
    with col_action2:
        st.button("🗑️ Clear", on_click=clear_inputs_callback, use_container_width=True)

    if analyze_clicked:
        if not article_body.strip() and not article_title.strip():
            st.warning("Please provide an article body or headline before analyzing.")
            return

        # Perform analysis
        predictor = get_cached_predictor()
        with st.spinner("Analyzing linguistic patterns with TF-IDF & Logistic Regression..."):
            try:
                res = predictor.predict(text=article_body, title=article_title)
            except Exception as e:
                st.error(f"Inference error: {e}")
                return

        if res.get("error"):
            st.warning(res["error"])
            return

        # Record observability event
        log_prediction_event(
            prediction=res["label"],
            confidence=res["confidence"],
            real_probability=res["real_probability"],
            fake_probability=res["fake_probability"],
            word_count=res["word_count"],
            char_count=res["char_count"],
            processing_time=res["processing_time"],
            input_source=input_source,
        )

        st.markdown("### Analysis Results")

        # Result badge
        is_real = res["label"] == "REAL"
        box_class = "result-box-real" if is_real else "result-box-fake"
        verdict_icon = "✅" if is_real else "⚠️"
        verdict_label = "REAL NEWS PATTERN" if is_real else "FAKE NEWS PATTERN"

        st.markdown(
            f"""
            <div class="{box_class}">
                <div class="badge-title">Predicted Classification</div>
                <div class="badge-verdict">{verdict_icon} {res['label']}</div>
                <div class="badge-confidence">Model Confidence: {res['confidence'] * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probabilities & metrics breakdown
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric(
                label="Real Class Probability",
                value=f"{res['real_probability'] * 100:.1f}%",
            )
        with col_m2:
            st.metric(
                label="Fake Class Probability",
                value=f"{res['fake_probability'] * 100:.1f}%",
            )
        with col_m3:
            st.metric(
                label="Article Length",
                value=f"{res['word_count']:,} words",
                help=f"Total characters: {res['char_count']:,}",
            )
        with col_m4:
            st.metric(
                label="Processing Latency",
                value=f"{res['processing_time'] * 1000:.1f} ms",
                help=f"Raw execution time: {res['processing_time']:.4f} seconds",
            )

        # Visual probability comparison bar
        st.markdown("##### Class Probability Distribution")
        real_pct = int(res["real_probability"] * 100)
        st.progress(real_pct, text=f"REAL: {res['real_probability']*100:.1f}%  |  FAKE: {res['fake_probability']*100:.1f}%")

        # Mandatory disclaimer
        st.markdown(
            """
            <div class="disclaimer-card">
                <strong>⚠️ Research & Verification Disclaimer:</strong><br>
                This system predicts whether the submitted article resembles real or fake news patterns learned from historical labeled data (the WELFake benchmark).
                It is a machine learning text classification model and is <strong>NOT</strong> an absolute proof or factual fact-checking verification system.
                Always cross-reference critical claims with primary, reputable journalistic and governmental sources.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_performance_page():
    """Model Performance View."""
    st.markdown('<div class="main-header">Model Performance & Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Independent test set metrics evaluated on unseen articles from the WELFake dataset (80/10/10 split)</div>',
        unsafe_allow_html=True,
    )

    eval_file = DEFAULT_MODEL_PATH.parent / "evaluation.json"

    if not eval_file.is_file():
        st.info(
            "Evaluation metrics will appear here once cloud training on Kaggle has been executed.\n\n"
            "The training script `cloud/train_kaggle.py` automatically generates `evaluation.json` "
            "evaluating accuracy, precision, recall, F1, and the confusion matrix on the held-out 10% test split."
        )
        return

    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"Failed to read evaluation file: {e}")
        return

    metrics = data.get("metrics", {})
    real_metrics = metrics.get("real_class", {})
    fake_metrics = metrics.get("fake_class", {})
    cm_data = data.get("confusion_matrix", {})

    # Top level metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
    with col2:
        st.metric("Macro F1-Score", f"{metrics.get('macro_f1', 0):.4f}")
    with col3:
        st.metric("Macro Precision", f"{metrics.get('macro_precision', 0):.4f}")
    with col4:
        st.metric("Macro Recall", f"{metrics.get('macro_recall', 0):.4f}")

    st.markdown("---")

    # Per-class breakdown
    col_cls1, col_cls2 = st.columns(2)
    with col_cls1:
        st.markdown("#### REAL News Class Performance (1)")
        df_real = pd.DataFrame(
            [
                {"Metric": "Precision", "Score": f"{real_metrics.get('precision', 0) * 100:.2f}%"},
                {"Metric": "Recall", "Score": f"{real_metrics.get('recall', 0) * 100:.2f}%"},
                {"Metric": "F1-Score", "Score": f"{real_metrics.get('f1_score', 0):.4f}"},
                {"Metric": "Test Support", "Score": f"{real_metrics.get('support', 0):,} samples"},
            ]
        )
        st.dataframe(df_real, use_container_width=True, hide_index=True)

    with col_cls2:
        st.markdown("#### FAKE News Class Performance (0)")
        df_fake = pd.DataFrame(
            [
                {"Metric": "Precision", "Score": f"{fake_metrics.get('precision', 0) * 100:.2f}%"},
                {"Metric": "Recall", "Score": f"{fake_metrics.get('recall', 0) * 100:.2f}%"},
                {"Metric": "F1-Score", "Score": f"{fake_metrics.get('f1_score', 0):.4f}"},
                {"Metric": "Test Support", "Score": f"{fake_metrics.get('support', 0):,} samples"},
            ]
        )
        st.dataframe(df_fake, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Confusion Matrix Display
    st.markdown("#### Confusion Matrix on Test Split")
    matrix = cm_data.get("matrix", [[0, 0], [0, 0]])
    cm_df = pd.DataFrame(
        matrix,
        index=["Actual FAKE (0)", "Actual REAL (1)"],
        columns=["Predicted FAKE", "Predicted REAL"],
    )
    st.dataframe(cm_df, use_container_width=True)

    col_raw1, col_raw2 = st.columns(2)
    with col_raw1:
        st.write(f"- **True Negatives (Correct FAKE):** `{cm_data.get('true_negatives', 0):,}`")
        st.write(f"- **False Positives (Fake marked Real):** `{cm_data.get('false_positives', 0):,}`")
    with col_raw2:
        st.write(f"- **False Negatives (Real marked Fake):** `{cm_data.get('false_negatives', 0):,}`")
        st.write(f"- **True Positives (Correct REAL):** `{cm_data.get('true_positives', 0):,}`")

    # Display saved image if available
    cm_img = DEFAULT_MODEL_PATH.parent / "confusion_matrix.png"
    if cm_img.is_file():
        st.image(str(cm_img), caption="Test Set Confusion Matrix", use_container_width=False, width=480)

    st.caption(f"Evaluated At: {data.get('evaluated_at', 'N/A')} | Benchmark: {data.get('dataset', 'WELFake')}")


def render_analytics_page():
    """Analytics and Observability View."""
    st.markdown('<div class="main-header">Application Observability & Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Local inference tracking: latency, confidence distributions, and session volume</div>',
        unsafe_allow_html=True,
    )

    metrics = get_analytics_metrics()
    total = metrics["total_predictions"]

    if total == 0:
        st.info("No inference events recorded yet. Analyze an article on the **News Detector** page to generate logs.")
        return

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Inferences", f"{total:,}")
    with col2:
        st.metric("Avg Confidence", f"{metrics['avg_confidence'] * 100:.1f}%")
    with col3:
        st.metric("Avg Processing Time", f"{metrics['avg_latency'] * 1000:.1f} ms")
    with col4:
        st.metric("Avg Word Count", f"{metrics['avg_word_count']:,} words")

    st.markdown("---")

    # Distribution breakdown
    col_chart, col_stats = st.columns([2, 1])
    with col_chart:
        st.markdown("##### Prediction Class Distribution")
        dist_df = pd.DataFrame(
            {
                "Class": ["REAL", "FAKE"],
                "Count": [metrics["real_count"], metrics["fake_count"]],
            }
        ).set_index("Class")
        st.bar_chart(dist_df)

    with col_stats:
        st.markdown("##### Summary Statistics")
        st.write(f"- **REAL Predictions:** `{metrics['real_count']}` ({metrics['real_pct']}%)")
        st.write(f"- **FAKE Predictions:** `{metrics['fake_count']}` ({metrics['fake_pct']}%)")
        st.write(f"- **Total Monitored Inferences:** `{total}`")
        st.caption("Logs stored strictly in `logs/prediction_logs.csv` without private text data.")

    st.markdown("---")
    st.markdown("##### Recent Inference Events")
    recent_df = metrics["recent_logs"]
    if not recent_df.empty:
        display_df = recent_df[
            [
                "timestamp",
                "prediction",
                "confidence",
                "real_probability",
                "fake_probability",
                "word_count",
                "processing_time",
                "input_source",
            ]
        ].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_about_page():
    """About & Architecture View."""
    st.markdown('<div class="main-header">About the Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Internship-grade NLP architecture for news classification</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 1. Problem Statement & Mission
        The proliferation of misinformation and digitally manipulated news creates urgent challenges for media literacy and digital trust.
        This project demonstrates a production-grade, explainable, classical NLP pipeline capable of analyzing article text structure,
        linguistic patterns, and vocabulary distributions to classify content as resembling **REAL** or **FAKE** news.

        ---

        ### 2. End-to-End System Architecture

        ```
        [Client Layer]
              │
              ├── Text Input (Headline + Article Body)
              └── URL Input (Web scraping via trafilatura)
                    │
                    ▼
        [Deterministic Preprocessing]
              │
              ├── HTML Entity Decoding & Tag Removal
              ├── URL & Hyperlink Stripping
              ├── Lowercase Normalization & Special Character Sanitization
              └── Whitespace Collapsing
                    │
                    ▼
        [NLP Vectorization Layer]
              │
              └── TF-IDF (100,000 features, Unigram + Bigram, Sublinear TF)
                    │
                    ▼
        [Classification Layer]
              │
              └── Logistic Regression (Balanced class weights, C=2.0)
                    │
                    ▼
        [Inference & Observability Output]
              │
              ├── Prediction: REAL or FAKE
              ├── Model Confidence %
              ├── Real vs Fake Probability Breakdown
              └── Local Observability Logger (logs/prediction_logs.csv)
        ```

        ---

        ### 3. Remote Cloud Training vs. Local Inference Separation
        To ensure reproducible, lightweight local engineering without overloading personal hardware:
        - **Training Environment**: Kaggle Cloud (CPU/GPU) with direct access to the 150MB+ **WELFake dataset** (~72k articles).
        - **Local Environment**: Antigravity IDE / VS Code running only inference with lightweight, trained `.joblib` model artifacts.
        - **No Local Dataset Training**: The WELFake raw dataset is never pulled onto the local development machine.

        ---

        ### 4. Technical Specifications
        - **Algorithm**: TF-IDF (Term Frequency-Inverse Document Frequency) + L2 Regularized Logistic Regression.
        - **Hyperparameters**: `max_features=100000`, `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, `C=2.0`, `class_weight='balanced'`.
        - **Data Split**: 80% Train, 10% Validation, 10% Test (Stratified based on label).
        - **Libraries**: `scikit-learn`, `pandas`, `numpy`, `joblib`, `trafilatura`, `streamlit`.

        ---

        ### 5. Ethical Considerations & Limitations
        > **Critical Disclaimer**: This system evaluates whether an article's vocabulary and phrasing resemble patterns in historical labeled datasets.
        > It does **not** perform live knowledge-base verification, fact-checking against real-time truth databases, or source credibility auditing.
        """
    )


def main():
    selected_page = render_sidebar()

    if selected_page == "📰 News Detector":
        render_detector_page()
    elif selected_page == "📊 Model Performance":
        render_performance_page()
    elif selected_page == "📈 Analytics & Observability":
        render_analytics_page()
    elif selected_page == "ℹ️ About & Architecture":
        render_about_page()


if __name__ == "__main__":
    main()
