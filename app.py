from pathlib import Path
import html
import time

import keras
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


st.set_page_config(page_title="NeuroDetect", page_icon="🧠", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phase2_clean.keras"
README_PATH = BASE_DIR / "README.md"
LOGO_PATH = BASE_DIR / "logo.png"

CLASS_NAMES = (
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor",
)
VALID_VIEWS = {"landing", "docs", "faq", "portal"}
VALID_THEMES = {"light", "dark"}
IMAGE_SIZE = (256, 256)
MIN_CONFIDENCE = 80.0


def init_state() -> None:
    defaults = {
        "theme": "dark",
        "disclaimer_accepted": False,
        "view": "landing",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def sync_state_from_query_params() -> None:
    view = st.query_params.get("view")
    theme = st.query_params.get("theme")

    if view in VALID_VIEWS:
        st.session_state.view = view
    if theme in VALID_THEMES:
        st.session_state.theme = theme


def navigate(view: str) -> None:
    st.session_state.view = view
    st.rerun()


def set_theme(theme: str) -> None:
    st.session_state.theme = theme
    st.rerun()


def apply_style(theme: str) -> None:
    is_dark = theme == "dark"
    bg_color = "#0E1117" if is_dark else "#FFFFFF"
    text_color = "#FFFFFF" if is_dark else "#111111"
    text_muted = "#8B949E" if is_dark else "#5A6478"
    card_bg = "#1B212C" if is_dark else "#F0F2F6"
    border_color = "#2D3748" if is_dark else "#D1D5DB"
    step_num_bg = "#1A2535" if is_dark else "#E8F0FB"

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        .block-container {{ padding-top: 88px !important; max-width: 760px; }}
        header[data-testid="stHeader"] {{
            background: transparent !important;
            height: 0 !important;
            min-height: 0 !important;
            overflow: visible !important;
        }}
        #stDecoration {{ display: none !important; }}

        .nd-nav {{
            position: fixed;
            top: 14px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 99999;
            width: min(760px, calc(100vw - 32px));
            background-color: {card_bg};
            border: 1px solid {border_color};
            border-radius: 14px;
            padding: 8px 18px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.22);
            backdrop-filter: blur(14px);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }}
        .nd-nav-brand {{
            font-weight: 800;
            font-size: 1rem;
            color: {text_color};
            white-space: nowrap;
            text-decoration: none !important;
        }}
        .nd-nav-links,
        .nd-nav-right {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .nd-nav-links a,
        .nd-nav-portal {{
            font-size: 0.875rem;
            font-weight: 700;
            text-decoration: none !important;
            padding: 6px 12px;
            border-radius: 8px;
            white-space: nowrap;
        }}
        .nd-nav-links a {{
            color: {text_muted};
            transition: background 0.15s, color 0.15s;
        }}
        .nd-nav-links a:hover,
        .nd-nav-links a.active {{
            background: rgba(74,144,226,0.14);
            color: #4A90E2;
        }}
        .nd-nav-portal {{
            background: linear-gradient(90deg, #4A90E2 0%, #E83E8C 100%);
            color: white !important;
        }}
        .nd-nav-portal:hover {{ opacity: 0.88; }}
        .nd-nav-theme {{
            font-size: 1.1rem;
            text-decoration: none !important;
            padding: 4px 8px;
            border-radius: 8px;
            border: 1px solid {border_color};
            line-height: 1;
        }}
        .nd-nav-marker,
        .nd-nav-active-marker,
        .nd-nav-brand-marker,
        .nd-nav-portal-marker,
        .nd-nav-theme-marker {{
            display: none;
        }}
        div[data-testid="stHorizontalBlock"]:has(.nd-nav-marker) {{
            position: fixed;
            top: 14px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 99999;
            width: min(760px, calc(100vw - 32px));
            background-color: {card_bg};
            border: 1px solid {border_color};
            border-radius: 14px;
            padding: 8px 18px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.22);
            backdrop-filter: blur(14px);
            align-items: center;
            gap: 6px;
        }}
        div[data-testid="stHorizontalBlock"]:has(.nd-nav-marker) div[data-testid="column"] {{
            padding: 0 !important;
        }}
        div[data-testid="stHorizontalBlock"]:has(.nd-nav-marker) div.stButton > button {{
            min-height: 34px !important;
            height: 34px !important;
            padding: 6px 12px !important;
            border-radius: 8px !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            color: {text_muted} !important;
            font-size: 0.875rem !important;
            font-weight: 700 !important;
            white-space: nowrap !important;
            transition: background 0.15s, color 0.15s, opacity 0.15s !important;
        }}
        div[data-testid="stHorizontalBlock"]:has(.nd-nav-marker) div.stButton > button:hover {{
            background: rgba(74,144,226,0.14) !important;
            color: #4A90E2 !important;
            border-color: transparent !important;
        }}
        div[data-testid="column"]:has(.nd-nav-brand-marker) div.stButton > button {{
            color: {text_color} !important;
            font-size: 1rem !important;
            font-weight: 800 !important;
            justify-content: flex-start !important;
            padding-left: 0 !important;
        }}
        div[data-testid="column"]:has(.nd-nav-active-marker) div.stButton > button {{
            background: rgba(74,144,226,0.12) !important;
            color: #4A90E2 !important;
        }}
        div[data-testid="column"]:has(.nd-nav-portal-marker) div.stButton > button {{
            background: linear-gradient(90deg, #4A90E2 0%, #E83E8C 100%) !important;
            color: white !important;
        }}
        div[data-testid="column"]:has(.nd-nav-portal-marker) div.stButton > button:hover {{
            opacity: 0.88 !important;
        }}
        div[data-testid="column"]:has(.nd-nav-theme-marker) div.stButton > button {{
            border: 1px solid {border_color} !important;
            color: {text_color} !important;
            font-size: 1.05rem !important;
            padding: 4px 8px !important;
        }}

        div.stButton > button {{
            background: linear-gradient(90deg, #4A90E2 0%, #E83E8C 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            transition: 0.25s !important;
        }}
        div.stButton > button:hover {{ opacity: 0.88 !important; }}
        .cta-wrap div.stButton > button {{
            width: 320px !important;
            height: 56px !important;
            font-size: 1.1rem !important;
            letter-spacing: 0.03em !important;
        }}

        .nd-hero {{ text-align: center; padding: 48px 0 20px; }}
        .nd-hero h1 {{
            font-size: 3.4rem;
            font-weight: 800;
            color: {text_color};
            margin: 0 0 10px;
            line-height: 1.1;
        }}
        .nd-hero p {{
            font-size: 1.1rem;
            color: {text_muted};
            letter-spacing: 0.18em;
            margin: 0 0 32px;
        }}
        .nd-section-title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: {text_color};
            margin: 40px 0 16px;
            text-align: center;
        }}
        .nd-about,
        .nd-doc,
        .nd-step,
        .nd-result {{
            background: {card_bg};
            border: 1px solid {border_color};
            color: {text_color};
        }}
        .nd-about {{
            border-left: 4px solid #4A90E2;
            border-radius: 14px;
            padding: 28px 32px;
            margin-bottom: 12px;
        }}
        .nd-about p,
        .nd-doc p,
        .nd-step-desc,
        .nd-page-sub,
        .nd-tip,
        .nd-upload-hint {{
            color: {text_muted};
        }}
        .nd-about p {{ font-size: 1rem; line-height: 1.75; margin: 0; }}
        .nd-about-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {text_color};
            margin-bottom: 10px;
        }}
        .nd-steps {{
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-bottom: 12px;
        }}
        .nd-step {{
            border-radius: 12px;
            padding: 18px 22px;
            display: flex;
            align-items: flex-start;
            gap: 16px;
        }}
        .nd-step-num {{
            min-width: 36px;
            height: 36px;
            border-radius: 50%;
            background: {step_num_bg};
            border: 1px solid #4A90E2;
            color: #4A90E2;
            font-size: 0.95rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }}
        .nd-step-title {{
            font-size: 0.95rem;
            font-weight: 700;
            color: {text_color};
            margin-bottom: 3px;
        }}
        .nd-step-desc {{ font-size: 0.875rem; line-height: 1.5; }}
        .nd-disclaimer {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-left: 5px solid #4A90E2;
            border-radius: 12px;
            padding: 35px 40px;
            max-width: 620px;
            margin: 40px auto;
            color: {text_color};
        }}
        .nd-disclaimer-title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #4A90E2;
            margin-bottom: 18px;
        }}
        .nd-disclaimer p {{
            font-size: 1.05rem;
            color: {text_muted};
            line-height: 1.7;
            margin-bottom: 12px;
        }}
        .nd-upload-hint {{
            background: {card_bg};
            border: 1px dashed {border_color};
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            font-size: 0.9rem;
            margin-bottom: 8px;
        }}
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderFile"] {{
            background-color: {card_bg} !important;
            border-radius: 12px !important;
            border: 1px solid {border_color} !important;
        }}
        [data-testid="stFileUploader"] section *,
        [data-testid="stFileUploaderFile"] * {{ color: {text_color} !important; }}
        [data-testid="stFileUploader"] button,
        [data-testid="baseButton-secondary"] {{
            background: {card_bg} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px !important;
        }}
        [data-testid="stStatusWidget"],
        [data-testid="stExpander"],
        div[data-testid="stExpander"] > details,
        div[data-testid="stExpander"] > details > summary {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stStatusWidget"] *,
        [data-testid="stExpander"] * {{ color: {text_color} !important; }}
        .nd-result {{
            border-width: 2px;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            margin-top: 20px;
        }}
        .nd-result-label {{
            font-size: 0.85rem;
            font-weight: 700;
            color: #4A90E2;
            letter-spacing: 0.1em;
            margin-bottom: 6px;
        }}
        .nd-result-name {{
            font-size: 2.4rem;
            font-weight: 800;
            color: {text_color};
            margin: 0 0 18px;
        }}
        .nd-result-divider {{
            border: 0;
            border-top: 1px solid {border_color};
            margin: 18px 0;
        }}
        .nd-conf-label {{ font-size: 0.9rem; color: {text_muted}; margin-bottom: 6px; }}
        .nd-conf-bar {{
            background: {border_color};
            border-radius: 10px;
            height: 26px;
            overflow: hidden;
        }}
        .nd-conf-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4A90E2 0%, #63B3ED 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: bold;
            color: white;
            border-radius: 10px;
        }}
        .nd-tip {{
            background: {card_bg};
            border-left: 3px solid #4A90E2;
            border-radius: 6px;
            padding: 12px 16px;
            font-size: 0.85rem;
            margin-top: 16px;
        }}
        .nd-doc {{
            border-radius: 12px;
            padding: 36px 40px;
        }}
        .nd-doc h4 {{
            color: {text_color};
            font-size: 1rem;
            font-weight: 700;
            margin: 0 0 6px;
        }}
        .nd-doc p {{
            font-size: 0.925rem;
            line-height: 1.7;
            margin-bottom: 22px;
        }}
        @keyframes brainRotateFade {{
            0% {{ transform: translate(-50%,-50%) rotate(0deg) scale(0.5); opacity: 0; }}
            20% {{ transform: translate(-50%,-50%) rotate(72deg) scale(1.2); opacity: 1; }}
            80% {{ transform: translate(-50%,-50%) rotate(288deg) scale(1.2); opacity: 1; }}
            100% {{ transform: translate(-50%,-50%) rotate(360deg) scale(2); opacity: 0; }}
        }}
        .brain-overlay {{
            position: fixed;
            top: 50%;
            left: 50%;
            z-index: 9999;
            font-size: 200px;
            pointer-events: none;
            animation: brainRotateFade 2.5s ease-in-out forwards;
        }}
        .nd-page-title {{
            font-size: 1.9rem;
            font-weight: 800;
            color: {text_color};
            margin-bottom: 6px;
        }}
        .nd-page-sub {{
            font-size: 0.95rem;
            margin-bottom: 28px;
        }}
        @media (max-width: 640px) {{
            .block-container {{ padding-top: 120px !important; }}
            .nd-nav,
            div[data-testid="stHorizontalBlock"]:has(.nd-nav-marker) {{
                align-items: flex-start;
                flex-wrap: wrap;
                padding: 10px 12px;
            }}
            .nd-nav-links {{
                order: 3;
                width: 100%;
                justify-content: space-between;
            }}
            .nd-hero h1 {{ font-size: 2.4rem; }}
            .nd-result-name {{ font-size: 1.8rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_neuro_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception as exc:
        st.session_state.model_load_error = str(exc)
        return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def normalize_predictions(predictions: np.ndarray) -> np.ndarray:
    scores = np.asarray(predictions, dtype=np.float32).reshape(-1)
    if scores.size != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} model outputs, got {scores.size}."
        )
    if np.any(scores < 0) or not np.isclose(float(scores.sum()), 1.0, atol=1e-3):
        exp_scores = np.exp(scores - np.max(scores))
        scores = exp_scores / exp_scores.sum()
    return scores


def render_navbar() -> None:
    active = st.session_state.view
    next_theme = "light" if st.session_state.theme == "dark" else "dark"
    theme_label = "☀️" if st.session_state.theme == "dark" else "🌙"

    st.markdown(
        f"""
        <style>
        .st-key-nd_navbar {{
            position: relative;
            top: auto;
            left: auto;
            transform: none;
            z-index: 99999;
            width: 100%;
            background: transparent !important;
            border: none !important;
            border-radius: 14px;
            padding: 0 !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
            margin-bottom: 24px;
        }}
        .st-key-nd_navbar div[data-testid="column"] {{
            padding: 0 3px !important;
        }}
        .nd-nav-brand-text {{
            color: {"#FFFFFF" if st.session_state.theme == "dark" else "#111111"};
            font-size: 1.08rem;
            font-weight: 800;
            line-height: 34px;
            white-space: nowrap;
        }}
        .st-key-nd_navbar button {{
            min-height: 34px !important;
            height: 34px !important;
            background: transparent !important;
            color: {"#8B949E" if st.session_state.theme == "dark" else "#5A6478"} !important;
            border: none !important;
            border-radius: 8px !important;
            font-size: 0.875rem !important;
            font-weight: 700 !important;
            padding: 6px 12px !important;
            box-shadow: none !important;
            white-space: nowrap !important;
        }}
        .st-key-nd_navbar button:hover {{
            background: rgba(74,144,226,0.14) !important;
            color: #4A90E2 !important;
            opacity: 1 !important;
        }}
        .st-key-nav_home button,
        .st-key-nav_docs button,
        .st-key-nav_faq button {{
            background: transparent !important;
        }}
        .st-key-nav_{active} button {{
            background: rgba(74,144,226,0.12) !important;
            color: #4A90E2 !important;
        }}
        .st-key-nav_portal button {{
            background: linear-gradient(90deg, #4A90E2 0%, #E83E8C 100%) !important;
            color: white !important;
        }}
        .st-key-nav_portal button:hover {{
            opacity: 0.88 !important;
            color: white !important;
        }}
        .st-key-nav_theme button {{
            border: 1px solid {"#2D3748" if st.session_state.theme == "dark" else "#D1D5DB"} !important;
            color: {"#FFFFFF" if st.session_state.theme == "dark" else "#111111"} !important;
            padding: 4px 8px !important;
        }}
        @media (max-width: 640px) {{
            .st-key-nd_navbar {{
                width: calc(100vw - 24px);
            }}
            .nd-nav-brand-text {{
                font-size: 0.95rem;
            }}
            .st-key-nd_navbar button {{
                font-size: 0.78rem !important;
                padding: 5px 7px !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="nd_navbar"):
        col1, col2, col3, col4, col5, col6 = st.columns(
            [1.8, 0.8, 0.8, 0.8, 1.0, 0.5],
            vertical_alignment="center",
        )

        with col1:
            st.markdown('<div class="nd-nav-brand-text">🧠 NeuroDetect</div>', unsafe_allow_html=True)

        with col2:
            if st.button("Home", key="nav_landing", use_container_width=True):
                navigate("landing")

        with col3:
            if st.button("Docs", key="nav_docs", use_container_width=True):
                navigate("docs")

        with col4:
            if st.button("FAQ", key="nav_faq", use_container_width=True):
                navigate("faq")

        with col5:
            if st.button("Portal", key="nav_portal", use_container_width=True):
                navigate("portal")

        with col6:
            if st.button(theme_label, key="nav_theme", help="Toggle theme", use_container_width=True):
                set_theme(next_theme)


def render_disclaimer() -> None:
    st.markdown(
        """
        <h1 style="text-align:center;font-size:3rem;font-weight:800;margin-top:40px">NEURO DETECT</h1>
        <p style="text-align:center;font-size:1.1rem;color:#4A90E2;letter-spacing:0.18em;margin-bottom:0">
            INTELLIGENCE AMPLIFIED
        </p>
        <div class="nd-disclaimer">
            <div class="nd-disclaimer-title">⚠️ Acknowledgment</div>
            <p>This website is part of a <strong>University Senior Project</strong> and is
            intended for educational and research purposes only.</p>
            <p>It is not intended for clinical use and should not be used as a substitute
            for professional medical diagnosis, treatment, or radiological evaluation.</p>
            <p>By proceeding, you acknowledge that the results are for informational purposes
            only and should be reviewed by a qualified healthcare professional.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 1, 1])
    with center:
        if st.button("I Understand", use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.rerun()


def render_landing() -> None:
    st.markdown(
        """
        <div class="nd-hero">
            <h1>NEURO DETECT</h1>
            <p>INTELLIGENCE AMPLIFIED</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if LOGO_PATH.exists():
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.image(str(LOGO_PATH), use_container_width=True)

    st.markdown('<div class="cta-wrap">', unsafe_allow_html=True)
    if st.button("Enter Diagnostic Portal"):
        navigate("portal")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="nd-section-title">About NeuroDetect</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="nd-about">
            <div class="nd-about-title">What is NeuroDetect?</div>
            <p>
                NeuroDetect is a deep learning-powered diagnostic support tool developed as a
                University Senior Project. It uses a Convolutional Neural Network trained on
                labeled MRI slices to classify brain scans into four categories: Glioma,
                Meningioma, Pituitary Tumor, or No Tumor.<br><br>
                The system demonstrates the potential of computer vision in clinical decision
                support. It does not replace radiological expertise, and all results must be
                reviewed by a qualified healthcare professional.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nd-section-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="nd-steps">
            <div class="nd-step">
                <div class="nd-step-num">1</div>
                <div>
                    <div class="nd-step-title">Upload a brain MRI scan</div>
                    <div class="nd-step-desc">Use the Diagnostic Portal to upload a JPG or PNG MRI slice.</div>
                </div>
            </div>
            <div class="nd-step">
                <div class="nd-step-num">2</div>
                <div>
                    <div class="nd-step-title">AI analyzes the image</div>
                    <div class="nd-step-desc">The scan is resized, normalized, and classified across all four categories.</div>
                </div>
            </div>
            <div class="nd-step">
                <div class="nd-step-num">3</div>
                <div>
                    <div class="nd-step-title">Review the results</div>
                    <div class="nd-step-desc">You receive the primary prediction, confidence score, and probability breakdown.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_docs() -> None:
    st.markdown('<div class="nd-page-title">📄 Project Documentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="nd-page-sub">Technical details and project overview.</div>', unsafe_allow_html=True)
    st.markdown('<div class="nd-doc">', unsafe_allow_html=True)
    if README_PATH.exists():
        st.markdown(README_PATH.read_text(encoding="utf-8"))
    else:
        st.markdown("<p>No README.md found. Add one to your project root.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_faq() -> None:
    st.markdown('<div class="nd-page-title">❓ Frequently Asked Questions</div>', unsafe_allow_html=True)
    st.markdown('<div class="nd-page-sub">Everything you need to know about NeuroDetect.</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="nd-doc">
            <h4>What is NeuroDetect?</h4>
            <p>NeuroDetect is a deep learning-based diagnostic support tool designed to classify
            brain tumor MRI scans into Glioma, Meningioma, Pituitary Tumor, or No Tumor.</p>
            <h4>What is the purpose of this tool?</h4>
            <p>It demonstrates computer vision in healthcare for educational and research
            purposes. It is intended to assist, not replace, professional radiological evaluation.</p>
            <h4>What model architecture is used?</h4>
            <p>NeuroDetect uses a Convolutional Neural Network trained to recognize spatial
            patterns and textures in labeled MRI slices.</p>
            <h4>Is it 100% accurate?</h4>
            <p>No. This is a research project, and every finding must be reviewed by a specialist.</p>
            <h4>Is my scan data stored?</h4>
            <p>No. Uploaded scans are used only for temporary in-session processing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_offline() -> None:
    details = st.session_state.get("model_load_error")
    detail_html = ""
    if details:
        detail_html = f"<br><small>{html.escape(details)}</small>"
    st.markdown(
        f"""
        <div style="
            background-color:#2D0000;border-left:4px solid #E53E3E;
            border-radius:6px;padding:16px 20px;margin-bottom:20px;
            color:#FC8181;font-size:0.95rem;">
            ⚠️ <strong>Diagnostic system offline.</strong> The AI model could not be loaded.
            Please confirm the model exists at <code>{html.escape(str(MODEL_PATH))}</code>.
            {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_card(class_name: str, probability: float, is_top: bool) -> None:
    is_dark = st.session_state.theme == "dark"
    b_color = "#4A90E2" if is_top else ("#2D3748" if is_dark else "#D1D5DB")
    t_color = "#4A90E2" if is_top else ("#8B949E" if is_dark else "#5A6478")
    bg = ("#1A2535" if is_dark else "#E8F0FB") if is_top else ("#1B212C" if is_dark else "#F0F2F6")
    fw = "700" if is_top else "400"
    safe_name = html.escape(class_name)
    pct = probability * 100

    st.markdown(
        f"""
        <div style="background:{bg};border:2px solid {b_color};
            border-radius:10px;padding:18px 8px;text-align:center;">
            <div style="font-size:0.78rem;color:{t_color};font-weight:{fw};margin-bottom:6px;">
                {safe_name}
            </div>
            <div style="font-size:1.35rem;font-weight:700;color:{t_color};">
                {pct:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_portal() -> None:
    st.markdown('<div class="nd-page-title">🧠 Diagnostic Portal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nd-page-sub">Upload a brain MRI scan to receive an AI-assisted classification.</div>',
        unsafe_allow_html=True,
    )

    model = load_neuro_model()
    if model is None:
        render_model_offline()
        return

    uploaded_file = st.file_uploader(
        "Upload MRI scan (JPG / JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    if not uploaded_file:
        st.markdown(
            """
            <div class="nd-upload-hint">
                Drop a JPG or PNG brain MRI scan above to begin analysis.<br>
                <small>Scans are processed in-session only and never stored.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file could not be read as an image. Please upload a valid JPG or PNG scan.")
        return

    st.image(image, caption="Current patient scan", use_container_width=True)

    if not st.button("🧠 INITIATE NEURAL DIAGNOSTIC"):
        return

    try:
        with st.status("🧬 Analyzing neural patterns...", expanded=True) as status:
            st.write("Preparing scan...")
            img_array = preprocess_image(image)

            time.sleep(0.3)
            st.write("Extracting deep features...")
            raw_predictions = model.predict(img_array, verbose=0)
            scores = normalize_predictions(raw_predictions)

            time.sleep(0.3)
            st.write("Classifying pathology...")
            idx = int(np.argmax(scores))
            confidence = float(scores[idx]) * 100

            if confidence < MIN_CONFIDENCE:
                status.update(label="❌ Invalid scan", state="error", expanded=False)
                st.error("Invalid scan. Please upload a correct brain MRI scan.")
                return

            status.update(label="✅ Analysis complete", state="complete", expanded=False)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return

    brain_ph = st.empty()
    brain_ph.markdown('<div class="brain-overlay">🧠</div>', unsafe_allow_html=True)

    safe_prediction = html.escape(CLASS_NAMES[idx])
    st.markdown(
        f"""
        <div class="nd-result">
            <div class="nd-result-label">PRIMARY DIAGNOSIS</div>
            <div class="nd-result-name">{safe_prediction}</div>
            <hr class="nd-result-divider">
            <div class="nd-conf-label">Confidence Score</div>
            <div class="nd-conf-bar">
                <div class="nd-conf-fill" style="width:{confidence:.1f}%;">
                    {confidence:.1f}%
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>**Probability breakdown:**", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (class_name, probability) in enumerate(zip(CLASS_NAMES, scores)):
        with cols[i]:
            render_probability_card(class_name, float(probability), i == idx)

    st.markdown(
        """
        <div class="nd-tip">
            ⚕️ This result is for research and educational purposes only.
            Please consult a qualified radiologist before drawing any clinical conclusions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if CLASS_NAMES[idx] == "No Tumor":
        st.toast("✅ Analysis complete: No abnormalities detected.", icon="🧠")
    else:
        st.toast(f"⚠️ Potential {CLASS_NAMES[idx]} identified.", icon="❗")

    time.sleep(2.5)
    brain_ph.empty()


def render_footer() -> None:
    st.markdown(
        """
        <br><hr style="border-color:#2D3748;margin-top:40px;">
        <p style="text-align:center;color:#5A6478;font-size:0.82rem;padding-bottom:20px;">
            Senior Project 2026 &nbsp;|&nbsp; NeuroDetect AI &nbsp;|&nbsp; By fatima &amp; Yusra
        </p>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    sync_state_from_query_params()
    apply_style(st.session_state.theme)

    if not st.session_state.disclaimer_accepted:
        render_disclaimer()
        st.stop()

    render_navbar()

    pages = {
        "landing": render_landing,
        "docs": render_docs,
        "faq": render_faq,
        "portal": render_portal,
    }
    pages.get(st.session_state.view, render_landing)()
    render_footer()


if __name__ == "__main__":
    main()
