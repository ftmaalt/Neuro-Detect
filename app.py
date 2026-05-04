import sys

import streamlit as st
import tensorflow as tf
from PIL import Image
from pathlib import Path
import numpy as np
import os
import base64
import time

# configuring page and initializing theme
st.set_page_config(page_title="NeuroDetect", page_icon="🧠", layout="centered")

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False
# css

def apply_style(theme):
    bg_color = "#0E1117" if theme == 'dark' else "#FFFFFF"
    text_color = "#FFFFFF" if theme == 'dark' else "#000000"
    card_bg = "#1B212C" if theme == 'dark' else "#F0F2F6"
    border_color = "#2D3748" if theme == 'dark' else "#D1D5DB"
    
    
    st.markdown(f"""
        <style>
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        st.html{{
    scroll-behavior: smooth;
    }}
        /* Global Button */
        div.stButton > button {{
            background: linear-gradient(90deg, #4A90E2 0%, #E83E8C 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            transition: 0.3s !important;
        }}

        /* Navigation Buttons (home page) */
        [data-testid="column"] div.stButton > button {{
            padding: 0.5rem 1rem !important;
            font-size: 0.85rem !important;
            width: 100% !important; /* Changed to fill column width */
            min-width: 220px !important; /* Increased to keep letters on one line */
            white-space: nowrap !important;
        }}

        /* Landing Page Center Button */
        .main-btn-container {{
            display: flex;
            justify-content: center;
            padding: 20px 0;
        }}
        
        .main-btn-container div.stButton > button {{
            width: 350px !important;
            height: 60px !important;
            font-size: 1.2rem !important;
        }}
/* Style the form container to match doc-container */
        [data-testid="stForm"] {{
            background-color: {card_bg} !important;
            padding: 30px !important;
            border-radius: 12px !important;
            border: 1px solid {border_color} !important;
        }}

        /* Upload container Fix */
        [data-testid="stFileUploader"] section {{
            background-color: {card_bg} !important;
            border-radius: 12px !important;
            border: 1px solid {border_color} !important;
        }}

        /* Upload button Fix and text*/
        [data-testid="stFileUploader"] section * {{
            color: {text_color} !important;
        }}
            [data-testid="stFileUploader"] button {{
            background-color: #111827 !important;
            color: white !important;
            border-radius: 8px !important;
            
        }}
        [data-testid="stFileUploader"] button * {{
        color: white !important;
    }}
        

        /* white background on the form submit button */
        [data-testid="stForm"] button {{
            background: linear-gradient(90deg, #4A90E2 0%, #E83E8C 100%) !important;
            color: white !important;
            border: none !important;
            width: 100% !important;
            height: 45px !important;
        }}
        .main-title {{ text-align: center; font-size: 3.5rem; font-weight: 800; color: {text_color}; margin-top: 20px; }}
        .sub-title {{ text-align: center; font-size: 1.2rem; color: #4A90E2; margin-bottom: 30px; letter-spacing: 2px; }}

        .clinical-body {{
            background-color: {card_bg}; padding: 25px; border-radius: 12px;
            border-left: 5px solid #4A90E2; margin-bottom: 25px; color: {text_color};
        }}

        .hover-card  {{
            background: {card_bg}; border-radius: 15px; padding: 30px !important;
            border: 2px solid {border_color}; text-align: center; margin-top: 20px !important;
            color: {text_color} !important;
        }}

        .confidence-container {{
            background-color: #0E1117; border-radius: 10px; height: 26px;
            width: 100%; margin-top: 15px; border: 1px solid {border_color}; overflow: hidden;
        }}
        .confidence-fill {{
            height: 100%; background: linear-gradient(90deg, #4A90E2 0%, #63B3ED 100%);
            display: flex; align-items: center; justify-content: center;
            font-size: 13px; font-weight: bold; color: white;
        }}

        @keyframes brainRotateFade {{
            0% {{ transform: translate(-50%, -50%) rotate(0deg) scale(0.5); opacity: 0; }}
            20% {{ transform: translate(-50%, -50%) rotate(72deg) scale(1.2); opacity: 1; }}
            80% {{ transform: translate(-50%, -50%) rotate(288deg) scale(1.2); opacity: 1; }}
            100% {{ transform: translate(-50%, -50%) rotate(360deg) scale(2); opacity: 0; }}
        }}
        .brain-overlay {{
            position: fixed; top: 50%; left: 50%; z-index: 9999;
            font-size: 200px; pointer-events: none;
            animation: brainRotateFade 2.5s ease-in-out forwards;
        }}
        
        .doc-container {{ background-color: {card_bg}; padding: 40px; border-radius: 12px; border: 1px solid {border_color}; }}
        </style>
        """, unsafe_allow_html=True)

apply_style(st.session_state.theme)

# load asset
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phase2_clean.keras"
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

@st.cache_resource
def load_neuro_model():
    if MODEL_PATH.exists():
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    return None

model = load_neuro_model()

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

if 'view' not in st.session_state:
    st.session_state.view = 'landing'

#  navigation on top
#  fit colums on one line, width of colums
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns([0.6, 2.0, 1.8, 1.2, 1.8, 0.1])

with nav_col1:
    theme_label = "☀️" if st.session_state.theme == 'dark' else "🌙"
    if st.button(theme_label):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()

if st.session_state.view == 'landing':
    with nav_col3:
        if st.button("📄 Documentation"):
            st.session_state.view = 'docs'
            st.rerun()
    with nav_col4:
        if st.button("❓ FAQ"):
            st.session_state.view = 'faq'
            st.rerun()
    with nav_col5:
        if st.button("📧 Contact Us"):
            st.session_state.view = 'contact'
            st.rerun()

#each page

# landing PAGE
if st.session_state.view == 'landing':
    st.markdown("<h1 class='main-title'>NEURO DETECT</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>INTELLIGENCE AMPLIFIED</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists("logo.png"):
            st.image("logo.png", use_container_width=True)
        
    st.write("##")
    
    st.markdown('<div class="main-btn-container">', unsafe_allow_html=True)
    if st.button("Enter Diagnostic Portal"):
        st.session_state.view = 'portal'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# CONTACT US page
elif st.session_state.view == 'contact':
    if st.button("← Back to Home"):
        st.session_state.view = 'landing'
        st.rerun()
    
    st.markdown("## 📧 Contact Support")
    
    
    with st.form("contact_form", clear_on_submit=True):
        st.write("Send a message to **spneurodetect@gmail.com**")
        user_email = st.text_input("Your Email Address")
        user_msg = st.text_area("Message", height=150)
        
        submit_button = st.form_submit_button("Send Message")
        
        if submit_button:
            if user_email and user_msg:
                st.success(f"Message received. We will contact you shortly!")
                st.balloons()
            else:
                st.error("Please fill out both fields.")



# PROJECT DOCUMENTATION page
elif st.session_state.view == 'docs':
    if st.button("← Back to Home"):
        st.session_state.view = 'landing'
        st.rerun()
    st.markdown("## 📄 Project Documentation")
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            readme_text = f.read()
            # Adding \n\n ensures the Markdown headings (#) work correctly inside the div
            st.markdown(f'<div class="doc-container">\n\n{readme_text}\n\n</div>', unsafe_allow_html=True)

# FAQ page with questions and answers 

elif st.session_state.view == 'faq':
    if st.button("← Back to Home"):
        st.session_state.view = 'landing'
        st.rerun()
    st.markdown("## ❓ Frequently Asked Questions")
    st.markdown("""
        <div class="doc-container">
            <h4>What is NeuroDetect</h4>
            <p>NeuroDetect  is a deep learning-based diagnostic support tool designed to identify and classify brain tumors from MRI scans into four categories:
                 Glioma, Meningioma, Pituitary, or No Tumor.</p>    
            <h4>What is the purpose of this tool?</h4>
            <p>To assist in the classification of brain tumors using deep learning patterns from MRI scans.
                This portal is designed for educational and research purposes to demonstrate the capabilities of Computer Vision in healthcare.
                 It is intended to assist, not replace, professional radiological evaluation.</p>
            <h4>What model architecture is used?</h4>
            <p>Neurodetect is built on a Convolutional Neural Network (CNN) trained on thousands of labeled MRI slices to recognize spatial patterns and textures indicative of different tumor pathologies.</p>
                 <h4>Is it 100% accurate?</h4>
            <p>This is research based and still in early stages of deployment but with more datasets and model training we believe it will optimize to be highly accurate.
                 All findings must be reviewed by a specialist.</p>
        </div>
    """, unsafe_allow_html=True)

# MRI ANALYSIS PORTAL page
elif st.session_state.view == 'portal':

    @st.dialog("Acknowledgment")
    def ack_modal():
        st.markdown("""
        This website is part of a **University Senior Project** and is intended for
        educational and research purposes only.

        It is not intended for clinical use and should not be used as a substitute
        for professional medical diagnosis, treatment, or radiological evaluation.

        By proceeding, you acknowledge that the results are for informational purposes
        only and should be reviewed by a qualified healthcare professional.
        """)

        if st.button("I acknowledge"):
            st.session_state.disclaimer_accepted = True
            st.rerun()

    if not st.session_state.disclaimer_accepted:
        ack_modal()
        st.stop()

    if os.path.exists("ai_head.png"):
        img_data = get_base64_image("ai_head.png")

    uploaded_file = st.file_uploader("Upload MRI", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Current Patient Scan", use_container_width=True)
        
        if st.button("🧠 INITIATE NEURAL DIAGNOSTIC"):
              if model is None:
                st.error(f"Model not found at: {MODEL_PATH}")
     
              else:
                with st.status("🧬 Analyzing Neural Patterns...", expanded=True) as status:
                    st.write("Isolating region of interest...")

                    img = image.convert("RGB").resize((256, 256))
                    img_array = np.array(img).astype("float32")
                    img_array = np.expand_dims(img_array, axis=0)

                    time.sleep(0.5)
                    st.write("Extracting deep features...")
                    preds = model.predict(img_array)

                    time.sleep(0.5)
                    st.write("Classifying pathology...")
                    idx = np.argmax(preds[0])
                    confidence = np.max(preds[0]) * 100
                    if confidence < 80:
                        status.update(label="❌ Invalid Scan", state="error", expanded=False)
                        st.error("Invalid scan. Please upload a correct brain MRI scan.")
                        st.stop()
                        status.update(label="✅ Analysis Complete", state="complete", expanded=False)
               
            

                brain_placeholder = st.empty()
                brain_placeholder.markdown('<div class="brain-overlay">🧠</div>', unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="hover-card">
                        <h4 style="color: #4A90E2; margin-bottom: 5px; letter-spacing: 1px;">PRIMARY DIAGNOSIS</h4>
                        <h1 style="margin: 0; font-size: 2.5rem;">{CLASS_NAMES[idx]}</h1>
                        <hr style="border: 0; border-top: 1px solid #30363D; margin: 20px 0;">
                        <p style="font-size: 1rem; color: #8B949E; margin-bottom: 5px;">Confidence Score</p>
                        <div class="confidence-container">
                            <div class="confidence-fill" style="width: {confidence}%;">
                                {confidence:.1f}%
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                if CLASS_NAMES[idx] == "No Tumor":
                    st.toast("✅ Analysis complete: No abnormalities detected.", icon="🧠")
                else:
                    st.toast(f"⚠️ Potential {CLASS_NAMES[idx]} identified.", icon="❗")

                time.sleep(2.5)
                brain_placeholder.empty()

st.markdown("<br><hr><center><p style='color: #666;'>Senior Project 2026 | NeuroDetect AI</p></center>", unsafe_allow_html=True)
