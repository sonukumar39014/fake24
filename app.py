import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import difflib
import os
import gdown
# Page settings
st.set_page_config(
    page_title="Fake Logo Detector",
    page_icon="🕵️‍♂️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .title {
            font-size:40px;
            font-weight:bold;
            text-align:center;
            color:#FF4B4B;
            margin-bottom: 0;
        }
        .result {
            font-size:24px;
            font-weight:bold;
            color:green;
        }
        .fake {
            font-size:24px;
            font-weight:bold;
            color:red;
        }
        .confidence-bar {
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            margin: 10px 0;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            background-color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Brand classes
classes = [
    'Adidas', 'Amazon', 'Android', 'Apple', 'Ariel', 'Bic', 'BMW', 'Burger King', 'Cadbury', 'Chevrolet',
    'Chrome', 'Coca Cola', 'Cowbell', 'Dominos', 'Fila', 'Gillette', 'Google', 'Goya oil', 'Guinness', 'Heinz',
    'Honda', 'Hp', 'Huawei', 'Instagram', 'Kfc', 'Krisspy Kreme', 'Lays', "Levi's", 'Lg', 'Lipton', 'M&m', 'Mars',
    'Marvel', 'McDonald', 'Mercedes Benz', 'Microsoft', 'Mtn', 'Mtn dew', 'NASA', 'Nescafe', 'Nestle', 'Nestle milo',
    'Netflix', 'Nike', 'Nutella', 'Oral b', 'Oreo', 'Pay pal', 'Peak milk', 'Pepsi', 'PlayStation', 'Pringles',
    'Puma', 'Reebok', 'Rolex', 'Samsung', 'Sprite', 'Starbucks', 'Tesla', 'Tiktok', 'Twitter', 'YouTube', 'Zara'
]

# Load model with caching
from tensorflow.keras.saving import custom_object_scope

@st.cache_resource
def load_model_cached():
    import gdown
    model_path = "fine_tuned_model.h5"

    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            file_id = "1uGjmYnj1i9606lX_CIzwEw_kjGBR_FuH"
            url = f"https://drive.google.com/uc?id={file_id}"
            try:
                gdown.download(url, model_path, quiet=False)
            except Exception as e:
                st.error(f"❌ Failed to download model: {e}")
                return None

    try:
        return load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

# ✅ Add this to fix the error:
model = load_model_cached()


# Sidebar
st.sidebar.header("🔍 Logo Detection")
uploaded_file = st.sidebar.file_uploader("Upload Logo Image", type=["jpg", "jpeg", "png", "webp"])
brand_name = st.sidebar.text_input("Expected Brand Name", placeholder="e.g., Nike, Apple, Starbucks")
st.sidebar.markdown("---")
st.sidebar.markdown("**How to use:**\n1. Upload a logo image\n2. Enter expected brand name\n3. Click 'Check Authenticity'")

# Header
st.markdown('<div class="title">🕵️‍♂️ Fake Logo Detector</div>', unsafe_allow_html=True)
st.markdown("---")

# Normalize brand names
def normalize_brand_name(name):
    return name.lower().replace(" ", "").strip()

# Preprocess image
def preprocess_image(uploaded_file):
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return None, None

# Prediction logic
def predict_logo(img_array, user_brand_name):
    preds = model.predict(img_array)[0]
    predicted_idx = np.argmax(preds)

    if predicted_idx >= len(classes):
        st.error(f"Model predicted index {predicted_idx}, but classes only has {len(classes)} items.")
        return None, None, None, None, preds

    predicted_name = classes[predicted_idx]
    confidence = preds[predicted_idx]

    # Normalize
    normalized_user = normalize_brand_name(user_brand_name)
    normalized_pred = normalize_brand_name(predicted_name)

    # Fuzzy match score
    similarity_score = difflib.SequenceMatcher(None, normalized_pred, normalized_user).ratio()

    # Mark Real if high similarity or exact match, even with low confidence
    if similarity_score >= 0.8:
        result = "Real"
    else:
        result = "Fake"

    return result, predicted_name, confidence, similarity_score, preds

    # Normalize both input and prediction
    normalized_user = normalize_brand_name(user_brand_name)
    normalized_pred = normalize_brand_name(predicted_name)

    score = difflib.SequenceMatcher(None, normalized_pred, normalized_user).ratio()

    if normalized_user == normalized_pred and confidence >= 0.85 and score >= 0.95:
        result = "Real"
    else:
        result = "Fake"

    return result, predicted_name, confidence, score, preds

# Main logic
if model is None:
    st.stop()

if uploaded_file:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Uploaded Logo", use_container_width=True)

    if brand_name.strip() == "":
        st.warning("⚠️ Please enter the expected brand name.")
    else:
        if st.button("🔍 Check Authenticity", type="primary"):
            with st.spinner("Analyzing logo..."):
                img_array, img = preprocess_image(uploaded_file)
                if img_array is not None:
                    result, predicted_brand, confidence, similarity, preds = predict_logo(img_array, brand_name)
                    if result is not None:
                        with col2:
                            st.markdown("### Analysis Results")
                            st.markdown(f"**Model Confidence:** {confidence:.1%}")
                            st.markdown(f'<div class="confidence-bar"><div class="confidence-fill" style="width: {confidence*100}%"></div></div>', unsafe_allow_html=True)
                            st.markdown(f"**Brand Match Score:** {similarity:.1%}")
                            st.progress(similarity)
                            if result == "Real":
                                st.markdown(f'<div class="result">✅ Real {predicted_brand} Logo</div>', unsafe_allow_html=True)
                                st.balloons()
                            else:
                                st.markdown(f'<div class="fake">❌ Potential Fake Logo</div>', unsafe_allow_html=True)
                                if confidence >= 0.5:
                                    st.error(f"Expected '{brand_name}' but detected '{predicted_brand}'")
                            with st.expander("Detailed Analysis"):
                                st.markdown(f"""
                                    - **Predicted Brand:** {predicted_brand}
                                    - **Confidence Level:** {confidence:.1%}
                                    - **Brand Match Score:** {similarity:.1%}
                                    - **Verdict:** {result}
                                """)
                                if confidence > 0.1:
                                    st.markdown("**Top Predictions:**")
                                    top_indices = np.argsort(preds)[-3:][::-1]
                                    for idx in top_indices:
                                        if idx < len(classes):
                                            st.markdown(f"- {classes[idx]}: {preds[idx]:.1%}")
else:
    st.info("👈 Please upload a logo image and enter the expected brand name in the sidebar.")
