"""
Stage 5: Streamlit Web Application for Skin Disease Classification
===================================================================

This module provides an interactive web interface for the skin disease
classification model using Streamlit. Users can upload dermatological
images and receive predictions with confidence scores.

Features:
- Image upload and preview
- Real-time model predictions  
- Probability breakdown for all 7 classes
- Disease information and care recommendations
- Professional UI/UX with proper error handling
- Academic-grade documentation

Model: MobileNetV2 with Transfer Learning
Dataset: HAM10000 (7 disease types)

Author: Academic Deep Learning Project
Date: March 2026
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stTitle { color: #1f77b4; }
    .metric-container {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DISEASE INFORMATION DATABASE
# ============================================================

DISEASE_INFO = {
    "actinic_keratosis": {
        "name": "Actinic Keratosis",
        "description": "Rough, scaly patches on skin caused by prolonged UV exposure.",
        "care": "Use sunscreen (SPF 30+), wear protective clothing, seek annual dermatology check.",
        "note": "May develop into skin cancer if untreated. Medical consultation recommended."
    },
    "basal_cell_carcinoma": {
        "name": "Basal Cell Carcinoma",
        "description": "Most common form of skin cancer, usually slow-growing and manageable.",
        "care": "Requires immediate medical attention from a dermatologist for treatment.",
        "note": "Early detection and treatment significantly improve outcomes."
    },
    "benign_keratosis": {
        "name": "Benign Keratosis",
        "description": "Non-cancerous skin growth, common with age. Completely harmless.",
        "care": "Usually no treatment needed unless cosmetically bothersome.",
        "note": "Harmless but dermatologist confirmation recommended."
    },
    "dermatofibroma": {
        "name": "Dermatofibroma",
        "description": "Benign skin nodule, often resulting from minor injuries or irritation.",
        "care": "No treatment necessary. Avoid irritation or picking at the area.",
        "note": "Harmless but confirm diagnosis with specialist if concerned."
    },
    "melanoma": {
        "name": "Melanoma",
        "description": "Most serious type of skin cancer with high metastasis risk if untreated.",
        "care": "Requires urgent medical evaluation and immediate dermatology referral.",
        "note": "URGENT: Seek immediate attention. Prognosis depends on early detection."
    },
    "melanocytic_nevus": {
        "name": "Melanocytic Nevus (Mole)",
        "description": "Common benign skin growth consisting of melanocytes. Usually not dangerous.",
        "care": "Monitor for changes (ABCDE rule). Yearly skin checks recommended.",
        "note": "Most moles are harmless. Regular monitoring ensures safety."
    },
    "vascular_lesion": {
        "name": "Vascular Lesion",
        "description": "Blood vessel-related skin abnormality, usually cosmetic concern.",
        "care": "Cosmetic or medical treatment available. Consult dermatologist for options.",
        "note": "Usually benign but diagnosis confirmation recommended."
    }
}

# Class names mapping
CLASS_NAMES = list(DISEASE_INFO.keys())
CLASS_DISPLAY_NAMES = [DISEASE_INFO[cls]["name"] for cls in CLASS_NAMES]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

@st.cache_resource
def load_trained_model():
    """
    Load the pre-trained skin disease classification model.
    
    Returns:
        tensorflow.keras.Model: Trained MobileNetV2 model
    """
    try:
        model = load_model("skin_disease_cnn.h5")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'skin_disease_cnn.h5' not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


def preprocess_image(image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model prediction.
    
    Steps:
    1. Convert to RGB (remove alpha channel if present)
    2. Resize to model input size (224x224)
    3. Convert to numpy array
    4. Normalize to [0, 1] range
    5. Add batch dimension
    
    Args:
        image: PIL Image object
        target_size: Target dimensions (height, width)
    
    Returns:
        np.ndarray: Preprocessed image ready for prediction
    """
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize image
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array
    image_array = img_to_array(image)
    
    # Normalize (rescale to [0, 1])
    image_array = image_array / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array


def predict_image(model, image_array: np.ndarray) -> tuple:
    """
    Make prediction on preprocessed image.
    
    Args:
        model: Trained Keras model
        image_array: Preprocessed image array with batch dimension
    
    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    # Get predictions
    predictions = model.predict(image_array, verbose=0)
    probabilities = predictions[0]
    
    # Get predicted class and confidence
    predicted_idx = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx]) * 100
    
    return predicted_class, confidence, probabilities


# ============================================================
# HEADER SECTION
# ============================================================

st.title("🔬 Skin Disease Classification System")

st.markdown("""
---
### Project Overview
This application uses **Deep Learning** to classify skin diseases from dermatological images.

**Technology Stack:**
- **Model Architecture:** MobileNetV2 (Transfer Learning)
- **Pre-training:** ImageNet (1.4M images)
- **Dataset:** HAM10000 (10,015 dermatoscopic images)
- **Classes:** 7 skin disease types
- **Framework:** TensorFlow/Keras

⚠️ **Academic Disclaimer:** This tool is for **educational purposes only**. 
It is **NOT a substitute for professional medical diagnosis**. 
Please consult a qualified dermatologist for accurate diagnosis and treatment.
""")

st.markdown("---")

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================

with st.sidebar:
    st.header("📋 Model Information")
    
    st.subheader("Dataset")
    st.write("""
    **Name:** HAM10000  
    **Images:** 10,015  
    **Resolution:** 224×224 pixels
    """)
    
    st.subheader("Model Architecture")
    st.write("""
    **Base Model:** MobileNetV2  
    **Weights:** ImageNet  
    **Approach:** Transfer Learning  
    **Frozen Layers:** Base model (feature extraction)  
    **Custom Head:** Dense → Dropout → Output
    """)
    
    st.subheader("Training Configuration")
    st.write("""
    **Classes:** 7  
    **Epochs:** 15  
    **Batch Size:** 32  
    **Optimizer:** Adam (lr=0.001)  
    **Augmentation:** Enabled
    """)
    
    st.divider()
    
    st.subheader("🎯 Supported Classes")
    for i, disease in enumerate(CLASS_DISPLAY_NAMES, 1):
        st.write(f"{i}. {disease}")
    
    st.divider()
    
    st.info("""
    📚 **How to Use:**
    1. **Take photo** with camera or **upload image**
    2. Click "Analyze Image"
    3. View predictions and confidence
    4. Read disease information and care tips
    """)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if "last_image_hash" not in st.session_state:
    st.session_state.last_image_hash = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ============================================================
# MAIN LAYOUT
# ============================================================

# Load model once
model = load_trained_model()

# Create two-column layout
col1, col2 = st.columns(2, gap="large")

# ============================================================
# LEFT COLUMN: IMAGE UPLOAD & PREVIEW
# ============================================================

with col1:
    st.subheader("📸 Capture or Upload Image")
    
    # Tab selection for camera or upload
    input_tab1, input_tab2 = st.tabs(["📷 Live Camera", "📁 Upload File"])
    
    image = None
    
    # TAB 1: LIVE CAMERA INPUT
    with input_tab1:
        st.write("📷 **Take a photo using your device camera**")
        
        # Camera control buttons
        button_col1, button_col2 = st.columns(2)
        
        with button_col1:
            if st.button("🎬 Start Camera", use_container_width=True, key="start_camera"):
                st.session_state.camera_active = True
                st.session_state.prediction = None  # Clear prediction
                st.rerun()
        
        with button_col2:
            if st.button("⏹️ Stop Camera", use_container_width=True, key="stop_camera"):
                st.session_state.camera_active = False
                st.rerun()
        
        st.divider()
        
        # Show camera input only if active
        if st.session_state.camera_active:
            camera_image = st.camera_input(
                "Point camera at the skin lesion",
                help="Ensure good lighting and clear view of the lesion",
                key="camera_widget"
            )
            
            if camera_image is not None:
                # Create hash of image to detect changes
                import hashlib
                image_bytes = camera_image.getvalue()
                image_hash = hashlib.md5(image_bytes).hexdigest()
                
                # Clear prediction if new image captured
                if st.session_state.last_image_hash != image_hash:
                    st.session_state.prediction = None
                    st.session_state.last_image_hash = image_hash
                
                image = Image.open(camera_image)
                st.image(
                    image,
                    caption="📸 Camera Capture",
                    use_container_width=True,
                    clamp=True
                )
                st.write(f"📐 Image Size: {image.size[0]}×{image.size[1]} pixels")
                st.write(f"✅ Ready to analyze!")
        else:
            st.info("Click 'Start Camera' to enable camera access")

    
    # TAB 2: FILE UPLOAD
    with input_tab2:
        st.write("📁 **Upload an image file**")
        
        uploaded_file = st.file_uploader(
            "Choose a dermatological image:",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload a clear image of the skin lesion"
        )
        
        if uploaded_file is not None:
            # Create hash of file to detect changes
            import hashlib
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()
            
            # Clear prediction if new file uploaded
            if st.session_state.last_image_hash != file_hash:
                st.session_state.prediction = None
                st.session_state.last_image_hash = file_hash
                st.info("📝 New image loaded! Click 'Analyze Image' to get fresh predictions.")
            
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            st.image(
                image,
                caption="🖼️ Uploaded Image",
                use_container_width=True,
                clamp=True
            )
            
            # Display image info
            st.write(f"📐 Image Size: {image.size[0]}×{image.size[1]} pixels")
            st.write(f"📦 Image Mode: {image.mode}")
            st.write(f"✅ Ready to analyze!")
    
    # Show instructions if no image
    if image is None:
        st.info("👆 Take a photo or upload an image to begin analysis")



# ============================================================
# RIGHT COLUMN: ANALYSIS & PREDICTIONS
# ============================================================

with col2:
    st.subheader("🔍 Analysis")
    
    # Placeholder for results
    result_container = st.container(border=True)
    
    if image is not None:
        # Analyze button
        if st.button("🚀 Analyze Image", use_container_width=True, type="primary"):
            with st.spinner("🔄 Analyzing image with trained model..."):
                try:
                    # Preprocess the current image
                    processed_image = preprocess_image(image)
                    
                    # Get fresh prediction from model
                    predicted_class, confidence, probabilities = predict_image(model, processed_image)
                    
                    # Display prediction
                    with result_container:
                        st.success("✅ Analysis Complete!")
                        
                        disease_name = DISEASE_INFO[predicted_class]["name"]
                        
                        st.markdown(f"""
                        ### 🎯 Prediction Result
                        **Disease:** {disease_name}  
                        **Confidence:** {confidence:.2f}%
                        """)
                        
                        # Progress bar
                        st.progress(confidence / 100.0, text=f"Confidence: {confidence:.1f}%")
                        
                        # Store in session for display sections
                        st.session_state.prediction = {
                            'class': predicted_class,
                            'confidence': confidence,
                            'probabilities': probabilities,
                            'disease_name': disease_name
                        }
                    
                    st.toast(f"✅ Prediction: {disease_name}", icon="🎯")
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.error("Please ensure the model file 'skin_disease_cnn.h5' exists and is valid.")
        
        else:
            if st.session_state.prediction is None:
                with result_container:
                    st.write("👆 Click 'Analyze Image' to get predictions")
            else:
                with result_container:
                    pred = st.session_state.prediction
                    st.success("✅ Analysis Complete!")
                    st.markdown(f"""
                    ### 🎯 Prediction Result
                    **Disease:** {pred['disease_name']}  
                    **Confidence:** {pred['confidence']:.2f}%
                    """)
                    st.progress(pred['confidence'] / 100.0, text=f"Confidence: {pred['confidence']:.1f}%")
    
    else:
        with result_container:
            st.write("⬅️ Upload an image first")


# ============================================================
# PROBABILITY BREAKDOWN
# ============================================================

if image is not None and st.session_state.prediction is not None:
    st.divider()
    
    st.subheader("📊 Prediction Probabilities")
    
    probabilities = st.session_state.prediction['probabilities']
    
    # Create probability dataframe
    prob_df = pd.DataFrame({
        "Disease": CLASS_DISPLAY_NAMES,
        "Probability (%)": probabilities * 100
    })
    
    # Display as bar chart
    st.bar_chart(prob_df.set_index("Disease"), use_container_width=True, height=300)
    
    # Display detailed table
    st.dataframe(
        prob_df.sort_values("Probability (%)", ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DISEASE INFORMATION SECTION
# ============================================================

if image is not None and st.session_state.prediction is not None:
    st.divider()
    st.subheader("📖 Disease Information")
    
    # Get predicted disease info
    predicted_class = st.session_state.prediction['class']
    disease_data = DISEASE_INFO[predicted_class]
    
    # Display predicted disease details
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.success(f"### {disease_data['name']}")
        st.write(f"**Description:**  \n{disease_data['description']}")
    
    with col_right:
        st.info(f"**Care & Prevention:**  \n{disease_data['care']}")
    
    # Important note
    with st.expander("⚠️ Important Medical Note", expanded=True):
        st.warning(f"**{disease_data['note']}**")
        st.write("""
        This prediction is based on image analysis only and should NOT be used 
        for self-diagnosis or self-treatment. Always consult a qualified 
        dermatologist for professional medical evaluation and treatment.
        """)
    
    # Learn more about all diseases
    with st.expander("📚 Learn More About All Diseases"):
        for class_name, info in DISEASE_INFO.items():
            st.subheader(f"🔹 {info['name']}")
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Care:** {info['care']}")
            st.write(f"**Note:** {info['note']}")
            st.divider()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px; padding: 20px;">
    <p>🎓 Developed as a Deep Learning Academic Project | 
    <strong>NOT FOR MEDICAL USE</strong> | 
    Educational Purposes Only</p>
    <p>© 2026 Skin Disease Classification Project | 
    MobileNetV2 Transfer Learning Architecture</p>
</div>
""", unsafe_allow_html=True)
