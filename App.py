# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="Image Editing App",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Editing App")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("Controls")

blur_value = st.sidebar.slider("Blur", 1, 25, 1, step=2)
sharpness_value = st.sidebar.slider("Sharpness", 0.5, 3.0, 1.0, step=0.1)
brightness_value = st.sidebar.slider("Brightness", -100, 100, 0, step=5)
contrast_value = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0, step=0.1)

edge_detection = st.sidebar.checkbox("Edge Detection")
threshold1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
threshold2 = st.sidebar.slider("Threshold 2", 0, 255, 200)

grayscale = st.sidebar.checkbox("Grayscale")

# Upload Image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Convert to BGR (OpenCV format)
    processed = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # -----------------------------
    # Blur
    # -----------------------------
    if blur_value > 1:
        if blur_value % 2 == 0:
            blur_value += 1  # must be odd
        processed = cv2.GaussianBlur(processed, (blur_value, blur_value), 0)

    # -----------------------------
    # Sharpness (Fixed kernel scaling)
    # -----------------------------
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]) * sharpness_value

    processed = cv2.filter2D(processed, -1, kernel)

    # -----------------------------
    # Brightness + Contrast
    # -----------------------------
    processed = cv2.convertScaleAbs(
        processed,
        alpha=contrast_value,
        beta=brightness_value
    )

    # -----------------------------
    # Grayscale
    # -----------------------------
    if grayscale:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # Edge Detection (FIXED order)
    # -----------------------------
    if edge_detection:
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed

        # Ensure correct threshold order
        t1, t2 = sorted([threshold1, threshold2])

        processed = cv2.Canny(gray, t1, t2)

    # -----------------------------
    # Convert for display
    # -----------------------------
    if len(processed.shape) == 2:
        display_image = processed
    else:
        display_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # Layout
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img_array, use_container_width=True)

    with col2:
        st.subheader("Processed Image")
        st.image(display_image, use_container_width=True)

    # -----------------------------
    # Download
    # -----------------------------
    final_img = Image.fromarray(display_image if len(processed.shape) != 2 else processed)

    buf = io.BytesIO()
    final_img.save(buf, format="PNG")

    st.download_button(
        label="Download Image",
        data=buf.getvalue(),
        file_name="edited_image.png",
        mime="image/png"
    )
