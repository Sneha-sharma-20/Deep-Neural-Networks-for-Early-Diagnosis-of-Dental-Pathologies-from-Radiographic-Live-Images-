import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Deep Neural Networks for Early Diagnosis of Dental Pathologies",
    layout="wide",
)

# CLEAN PROFESSIONAL THEME
st.markdown(
    """
    <style>

    .stApp {
        background-color: #ffffff;
    }

    /* Fix main text */
    p, h1, h2, h3, h4, h5, h6, span, label {
        color: #000000 !important;
    }

    /* Header */
    .main-header {
        background-color: #0a3d62;
        padding: 22px;
        border-radius: 6px;
        color: white !important;
        text-align: center;
        font-size: 24px;
        font-weight: 600;
    }

    /* Cards */
    .section-card {
        background-color: #f5f9ff;
        padding: 20px;
        border-radius: 6px;
        border-left: 5px solid #1e5f9e;
        margin-bottom: 20px;
        color: #000000 !important;
    }

    .sub-heading {
        color: #0a3d62 !important;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e1e2f;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Detection result fix */
    div {
        color: #000000;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# SIDEBAR
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["Home", "Project Details", "Methodology", "Detection System", "About"],
)

# CLASSES & COLORS
XRAY_CLASSES = {
    0: "Healthy Teeth",
    1: "Caries",
    2: "Impacted Teeth",
    3: "Broken Down Crown/Root",
    4: "Infection",
    5: "Fractured Teeth",
}

CAMERA_CLASSES = {
    0: "Caries",
    1: "Ulcer",
    2: "Tooth Discoloration",
    3: "Gingivitis",
}

COLORS = {
    "Healthy Teeth": (0, 255, 0),
    "Caries": (255, 0, 0),
    "Impacted Teeth": (0, 255, 255),
    "Broken Down Crown/Root": (255, 165, 0),
    "Infection": (0, 0, 255),
    "Fractured Teeth": (255, 0, 255),
    "Ulcer": (128, 0, 128),
    "Tooth Discoloration": (0, 128, 255),
    "Gingivitis": (0, 255, 128),
}


# MODEL LOADING
@st.cache_resource
def load_xray_model():
    from ultralytics import YOLO

    return YOLO("models/xray_best.pt")


@st.cache_resource
def load_camera_model():
    from ultralytics import YOLO

    return YOLO("models/camera_best.pt")


# HOME PAGE
if page == "Home":
    st.markdown(
        "<div class='main-header'>Deep Neural Networks for Early Diagnosis of Dental Pathologies</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='sub-heading'>Introduction</div>", unsafe_allow_html=True)

    st.write("""
Dental pathologies such as caries, gingivitis, infections, ulcers, and structural abnormalities are highly prevalent.

This project uses YOLO-based deep learning models to automatically detect dental issues from X-ray and clinical images.
""")
    st.markdown("</div>", unsafe_allow_html=True)

# PROJECT DETAILS
elif page == "Project Details":
    st.markdown(
        "<div class='main-header'>Project Overview</div>", unsafe_allow_html=True
    )

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-heading'>Problem Statement</div>", unsafe_allow_html=True
    )

    st.write("""
• Visual similarities between healthy and diseased tissues  
• Variations in image quality  
• Early-stage detection difficulty  

AI-based detection improves consistency and accuracy.
""")
    st.markdown("</div>", unsafe_allow_html=True)

# METHODOLOGY
elif page == "Methodology":
    st.markdown(
        "<div class='main-header'>System Methodology</div>", unsafe_allow_html=True
    )

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="font-size:16px; line-height:1.8;">
        <b>1.</b> Data Acquisition <br>
        <b>2.</b> Annotation <br>
        <b>3.</b> Preprocessing <br>
        <b>4.</b> Data Augmentation <br>
        <b>5.</b> Model Training <br>
        <b>6.</b> Evaluation Metrics
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# DETECTION SYSTEM
elif page == "Detection System":
    st.markdown(
        "<div class='main-header'>AI-Based Detection System</div>",
        unsafe_allow_html=True,
    )

    input_type = st.radio(
        "Select Input Type", ["X-ray Image", "Camera Image"], horizontal=True
    )
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    uploaded = st.file_uploader("Upload Dental Image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        img = np.array(image)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Input Image", use_container_width=True)

        model = (
            load_xray_model() if input_type == "X-ray Image" else load_camera_model()
        )
        class_map = XRAY_CLASSES if input_type == "X-ray Image" else CAMERA_CLASSES

        results = model(img_bgr, conf=conf_threshold)
        output = img.copy()
        detections = results[0].boxes

        detected_items = []

        if detections is not None and len(detections) > 0:
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                label = class_map.get(cls_id, f"class{cls_id}")
                color = COLORS.get(label, (255, 255, 255))

                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    output,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                detected_items.append((label, conf))

            col2.image(output, caption="Detection Output", use_container_width=True)

            # LEGEND
            st.markdown("###  Legend")

            # Select correct class map
            if input_type == "X-ray Image":
                legend_classes = XRAY_CLASSES
                st.markdown("###  X-ray Legend")
            else:
                legend_classes = CAMERA_CLASSES
                st.markdown("###  Camera Image Legend")

            # Show only relevant legend
            for cls_id, label in legend_classes.items():
                color = COLORS.get(label, (255, 255, 255))

                st.markdown(
                    f"""
                            <div style="display:flex; align-items:center; margin-bottom:6px;">
                            <div style="width:15px; height:15px; background-color:rgb{color}; margin-right:10px;"></div>
                            <span style="color:#000000;">{label}</span>
                            </div>
                            """,
                    unsafe_allow_html=True,
                )

            # RESULTS
            st.markdown("### Detection Results")
            for label, conf in detected_items:
                st.markdown(
                    f"""
                    <div style="padding:10px; margin:5px; border-left:5px solid #0a3d62; background:#f5f9ff;">
                        <b>Condition:</b> {label} <br>
                        <b>Confidence Score:</b> {conf:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            col2.image(output, caption="Detection Output", use_container_width=True)
            st.success("No abnormalities detected.")

# ABOUT
else:
    st.markdown(
        "<div class='main-header'>About the Project</div>", unsafe_allow_html=True
    )

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)

    st.write("""
BVRIT Hyderabad College of Engineering for Women  

This project demonstrates AI-based dental diagnosis using YOLO models.

Future enhancements include real-time detection and edge deployment.
""")

    st.markdown("</div>", unsafe_allow_html=True)
