import streamlit as st
import cv2
import numpy as np
from PIL import Image

# PAGE CONFIG
st.set_page_config(
    page_title="Dental Pathology Detection",
    layout="wide",
)

# SIDEBAR
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["Home", "Project Details", "Methodology", "Detection System", "About"],
)

# CLASSES
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


# LOAD MODELS
@st.cache_resource
def load_xray_model():
    from ultralytics import YOLO

    return YOLO("models/xray_best.pt")


@st.cache_resource
def load_camera_model():
    from ultralytics import YOLO

    return YOLO("models/camera_best.pt")


# HOME
if page == "Home":
    st.title("Dental Pathology Detection using YOLO")

# DETECTION SYSTEM
elif page == "Detection System":
    st.title("AI Detection System")

    input_type = st.radio(
        "Select Input Type", ["X-ray Image", "Camera Image"], horizontal=True
    )

    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        try:
            # READ IMAGE
            image = Image.open(uploaded).convert("RGB")
            img = np.array(image)

            # SAFETY CHECK
            if img is None or img.size == 0:
                st.error("Invalid image")
                st.stop()

            # DISPLAY INPUT
            col1, col2 = st.columns(2)
            col1.image(image, caption="Input Image", use_container_width=True)

            # LOAD MODEL
            model = (
                load_xray_model()
                if input_type == "X-ray Image"
                else load_camera_model()
            )

            class_map = XRAY_CLASSES if input_type == "X-ray Image" else CAMERA_CLASSES

            # YOLO PREDICTION (use RGB directly)
            results = model(img, conf=conf_threshold)

            # COPY IMAGE FOR DRAWING
            output = img.copy()

            detected_items = []

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for box in boxes:
                    try:
                        # SAFE EXTRACTION
                        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
                        x1, y1, x2, y2 = xyxy

                        cls_id = int(box.cls.cpu().numpy()[0])
                        conf = float(box.conf.cpu().numpy()[0])

                        label = class_map.get(cls_id, f"class{cls_id}")
                        color = COLORS.get(label, (255, 255, 255))

                        # DRAW BOX
                        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                        cv2.putText(
                            output,
                            f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

                        detected_items.append((label, conf))

                    except Exception as e:
                        st.warning(f"Skipping one detection due to error: {e}")

            # DISPLAY OUTPUT
            col2.image(output, caption="Detection Output", use_container_width=True)

            # LEGEND
            st.subheader("Legend")

            legend_classes = (
                XRAY_CLASSES if input_type == "X-ray Image" else CAMERA_CLASSES
            )

            for _, label in legend_classes.items():
                color = COLORS.get(label, (255, 255, 255))

                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; margin-bottom:6px;">
                    <div style="width:15px; height:15px; background-color:rgb{color}; margin-right:10px;"></div>
                    <span>{label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # RESULTS
            st.subheader("Detection Results")

            if detected_items:
                for label, conf in detected_items:
                    st.write(f"{label} → {conf:.2f}")
            else:
                st.success("No abnormalities detected.")

        except Exception as e:
            st.error(f"Error during processing: {e}")

# OTHER PAGES
elif page == "Project Details":
    st.title("Project Details")

elif page == "Methodology":
    st.title("Methodology")

else:
    st.title("About")
