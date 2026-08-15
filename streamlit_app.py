import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv8 Object Detection", page_icon="🎯", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("YOLOv8 Object Detection")
st.write("Upload an image and adjust the confidence threshold to detect objects.")

confidence = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        st.write(f"**Image:** {uploaded_file.name}  |  **Size:** {image.width} × {image.height}")

        if st.button("🔍 Detect Objects", type="primary"):
            with st.spinner("Running YOLOv8 detection..."):
                results = model.predict(img_array, conf=confidence, verbose=False)
                result = results[0]

            annotated_image = result.plot()
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(img_array, use_container_width=True)

            with col2:
                st.subheader("Detection Results")
                st.image(annotated_image_rgb, use_container_width=True)

            st.subheader("Detection Details")
            if len(result.boxes) > 0:
                rows = []
                for i, box in enumerate(result.boxes, 1):
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    rows.append({
                        "No.": i,
                        "Class": class_name,
                        "Confidence": f"{conf:.2%}",
                        "Bounding Box": f"({x1}, {y1}) → ({x2}, {y2})"
                    })
                st.success(f"Found {len(rows)} object(s).")
                st.table(rows)
            else:
                st.info(f"No objects detected with confidence ≥ {confidence:.0%}. Try lowering the threshold.")
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload a JPG or PNG image to begin.")
