# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load YOLO model
model = YOLO("best.pt")

st.title("Garbage Detection App")
st.write("Upload an image and detect garbage types.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    image.save(temp_file.name)

    # Run YOLO prediction
    results = model(temp_file.name)
    result_img = results[0].plot()

    st.image(result_img, caption="Detection Result", use_column_width=True)

    # Count detected garbage types
    counts = {}
    for box in results[0].boxes.cls:
        cls_id = int(box)
        cls_name = results[0].names[cls_id]
        counts[cls_name] = counts.get(cls_name, 0) + 1

    st.write("Detected Garbage Counts:", counts)