import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load the trained model
@st.cache_resource
def load_model():
    model = keras.models.load_model("mnist_digit_recognizer.h5")
    return model

model = load_model()

# Page setup
st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="✏️", layout="wide")

st.title("🧠 Handwritten Digit Recognizer")
st.markdown("Draw a digit (0–9) below and let the AI guess it!")

# Canvas for drawing
st.write("### Draw here:")
canvas_result = st_canvas(
    fill_color="white",        # background color
    stroke_width=10,           # brush thickness
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("🔍 Predict Digit"):
    if canvas_result.image_data is not None:
        # Convert canvas image to grayscale and resize to 28x28
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))
        img = ImageOps.grayscale(img)
        img = img.resize((28, 28))
        img_array = np.array(img).reshape(1, 784) / 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Display results
        st.success(f"### ✨ Predicted Digit: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Show processed image
        st.image(img, caption="Processed 28x28 Image", width=150)
    else:
        st.warning("Please draw something first!")
else:
    st.info("Press 'Predict Digit' after drawing your number.")
