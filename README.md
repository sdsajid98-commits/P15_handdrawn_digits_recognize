# Handwritten Digit Recognizer

This is an interactive web app built using Streamlit and TensorFlow that recognizes handwritten digits (0–9). Users can draw a digit on a canvas, and the AI model predicts the digit along with a confidence score.

---

## Features

- Interactive drawing canvas: Draw digits directly in the browser.
- Real-time prediction: Model predicts the digit on click.
- Confidence score: Shows how confident the model is about its prediction.
- Processed image preview: Displays the grayscale 28x28 image that the model uses.

---

## Libraries Used

- streamlit – For the web interface.
- numpy – Numerical operations.
- tensorflow / keras – To load the trained MNIST model.
- PIL (Pillow) – Image processing.
- streamlit_drawable_canvas – For the interactive drawing canvas.

---

## How to Run

1. Install required packages:
   pip install streamlit tensorflow pillow streamlit-drawable-canvas numpy

2. Place the trained model mnist_digit_recognizer.h5 in the same folder as the script.

3. Run the Streamlit app:
   streamlit run app.py

4. Use the app:
   - Draw a digit (0–9) in the canvas.
   - Press Predict Digit.
   - The predicted digit and confidence score will appear.
   - The processed 28x28 grayscale image will also be shown.

---

## File Structure

project-folder/
│
├─ app.py                      # Streamlit app script
├─ mnist_digit_recognizer.h5   # Pre-trained TensorFlow model
└─ README.md

---

## Notes

- The model is trained on the classic MNIST dataset.
- The app is intended for demonstration and learning purposes.
- This is a common beginner ML project, so it’s best included in a portfolio only as part of a "Small Experiments / Learning Projects" section.
