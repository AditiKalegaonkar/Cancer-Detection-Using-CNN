import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
@st.cache_resource
def load_model_from_path(path):
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model_from_path('cnn_model_updated.h5')

# Define the class names
class_names = ['malignant','benign']

# Define a function to preprocess the image
def preprocess_image(img):
    image = image / 255.0
    # Add batch dimension (1, height, width, channels)
    image = np.expand_dims(image, axis=0)
    return img_array

# Streamlit app
st.title('Cancer Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(256, 256))
    img_array = preprocess_image(img)
    
    # Make prediction
    if model:
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=-1)
        predicted_class = class_names[predicted_class_idx[0]]
        
        # Display the predicted category
        st.write(f"Prediction: {predicted_class}")