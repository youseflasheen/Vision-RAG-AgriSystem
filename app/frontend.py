import streamlit as st
import requests
from PIL import Image

# Basic page configuration
st.set_page_config(page_title="AgriVision AI", page_icon="🌿", layout="centered")

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/diagnose/"

# App UI headers
st.title("🌿 Vision-RAG Agricultural System")
st.write("Upload a picture of a diseased crop leaf, and our AI will diagnose it and provide expert treatment advice.")

# Image upload widget
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Trigger prediction
    if st.button("Diagnose Crop 🚀"):
        with st.spinner("Analyzing image and consulting Knowledge Base..."):
            try:
                # Prepare the file payload for the API
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Analysis Complete!")
                    
                    # Display disease name
                    disease_name = result['disease_class'].replace('_', ' ')
                    st.subheader(f"🦠 Disease Detected: {disease_name}")
                    
                    # Display confidence progress bar
                    st.progress(result['confidence'])
                    st.write(f"**Confidence:** {result['confidence']:.2%}")
                    
                    st.divider()
                    
                    # Display expert advice (Markdown formatting supported automatically)
                    st.subheader("📋 Expert Treatment Advice")
                    st.markdown(result['expert_advice'])
                    
                else:
                    st.error(f"Error from API: {response.status_code}")
            
            except Exception as e:
                st.error("Failed to connect to the backend. Is the FastAPI server running?")