import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image

st.title("Pneumonia Detection in Chest X-ray EDUCATIONAL USE ONLY")
st.header("Pneumonia classification from chest X-ray")
st.text("Upload a Chest X-ray Image for image classification as pneumonia or normal")

uploaded_file = st.file_uploader("Upload the chest X-ray ...", type=["jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Chest X-ray.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'model4.keras')
    if label > 0.5:
        st.write("The chest X-ray indicates pneumonia")
    else:
        st.write("The chest X-ray is normal")
