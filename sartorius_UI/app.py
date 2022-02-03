import streamlit as st
import io
import base64
from PIL import Image
import requests

model_server_URL = "http://sartorius-server:8080/predictions/sartorius"

if "uploaded_file" not in st.session_state:
    st.session_state['uploaded_file'] = None

st.title("Welcome to Sartorious Cell detection app")
st.session_state['uploaded_file'] = st.file_uploader(
    "Please upload your cell image here", accept_multiple_files=False)
prediction_button = st.button("Predict!")

if st.session_state['uploaded_file'] is not None:
    st.markdown("<h3> Uploaded image </h3>", unsafe_allow_html=True)
    st.image(Image.open(st.session_state['uploaded_file']))

    if prediction_button:
        st.markdown("<h3> Predictions </h3>", unsafe_allow_html=True)
        server_responds = requests.put(
            model_server_URL, data=st.session_state['uploaded_file'].getvalue())

        image_decoded = base64.b64decode(
            server_responds.content, validate=False)

        st.image(io.BytesIO(image_decoded))
