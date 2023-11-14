import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

IMG_SIZE = 227

st.title('Pneumonia prediction based on chest X-Ray image')

input = open('lrc_xray.pkl', 'rb')
model = pkl.load(input)

st.header('Upload chest X-Ray image')
uploaded_file = st.file_uploader("Choose an image file", type=(['png', 'jpg', 'jpeg']))

gre = st.number_input('Insert GRE Score')
toefl = st.number_input('Insert TOEFL Score')
uni_rate = st.number_input('Insert University Rating')
sop = st.number_input('Insert SOP')
lor = st.number_input('Insert LOR')
cgpa = st.number_input('Insert CGPA')
research = st.number_input('Insert Research')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Test image')

    if st.button('Predict'):
        image = image.resize((IMG_SIZE*IMG_SIZE*3, 1))
        feature_vector = np.array(image)
        result = str((model.predict(feature_vector))[0])

        st.header('Result')
        st.text(result)
