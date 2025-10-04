import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title='SpectraSense Demo', layout='centered')
st.title('SpectraSense — HgB estimation from lip images')
st.markdown('Upload a lip image and optional metadata CSV to get a predicted HgB (placeholder).')

uploaded_file = st.file_uploader('Upload lip image', type=['jpg','jpeg','png','heic'])
meta_file = st.file_uploader('Optional: meta.csv', type=['csv'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=300)
    st.write('Prediction (placeholder): 12.0 g/dL')

if meta_file:
    df = pd.read_csv(meta_file)
    st.dataframe(df.head())
