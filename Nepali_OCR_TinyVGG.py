import streamlit as st
import predict

FILEPATH = 'models/model.pth'
IMAGENAME = 'img.jpg'
FONT_SIZE = 100

if 'model' not in st.session_state:
    st.session_state.model = predict.load_model(FILEPATH)

if 'prediction' not in st.session_state:
    st.session_state.prediction = predict.predict_on_image(IMAGENAME, st.session_state.model)

st.set_page_config(
    page_title="Nepali OCR",
    page_icon="🇳🇵",
    layout="wide",
)


st.title("Nepali Character Recognition")

st.sidebar.markdown('# Nepali OCR')

uploaded_file = st.file_uploader(label="Upload your image file",
                 type=['jpg', 'jpeg'],
                 accept_multiple_files=False,
                 key="uploaded_file",
                 on_change=None,
                 args=None,
                 kwargs=None,
                 disabled=False,
                 label_visibility="visible"
)


col1, col2 = st.columns(2)

with col1:
    st.write("Image")
    if uploaded_file is not None:
        with open(IMAGENAME,'wb+') as f:
            f.write(uploaded_file.read())
        
    st.image(IMAGENAME, caption="Input Image", width=100)
    

with col2:
    st.write("Predicted Character")
    
    html_str = f"""
    <style>
    .big-font{{
    font: bold {FONT_SIZE}px Courier;
    }}
    </style>
    <p class="big-font">{st.session_state.prediction}</p>
    """

    st.markdown(html_str, unsafe_allow_html=True)

def perform_prediction():
    st.session_state.prediction = predict.predict_on_image(IMAGENAME, st.session_state.model)

st.button(label="Perform OCR", key="btn", on_click=perform_prediction)





