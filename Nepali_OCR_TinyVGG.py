import streamlit as st
import torch
import torchvision
import model_builder
from torchvision import transforms

# Setup class names
class_names = ['अ', 'अं', 'अः', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ', 'क', 'क्ष', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'ज्ञ', 'झ',
               'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'त्र', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स',
               'ह', '०', '१', '२', '३', '४', '५', '६', '७', '८', '९']

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to load in the model
def load_model(filepath):
  # Need to use same hyperparameters as saved model 
  model = model_builder.TinyVGG(input_shape=1, # number of color channels (3 for RGB) 
                  hidden_units=10, 
                  output_shape=len(class_names)).to(device)

  # Load in the saved model state dictionary from file                               
  model.load_state_dict(torch.load(filepath, map_location=torch.device(device)))
  return model

# Function to load in model + predict on select image
def predict_on_image(image_path, model):
    
    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    
    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255. 
    

    # 3. Transform if necessary
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1)
        ])
    
    target_image = transform(target_image)
    
    # 4. Make sure the model is on the target device
    model.to(device)
    
    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
    
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))
        
    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    return class_names[target_image_pred_label]

FILEPATH = 'models/model.pth'
IMAGENAME = 'img.jpg'
FONT_SIZE = 100

if 'model' not in st.session_state:
    st.session_state.model = load_model(FILEPATH)

if 'prediction' not in st.session_state:
    st.session_state.prediction = predict_on_image(IMAGENAME, st.session_state.model)

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
    st.session_state.prediction = predict_on_image(IMAGENAME, st.session_state.model)

st.button(label="Perform OCR", key="btn", on_click=perform_prediction)





