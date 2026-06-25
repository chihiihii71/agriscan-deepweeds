import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import plotly.express as px
import os

# -------------------------------
# 1. Config
# -------------------------------
NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia",
    "Parthenium", "Prickly Acacia", "Rubber Vine",
    "Siam Weed", "Snake Weed", "Negative"
]

# Models available locally (GitHub)
from huggingface_hub import hf_hub_download

REPO_ID = "https://huggingface.co/Jaoooooo9/agriscan-streamlit"

@st.cache_resource
def get_model_path(filename):
    return hf_hub_download(repo_id=REPO_ID, filename=filename)

MODEL_OPTIONS = {
    "ResNeSt50d": get_model_path("resnest50d_model.pth"),
    "ECA-ResNet50d": get_model_path("ecaresnet50d_model.pth"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2. Streamlit UI - Model Selection
# -------------------------------
st.set_page_config(page_title="DeepWeeds Image Classifier", layout="wide")
st.title(" DeepWeeds Image Classifier")
st.write("Upload an image of a weed, and select a model to predict its class.")

selected_model_name = st.sidebar.selectbox("Select Model", list(MODEL_OPTIONS.keys()))
MODEL_PATH = MODEL_OPTIONS[selected_model_name]

# -------------------------------
# 3. Load Model
# -------------------------------
@st.cache_resource
def load_model(model_name, model_path):
    # Determine architecture
    if "resnest" in model_name.lower():
        model = timm.create_model("resnest50d", pretrained=False, num_classes=NUM_CLASSES)
    elif "eca" in model_name.lower():
        model = timm.create_model("ecaresnet50d", pretrained=False, num_classes=NUM_CLASSES)
    else:
        raise ValueError("Unknown model selected!")

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model(selected_model_name, MODEL_PATH)

# -------------------------------
# 4. Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------------
# 5. Prediction Function
# -------------------------------
def predict(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        return CLASS_NAMES[pred_idx], probs.cpu().numpy()

# -------------------------------
# 6. Streamlit UI - Image Upload & Results
# -------------------------------
col1, col2 = st.columns(2)

# --- Left Column ---
with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)

# --- Right Column ---
with col2:
    if uploaded_file is not None:
        st.subheader("Prediction Results")
        label, probs = predict(image)

        # Top Prediction
        pred_prob = max(probs) * 100
        st.markdown(f"""
            <div style="background-color:#1e4620;padding:10px;border-radius:10px">
            <h3 style="color:white;"> Predicted Class: {label} ({pred_prob:.2f}%)</h3>
            </div>
        """, unsafe_allow_html=True)

        st.write("") # Add space

        # Probability Bar Chart
        st.subheader("Class Probabilities")
        df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probs * 100})
        fig = px.bar(df, x="Class", y="Probability")
        st.plotly_chart(fig, use_container_width=True)

        # Detailed Probability Table
        st.subheader("Detailed Probabilities")
        df_sorted = df.sort_values("Probability", ascending=False).reset_index(drop=True)
        st.dataframe(df_sorted.style.format({"Probability": "{:.2f}%"}), use_container_width=True)
