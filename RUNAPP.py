import streamlit as sl
import torch
from torchvision import transforms
from PIL import Image
import os
from model_setup import build_model 

sl.set_page_config(page_title="Pneumonia Detector", layout="centered")

sl.markdown("""
    <style>
        .title-header {
            text-align: center;
            color: #1f77b4;
            font-size: 40px;
            font-weight: 700;
        }
        .subtext {
            text-align: center;
            color: #555555;
            margin-bottom: 20px;
        }
        .stButton>button {
            display: block;
            margin: auto;
            background-color: #1f77b4;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 18px;
            width: 200px;
        }
        .result-box {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
            font-size: 24px;
            font-weight: bold;
            border: 2px solid #dcdcdc;
        }
    </style>
""", unsafe_allow_html=True)

sl.markdown('<div class="title-header">Pneumonia DETECTOR</div>', unsafe_allow_html=True)
sl.markdown('<div class="subtext">By: Sumukh Sudhir Jagirdar, Aedin Cowan, Brian Chan, Joys James</div>', unsafe_allow_html=True)

# --- 1. Model Loading Logic ---
# We use cache_resource so the model loads only once, not every time you click a button
@sl.cache_resource
def load_system_model():
    device = torch.device('cpu')
    
    model, _, _ = build_model(num_classes=2, gray_scale=False, freeze_backbone=False, lr=0.001)
    
    path = 'pneumonia_model_with_hparams.pth'
    
    if not os.path.exists(path):
        return None

    # Robust Weight Loading (Same as evaluate.py)
    try:
        checkpoint = torch.load(path, map_location=device)
        
        # Handle different saving formats (Dictionary vs State Dict)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("model.", "")
            new_state_dict[name] = v
            
        # Load weights
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        sl.error(f"Error loading model: {e}")
        return None

model = load_system_model()

if model is None:
    sl.error("⚠️ Model file 'pneumonia_model_with_hparams.pth' not found. Please place it in this folder.")

# --- 2. Image Preprocessing Logic ---
def process_image(img):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# --- 3. Main Interface ---
uploaded_file = sl.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    sl.image(image, caption='Uploaded X-Ray', use_column_width=True)

    # Detect Button
    if sl.button("Detect"):
        if model:
            with sl.spinner('Analyzing lungs...'):
                # Predict
                img_tensor = process_image(image)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                
                # Result
                classes = ['NORMAL', 'PNEUMONIA']
                result_text = classes[predicted.item()]
                conf_score = confidence.item() * 100
                
                # Dynamic Color
                color = "#28a745" if result_text == "NORMAL" else "#dc3545" 
                
                sl.markdown(f"""
                    <div class="result-box">
                        Prediction: <span style="color: {color};">{result_text}</span><br>
                        <span style="font-size: 18px; color: #555;">Confidence: {conf_score:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)