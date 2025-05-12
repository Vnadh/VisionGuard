import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import models
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import shap
import numpy as np
import torch.nn as nn

CFG = {
    'img_size': 224,
    'batch_size': 32,
    'num_classes': 8,  # N,D,G,C,A,H,M,O
    'metadata_size': 2, # Age (normalized), Sex (encoded)
    'lr': 1e-4,
    'epochs': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# model
class OcularDiseaseModel(nn.Module):
    """Multi-modal model combining image and metadata features"""

    def __init__(self):
        super().__init__()

        # Image feature extractor (shared weights for both eyes)
        self.cnn = models.efficientnet_b0(pretrained=True)
        self.cnn.classifier = nn.Identity()  # Remove final layers

        # Metadata processor
        self.meta_net = nn.Sequential(
            nn.Linear(CFG['metadata_size'], 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16))
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(1280*2 + 16, 512),  # EfficientNet features + metadata
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, CFG['num_classes']))

    def forward(self, left_img, right_img, metadata):
        # Extract image features
        left_features = self._get_image_features(left_img)
        right_features = self._get_image_features(right_img)

        # Process metadata
        meta_features = self.meta_net(metadata)

        # Combine features
        combined = torch.cat([left_features, right_features, meta_features], dim=1)
        return self.classifier(combined)

    def _get_image_features(self, x):
        """Extract features from single image"""
        features = self.cnn.features(x)
        return nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

# Configuration


# Load model
@st.cache_resource
def load_model():
    model = OcularDiseaseModel()
    model.load_state_dict(torch.load('saved_models/best_model.pth', map_location=CFG['device']))
    model.to(CFG['device'])
    model.eval()
    return model

model = load_model()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((CFG['img_size'], CFG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Title
st.title("Ocular Disease Classification with Grad-CAM ")

# Upload Images and Metadata
left_img_file = st.file_uploader("Upload Left Eye Image", type=['jpg', 'png'])
right_img_file = st.file_uploader("Upload Right Eye Image", type=['jpg', 'png'])

age = st.slider("Patient Age", 0, 100, 50)
sex = st.selectbox("Patient Sex", ["Male", "Female", "Unknown"])

if st.button("Predict"):
    if left_img_file and right_img_file:
        # Preprocess images
        left_img = Image.open(left_img_file).convert('RGB')
        right_img = Image.open(right_img_file).convert('RGB')
        l_img_tensor = transform(left_img).unsqueeze(0).to(CFG['device'])
        r_img_tensor = transform(right_img).unsqueeze(0).to(CFG['device'])

        # Encode metadata
        sex_dict = {"Male": 0, "Female": 1, "Unknown": 2}
        metadata = torch.tensor([[age / 100, sex_dict[sex]]], dtype=torch.float32).to(CFG['device'])

        with torch.set_grad_enabled(True):
            outputs = model(l_img_tensor, r_img_tensor, metadata)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()[0]
            preds = [int(p >= 0.5) for p in probs]

        disease_labels = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
        st.subheader("Prediction")
        for label, p in zip(disease_labels, preds):
            st.write(f"{label}: {'Yes' if p else 'No'}")

        # Grad-CAM
        cam_extractor = GradCAM(model.cnn, target_layer="features.8")
        pred_class = torch.argmax(outputs[0]).item()

        activation_map1 = cam_extractor(pred_class, model.cnn.features(l_img_tensor))[0].cpu()
        result1 = overlay_mask(to_pil_image(l_img_tensor.squeeze(0).cpu()), to_pil_image(activation_map1, mode='F'))

        activation_map2 = cam_extractor(pred_class, model.cnn.features(r_img_tensor))[0].cpu()
        result2 = overlay_mask(to_pil_image(r_img_tensor.squeeze(0).cpu()), to_pil_image(activation_map2, mode='F'))

        st.subheader("Grad-CAM Visualization")
        col1, col2 = st.columns(2)
        with col1:
            st.image(result1, caption="Left Eye CAM")
        with col2:
            st.image(result2, caption="Right Eye CAM")

    else:
        st.warning("Please upload both left and right eye images.")
