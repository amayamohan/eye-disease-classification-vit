import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.vision_transformer import VisionTransformer
from PIL import Image

# ---------------------------
# Net class
# ---------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.base = vit_b_16(weights=weights)
        in_features = self.base.heads.head.in_features
        self.base.heads.head = nn.Linear(in_features, 4)

    def forward(self, x):
        return self.base(x)

# ---------------------------
# Allow required classes for torch.load
# ---------------------------
import torch.serialization
torch.serialization.add_safe_globals([Net, VisionTransformer])

# ---------------------------
# Load full model
# ---------------------------
model = torch.load("best_model.pth", map_location="cpu", weights_only=False)
model.eval()

# ---------------------------
# Class labels
# ---------------------------
id2label = {
    0: "glaucoma",
    1: "normal",
    2: "diabetic_retinopathy",
    3: "cataract"
}

# ---------------------------
# Transforms
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# Prediction function
# ---------------------------
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img_t)
        probs = torch.softmax(out, dim=1).squeeze()

    c = torch.argmax(probs).item()
    conf = probs[c].item() * 100

    print(f"\nPredicted Class: {id2label[c]}")
    print(f"Confidence: {conf:.2f}%")

predict("images/test1.png")
