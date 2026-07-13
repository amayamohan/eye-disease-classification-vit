import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import torch.serialization

# ---------------------------------------------------
# 1. MODEL DEFINITION (MUST MATCH TRAINING)
# ---------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.base = vit_b_16(weights=weights)

        in_features = self.base.heads.head.in_features
        self.base.heads.head = nn.Linear(in_features, 4)

    def forward(self, x):
        return self.base(x)


# ---------------------------------------------------
# 2. SAFE LOAD
# ---------------------------------------------------
torch.serialization.add_safe_globals([Net])

MODEL_PATH = "best_model.pth"
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval()

st.set_page_config(
    page_title="Eye Disease Classifier",
    layout="wide"
)

# ---------------------------------------------------
# 3. LABELS + INFO
# ---------------------------------------------------
ID2LABEL = {
    0: "glaucoma",
    1: "normal",
    2: "diabetic_retinopathy",
    3: "cataract"
}

DISEASE_INFO = {
    "glaucoma": "Glaucoma damages the optic nerve due to high eye pressure and can cause permanent vision loss.",
    "normal": "The eye appears healthy with no visible abnormalities in the retina or optic disc.",
    "diabetic_retinopathy": "Diabetic Retinopathy affects retinal blood vessels and is caused by prolonged diabetes.",
    "cataract": "Cataract causes clouding of the eye lens, leading to blurred or dim vision."
}


# ---------------------------------------------------
# 4. TRANSFORMS
# ---------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ---------------------------------------------------
# 5. PREDICTION FUNCTION
# ---------------------------------------------------
def predict(img):
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)[0]

    pred_id = torch.argmax(probs).item()
    confidence = probs[pred_id].item()

    return ID2LABEL[pred_id], confidence


# ===================================================
# 🔵 SIDEBAR NAVIGATION (NEW)
# ===================================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏥 Home",
        "👁️ Eye Diseases",
        "🔍 Predict Disease",
        "👩‍⚕️ Find Eye Doctor"
    ]
)


# ===================================================
# 🏠 PAGE 1: HOME (NEW)
# ===================================================
if page == "🏥 Home":

    col_img, col_text = st.columns([1, 4])

    with col_img:
        st.image("images/Ai_image.png", width=260)

    with col_text:
        st.title(" **Eye Health Awareness**")

        st.markdown("""
        ### Eye diseases can cause **irreversible vision loss** if not detected early.
        ### This application creates awareness and provides AI-based eye disease detection using retinal fundus images.
        """)

        st.markdown("## 👁️ Why Eye Care Matters")

        st.markdown("""
        - 👁️ **Silent Diseases**  
        Many eye diseases show no early symptoms.

       - 🛑 **Prevent Blindness**  
        Early detection can prevent permanent vision loss.

       - 🩺 **Regular Screening**  
       Routine eye check-ups are essential for maintaining eye health.
""")

        

    




# ===================================================
# 👁 PAGE 2: EYE DISEASES (NEW)
# ===================================================
elif page == "👁️ Eye Diseases":
    st.title(" 👁️ Common Eye Diseases")
    st.write("""
    Eye diseases are among the leading causes of vision impairment worldwide.
    Early detection and timely treatment can significantly reduce the risk
    of permanent vision loss. """)
    st.write("""Below are some of the most common eye diseases
    detected using retinal fundus imaging.
    """)

    st.markdown("### 🔵 Glaucoma")
    st.markdown("""
    - Caused by damage to the **optic nerve**, often due to increased eye pressure  
    - Progresses slowly and may show **no early symptoms**  
    - Can lead to **irreversible blindness** if untreated  
    """)

    st.markdown("### 🟠 Diabetic Retinopathy")
    st.markdown("""
    - Occurs due to **long-term diabetes**  
    - Damages **retinal blood vessels**  
    - Can cause blurred vision, dark spots, or vision loss  
    - Early stages may be **asymptomatic**  
    """)
    

    st.markdown("### 🔴 Cataract")
    st.markdown("""
    - Causes **clouding of the eye lens**  
    - Leads to blurred or dim vision  
    - Common in older adults  
    - Treatable through **surgical intervention**  
    """)
    

    

# ===================================================
# 🔍 PAGE 3: PREDICT DISEASE (YOUR EXISTING CODE)
# ===================================================
elif page == "🔍 Predict Disease":
    st.title("🔍 Eye Disease Classifier")
    st.caption("AI-powered retinal fundus image analysis")

    uploaded = st.file_uploader(
        "Upload Fundus Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Uploaded Fundus Image", width=300)

        with col2:
            if st.button("🔍 Predict"):
                label, conf = predict(img)

                if label == "normal":
                    st.markdown("### 🟢 **NORMAL**")
                else:
                    st.markdown(
                        f"### 🔴 **{label.replace('_',' ').title()}**"
                    )

                st.markdown(f"### 📊 Confidence Score: **{conf:.2f}**")
                st.progress(conf)

                if conf >= 0.85:
                    st.success("High confidence prediction")
                elif conf >= 0.60:
                    st.warning("Moderate confidence prediction")
                else:
                    st.error("Low confidence – further examination required")

                st.divider()

                st.subheader("👁️ About this disease")
                st.write(DISEASE_INFO[label])

                st.subheader("📝 Recommended Action")
                if label != "normal":
                    st.write("• Consult an ophthalmologist")
                    st.write("• Schedule retinal screening")
                else:
                    st.write("• No abnormality detected")
                    st.write("• Maintain regular eye check-ups")

               

# ===================================================
# 🏥 PAGE 4: FIND EYE DOCTOR (NEW)
# ===================================================
elif page == "👩‍⚕️ Find Eye Doctor":
    st.title("👩‍⚕️ Find an Eye Specialist")

    st.write("""
    If the AI model detects an eye disease, it is strongly recommended
    to consult a certified eye specialist (ophthalmologist).
    """)

    st.subheader("⏰ When should you seek medical help?")
    st.write("""
    • Blurred or sudden vision loss  
    • Eye pain or redness  
    • Floaters or dark spots in vision  
    • Difficulty seeing at night  
    """)
    
    st.markdown("<h1 style='text-align:center;'>🏥 Find an Eye Specialist</h1>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center;'>Click the button below to consult an eye specialist.</p>",
        unsafe_allow_html=True
    )

    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.button("👨‍⚕️ Find Eye Specialist")
    st.markdown("</div>", unsafe_allow_html=True)


    

    st.warning(
        "⚠️ This application provides guidance only. "
        "Always consult a qualified eye specialist for medical diagnosis."
    )
