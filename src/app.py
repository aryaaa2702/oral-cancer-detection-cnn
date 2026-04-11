import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import plotly.graph_objects as go
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Explainable AI Framework for Oral Cancer Histopathology Grading Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = ["Normal", "OSCC"]

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "best_oral_cancer_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# SESSION STATE
# -----------------------------
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if "results" not in st.session_state:
    st.session_state.results = None

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>Explainable AI Framework for Oral Cancer Histopathology Grading Support</h1>
    <p style='text-align: center; font-size:18px;'>
    Deep learning-based oral tissue classification with visual explainability, texture-driven histopathology insights, and grading-oriented diagnostic support.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)
# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("Upload Case")

    uploaded_file = st.file_uploader(
        "Choose a histopathology image",
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )

    st.markdown("---")
    st.markdown("### Project Info")
    st.info("Upload a histopathology image to generate prediction, confidence, and visual explanation.")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------------
# GRAD-CAM HOOKS
# -----------------------------
gradients = []
activations = []
feature_maps = []

def save_gradient(module, grad_input, grad_output):
    gradients.clear()
    gradients.append(grad_output[0])

def save_activation(module, input, output):
    activations.clear()
    activations.append(output)

def save_feature_maps(module, input, output):
    feature_maps.clear()
    feature_maps.append(output.detach())

target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(save_activation)
target_layer.register_full_backward_hook(save_gradient)

feature_layer = model.layer1[0].conv1
feature_layer.register_forward_hook(save_feature_maps)

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess_image(uploaded_image):
    try:
        image = Image.open(uploaded_image)
        image = image.convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("The uploaded file could not be read as an image.")
    except Exception as e:
        raise ValueError(f"Image loading failed: {str(e)}")

    original_img = np.array(image)

    if original_img is None or len(original_img.shape) != 3:
        raise ValueError("Invalid image format after conversion.")

    img_resized = cv2.resize(original_img, (224, 224))
    img = img_resized.astype("float32") / 255.0

    preprocessed_display = img.copy()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(3):
        img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]

    input_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    return original_img, preprocessed_display, input_tensor

# -----------------------------
# GRAD-CAM
# -----------------------------
def generate_gradcam(original_img, input_tensor):
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()

    pred_class = np.argmax(probs)
    pred_confidence = probs[pred_class] * 100
    normal_prob = probs[0] * 100
    oscc_prob = probs[1] * 100

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    if cam.max() != 0:
        cam = cam / cam.max()

    cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return pred_class, pred_confidence, normal_prob, oscc_prob, heatmap, overlay

# -----------------------------
# MANUAL INTEGRATED GRADIENTS
# -----------------------------
def generate_integrated_gradients(input_tensor, target_class, steps=50):
    model.eval()

    baseline = torch.zeros_like(input_tensor).to(device)

    scaled_inputs = [
        baseline + (float(i) / steps) * (input_tensor - baseline)
        for i in range(steps + 1)
    ]

    gradients_list = []

    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.clone().detach().requires_grad_(True)

        output = model(scaled_input)
        target_score = output[0, target_class]

        model.zero_grad()
        target_score.backward()

        gradients_list.append(scaled_input.grad.detach().cpu().numpy())

    avg_gradients = np.mean(np.array(gradients_list), axis=0)

    integrated_grads = (input_tensor.detach().cpu().numpy() - baseline.detach().cpu().numpy()) * avg_gradients

    attr = integrated_grads.squeeze()
    attr = np.transpose(attr, (1, 2, 0))

    # Aggregate channels
    attr = np.sum(np.abs(attr), axis=2)

    # Normalize
    attr = attr - attr.min()
    if attr.max() != 0:
        attr = attr / attr.max()

    attr_map = np.uint8(255 * attr)
    attr_map = cv2.applyColorMap(attr_map, cv2.COLORMAP_VIRIDIS)
    attr_map = cv2.cvtColor(attr_map, cv2.COLOR_BGR2RGB)

    return attr_map

# -----------------------------
# FEATURE MAP VISUALIZATION
# -----------------------------
def get_feature_map_images():
    if len(feature_maps) == 0:
        return []

    fmap = feature_maps[0].cpu().numpy()[0]   # shape: [C, H, W]

    # Compute average activation per channel
    channel_strengths = np.mean(fmap, axis=(1, 2))

    # Get indices of top 6 activated channels
    top_indices = np.argsort(channel_strengths)[-6:][::-1]

    images = []

    for idx in top_indices:
        channel = fmap[idx]

        # Normalize channel
        channel = channel - channel.min()
        if channel.max() != 0:
            channel = channel / channel.max()

        channel_img = np.uint8(255 * channel)
        images.append((idx, channel_img))

    return images

# -----------------------------
# GLCM TEXTURE ANALYSIS
# -----------------------------
def compute_glcm_features(original_img):
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (224, 224))

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    return {
        "contrast": float(contrast),
        "homogeneity": float(homogeneity),
        "energy": float(energy),
        "correlation": float(correlation)
    }

def interpret_glcm(glcm_features):
    contrast = glcm_features["contrast"]
    homogeneity = glcm_features["homogeneity"]
    energy = glcm_features["energy"]

    interpretation = []

    if contrast > 500:
        interpretation.append("high local texture variation")
    elif contrast > 200:
        interpretation.append("moderate texture variation")
    else:
        interpretation.append("relatively smooth local texture")

    if homogeneity < 0.4:
        interpretation.append("reduced tissue uniformity")
    else:
        interpretation.append("moderate structural uniformity")

    if energy < 0.1:
        interpretation.append("less repetitive tissue organization")
    else:
        interpretation.append("some repetitive structural patterns")

    return ", ".join(interpretation).capitalize() + "."

# -----------------------------
# GRADING SUPPORT MODULE
# -----------------------------
def compute_grading_support(oscc_prob, glcm_features):
    contrast = glcm_features["contrast"]
    homogeneity = glcm_features["homogeneity"]

    score = 0

    # Contribution from model confidence
    if oscc_prob > 0.8:
        score += 2
    elif oscc_prob > 0.5:
        score += 1

    # Contribution from texture irregularity
    if contrast > 300:
        score += 2
    elif contrast > 150:
        score += 1

    if homogeneity < 0.4:
        score += 2
    elif homogeneity < 0.6:
        score += 1

    # Final level
    if score >= 5:
        level = "High Suspicion Pattern"
    elif score >= 3:
        level = "Moderate Suspicion Pattern"
    else:
        level = "Low Suspicion Pattern"

    return level, score

# -----------------------------
# CONFIDENCE DROP / FAITHFULNESS TEST
# -----------------------------
# def compute_confidence_drop(original_img, input_tensor, heatmap, pred_class, threshold=180):
#     # Resize heatmap to match model input size
#     heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
#     heatmap_resized = cv2.resize(heatmap_gray, (224, 224))

#     # Create binary mask of important regions
#     _, mask = cv2.threshold(heatmap_resized, threshold, 255, cv2.THRESH_BINARY)
#     mask = mask / 255.0

#     # Convert original image to model-size image
#     img_resized = cv2.resize(original_img, (224, 224)).astype("float32") / 255.0

#     # Mask out important regions (black them out)
#     masked_img = img_resized.copy()
#     for c in range(3):
#         masked_img[:, :, c] = masked_img[:, :, c] * (1 - mask)

#     # Save masked display image
#     masked_display = np.uint8(masked_img * 255)

#     # Normalize again for model input
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]

#     for i in range(3):
#         masked_img[:, :, i] = (masked_img[:, :, i] - mean[i]) / std[i]

#     masked_tensor = torch.tensor(masked_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

#     # Predict on masked image
#     with torch.no_grad():
#         output = model(masked_tensor)
#         probs = torch.softmax(output, dim=1)[0].cpu().numpy()

#     masked_confidence = probs[pred_class] * 100

#     # Original confidence
#     with torch.no_grad():
#         original_output = model(input_tensor)
#         original_probs = torch.softmax(original_output, dim=1)[0].cpu().numpy()

#     original_confidence = original_probs[pred_class] * 100

#     confidence_drop = original_confidence - masked_confidence

#     return original_confidence, masked_confidence, confidence_drop, masked_display

# -----------------------------
# LOGIC FUNCTIONS
# -----------------------------
def get_suspicion_level(pred_class, oscc_prob):
    if pred_class == 0:
        return "Low"
    if oscc_prob < 60:
        return "Low"
    elif oscc_prob < 85:
        return "Moderate"
    else:
        return "High"

def get_interpretation(pred_class, pred_confidence):
    if pred_class == 1:
        if pred_confidence >= 85:
            return "Strong indication of suspicious malignant tissue patterns."
        elif pred_confidence >= 60:
            return "Moderate confidence. Some suspicious patterns detected."
        else:
            return "Low confidence. Uncertain prediction."
    else:
        if pred_confidence >= 85:
            return "No strong malignant patterns detected."
        else:
            return "Leaning toward normal but requires expert review."

def get_recommendation(pred_class, pred_confidence):
    if pred_class == 1:
        return "Expert histopathological review recommended."
    else:
        return "Low malignancy suspicion but expert review is advised."

# -----------------------------
# GAUGE
# -----------------------------
def plot_probability_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Cancer Probability (%)"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    return fig

# -----------------------------
# MAIN FLOW
# -----------------------------
if uploaded_file is None:
    st.session_state.results_ready = False
    st.session_state.results = None
    st.info("Upload an image from the sidebar to begin.")

else:
    try:
        original_img, preprocessed_display, input_tensor = preprocess_image(uploaded_file)

        st.subheader("Uploaded Image")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(original_img, width=300)

        with col2:
            st.markdown("### Ready for Analysis")
            st.write("Click the button below to analyze the uploaded histopathology image.")
            analyze_clicked = st.button("Analyze Case", use_container_width=True)

        if analyze_clicked:
            with st.spinner("Analyzing..."):
                pred_class, pred_confidence, normal_prob, oscc_prob, heatmap, overlay = generate_gradcam(original_img, input_tensor)
                ig_map = generate_integrated_gradients(input_tensor, pred_class)
                feature_map_images = get_feature_map_images()

                glcm_features = compute_glcm_features(original_img)
                glcm_interpretation = interpret_glcm(glcm_features)

                grading_level, grading_score = compute_grading_support(oscc_prob, glcm_features)

                # original_conf, masked_conf, confidence_drop, masked_display = compute_confidence_drop(
                    # original_img, input_tensor, heatmap, pred_class)
            suspicion = get_suspicion_level(pred_class, oscc_prob)
            interpretation = get_interpretation(pred_class, pred_confidence)
            recommendation = get_recommendation(pred_class, pred_confidence)

            st.session_state.results = {
                "original_img": original_img,
                "preprocessed_display": preprocessed_display,
                "pred_class": pred_class,
                "pred_confidence": pred_confidence,
                "normal_prob": normal_prob,
                "oscc_prob": oscc_prob,
                "heatmap": heatmap,
                "overlay": overlay,
                "ig_map": ig_map,
                "feature_map_images": feature_map_images,
                # "original_conf": original_conf,
                # "masked_conf": masked_conf,
                # "confidence_drop": confidence_drop,
                # "masked_display": masked_display,
                "glcm_features": glcm_features,
                "glcm_interpretation": glcm_interpretation,
                "grading_level": grading_level,
                "grading_score": grading_score,
                "suspicion": suspicion,
                "interpretation": interpretation,
                "recommendation": recommendation
            }

            st.session_state.results_ready = True

        if st.session_state.results_ready and st.session_state.results is not None:
            results = st.session_state.results

            st.subheader("Analysis Results")

            tab1, tab2, tab3 = st.tabs(["Prediction", "Explainability", "Interpretation"])

            # -----------------------------
            # TAB 1: PREDICTION
            # -----------------------------
            with tab1:
                col1, col2, col3 = st.columns(3)
                col1.metric("Class", CLASS_NAMES[results["pred_class"]])
                col2.metric("Confidence", f'{results["pred_confidence"]:.2f}%')
                col3.metric("Suspicion", results["suspicion"])

                st.plotly_chart(plot_probability_gauge(results["oscc_prob"]), use_container_width=True)

                st.markdown("### Probability Breakdown")
                col1, col2 = st.columns(2)
                col1.metric("Normal Probability", f'{results["normal_prob"]:.2f}%')
                col2.metric("OSCC Probability", f'{results["oscc_prob"]:.2f}%')

            # -----------------------------
            # TAB 2: EXPLAINABILITY
            # -----------------------------
            with tab2:
                st.markdown("### Visual Explainability")

                col1, col2, col3 = st.columns(3)
                with col1:
                    col1.image(results["original_img"], width=250)
                    st.caption("Original uploaded histapathology image.")

                with col2:
                    col2.image(results["preprocessed_display"], width=250)
                    st.caption("Preprocessed image used for model input.")

                with col3:
                    col3.image(results["overlay"], width=250)
                    st.caption("Grad-CAM overlay showing highlighted regions on the original image.")

                st.markdown("### Heatmap-Based Explanations")

                col4, col5 = st.columns(2)
                with col4:
                    st.image(results["heatmap"], caption="Grad-CAM Heatmap", width=350)
                    st.caption("Highlights the most important tissue regions used by the model for prediction.")
                with col5:
                    st.image(results["ig_map"], caption="Integrated Gradients Map", width=350)
                    st.caption("Highlights the most important tissue regions used by the model for prediction.")

                st.markdown("### CNN Feature Maps")

                fmap_cols = st.columns(3)
                for i, (fmap_idx, fmap_img) in enumerate(results["feature_map_images"]):
                    with fmap_cols[i % 3]:
                        st.image(fmap_img, caption=f"Activated Filter {fmap_idx}", width=220)

                st.caption("Feature maps show internal CNN filter responses, capturing texture and structural patterns from the tissue image.")

            # -----------------------------
            # TAB 3: INTERPRETATION
            # -----------------------------
            with tab3:
                st.markdown("### AI Interpretation")
                st.info(results["interpretation"])

                st.markdown("### Recommendation")
                st.warning(results["recommendation"])

                # st.markdown("### Explainability Validation")

                # col1, col2, col3 = st.columns(3)
                # col1.metric("Original Confidence", f'{results["original_conf"]:.2f}%')
                # col2.metric("Masked Confidence", f'{results["masked_conf"]:.2f}%')
                # col3.metric("Confidence Drop", f'{results["confidence_drop"]:.2f}%')

                # st.image(results["masked_display"], caption="Masked Important Region Image", width=300)

                # if results["confidence_drop"] > 20:
                #     st.success("The model confidence dropped significantly after masking the highlighted region, suggesting that the explanation is meaningful.")
                # elif results["confidence_drop"] > 5:
                #     st.warning("The model confidence dropped moderately after masking the highlighted region, indicating partial explanation relevance.")
                # else:
                #     st.info("The confidence drop was small, suggesting that the highlighted region may not fully explain the prediction.")
                
                st.markdown("### Histopathology Grading Support")

                if results["grading_level"] == "High Suspicion Pattern":
                    st.error(results["grading_level"])
                elif results["grading_level"] == "Moderate Suspicion Pattern":
                    st.warning(results["grading_level"])
                else:
                    st.success(results["grading_level"])

                st.metric("Grading Support Score", results["grading_score"])
                st.caption("This grading support level is derived from model confidence and tissue texture characteristics. It represents a severity-oriented interpretation, not a clinical grade.")
                
                st.markdown("### Texture Analysis (GLCM)")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Contrast", f'{results["glcm_features"]["contrast"]:.2f}')
                col2.metric("Homogeneity", f'{results["glcm_features"]["homogeneity"]:.4f}')
                col3.metric("Energy", f'{results["glcm_features"]["energy"]:.4f}')
                col4.metric("Correlation", f'{results["glcm_features"]["correlation"]:.4f}')

                st.caption("These classical texture descriptors provide supporting evidence about tissue structural organization and local image irregularity.")

                st.info(results["glcm_interpretation"])

                with st.expander("What do these texture metrics mean?"):
                    st.markdown("""
                **Contrast**  
                Shows how much local intensity variation exists in the tissue image.  
                Higher values may indicate more structural irregularity or heterogeneous tissue patterns.

                **Homogeneity**  
                Shows how uniform and smooth the tissue texture appears.  
                Lower values may suggest reduced structural consistency.

                **Energy**  
                Measures how repetitive or orderly the tissue texture is.  
                Lower values often indicate less organized visual patterns.

                **Correlation**  
                Measures how strongly neighboring pixel intensities are related.  
                It reflects the local structural consistency of the tissue image.

                **Important Note:**  
                These texture features do **not diagnose cancer directly**.  
                They provide **supporting image-based evidence** about tissue organization and irregularity.
                """)

                st.caption("For educational and research purposes only.")

    except Exception as e:
        st.error(f"Error while processing image: {str(e)}")