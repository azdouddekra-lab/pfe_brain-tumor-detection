import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
 
# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)
 
# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; }
 
    .app-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #2d3561;
    }
    .app-header h1 { color: #e2e8f0; font-size: 2rem; font-weight: 700; margin: 0; }
    .app-header p  { color: #94a3b8; margin: 0.4rem 0 0 0; font-size: 0.95rem; }
 
    .card {
        background: #1e2130;
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card h3 { color: #e2e8f0; font-size: 1rem; font-weight: 600; margin: 0 0 1rem 0; }
 
    .result-glioma     { background:#3b0000; border:2px solid #ef4444; color:#fca5a5; }
    .result-meningioma { background:#1c1c00; border:2px solid #eab308; color:#fde047; }
    .result-pituitary  { background:#001c3b; border:2px solid #3b82f6; color:#93c5fd; }
    .result-no_tumor   { background:#00200e; border:2px solid #22c55e; color:#86efac; }
    .result-box {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1rem 0;
    }
 
    .conf-bar-bg {
        background: #2d3561;
        border-radius: 999px;
        height: 10px;
        margin: 4px 0 10px 0;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 10px;
        border-radius: 999px;
        transition: width 0.5s ease;
    }
 
    .stSelectbox label { color: #94a3b8 !important; font-size: 0.85rem; }
 
    .metric-row { display: flex; gap: 1rem; margin-top: 0.5rem; }
    .metric-item {
        flex: 1;
        background: #252b3b;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-val { font-size: 1.4rem; font-weight: 700; color: #60a5fa; }
    .metric-lbl { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
 
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .badge-blue   { background:#1e3a5f; color:#93c5fd; }
    .badge-green  { background:#14532d; color:#86efac; }
    .badge-purple { background:#3b0764; color:#d8b4fe; }
 
    [data-testid="stFileUploader"] {
        background: #1e2130;
        border: 2px dashed #2d3561;
        border-radius: 12px;
        padding: 1rem;
    }
 
    section[data-testid="stSidebar"] { background: #13161f; }
    section[data-testid="stSidebar"] .stMarkdown p { color: #94a3b8; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)
 
# ─── Constants ────────────────────────────────────────────────────────────────
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
 
CLASS_INFO = {
    'glioma': {
        'emoji': '🔴',
        'label': 'Gliome',
        'desc': 'Tumeur maligne des cellules gliales. Représente ~30% des tumeurs cérébrales.',
        'color': '#ef4444',
        'css': 'result-glioma'
    },
    'meningioma': {
        'emoji': '🟡',
        'label': 'Méningiome',
        'desc': 'Tumeur des méninges, souvent bénigne et à croissance lente.',
        'color': '#eab308',
        'css': 'result-meningioma'
    },
    'pituitary': {
        'emoji': '🔵',
        'label': 'Tumeur hypophysaire',
        'desc': "Tumeur de la glande pituitaire, généralement bénigne et traitable.",
        'color': '#3b82f6',
        'css': 'result-pituitary'
    },
    'no_tumor': {
        'emoji': '🟢',
        'label': 'Aucune tumeur',
        'desc': "Aucune anomalie tumorale détectée sur l'IRM analysée.",
        'color': '#22c55e',
        'css': 'result-no_tumor'
    }
}
 
# IDs Google Drive dyalk
DRIVE_IDS = {
    'MobileNetV2 Fine Tuning (Recommandé)': '1uGofq96oRk6E5_9vYhVqejarUijrpQMc',
    'MobileNetV2 Transfer Learning':         '1dHhJq3x5yyCMSJThv7mVF5OBd4Nqjruc',
    'VGG16 Fine Tuning':                     '1AZQTwYCb3w3p3YJzKW_WWl91UmmzKgeU',
    'VGG16 Transfer Learning':               '1CNa4-I_sVUdlj4npBcPXNbWrvHwmT-VI',
    'CNN Custom':                            '1jgNpnvfvQMyyjjLI5mrgyhrdN7E4e8If',
}
 
MODEL_INFO = {
    'MobileNetV2 Fine Tuning (Recommandé)': {
        'file': 'mobilenet_ft.h5',
        'acc': '96.00%',
        'badge': 'badge-green',
        'badge_txt': 'Meilleur',
        'desc': 'Transfer Learning + Fine Tuning sur 30 couches — léger et précis'
    },
    'MobileNetV2 Transfer Learning': {
        'file': 'mobilenet_tl.h5',
        'acc': '88.50%',
        'badge': 'badge-blue',
        'badge_txt': 'Rapide',
        'desc': 'MobileNetV2 pré-entraîné sur ImageNet — base gelée'
    },
    'VGG16 Fine Tuning': {
        'file': 'vgg16_ft.h5',
        'acc': '96.38%',
        'badge': 'badge-blue',
        'badge_txt': 'Robuste',
        'desc': 'Architecture VGG16 fine-tunée sur les dernières couches'
    },
    'VGG16 Transfer Learning': {
        'file': 'vgg16_tl.h5',
        'acc': '89.96%',
        'badge': 'badge-purple',
        'badge_txt': 'Classique',
        'desc': 'VGG16 pré-entraîné — base gelée'
    },
    'CNN Custom': {
        'file': 'cnn_custom.h5',
        'acc': '84.04%',
        'badge': 'badge-purple',
        'badge_txt': 'Baseline',
        'desc': 'CNN entraîné from scratch — architecture personnalisée'
    },
}
 
# ─── Model loading ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    try:
        import tensorflow as tf
        model_file = MODEL_INFO[model_name]['file']
 
        # Télécharger depuis Drive si pas déjà présent
        if not os.path.exists(model_file):
            drive_id = DRIVE_IDS[model_name]
            url = f"https://drive.google.com/uc?id={drive_id}"
            with st.spinner(f"⏳ Téléchargement du modèle..."):
                gdown.download(url, model_file, quiet=False)
 
        model = tf.keras.models.load_model(model_file)
        return model
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle : {e}")
        return None
 
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)
 
def predict(model, img_array):
    preds = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASSES[idx], preds
 
# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Brain Tumor Detection")
    st.markdown("---")
    st.markdown("**Projet :** PFE Deep Learning")
    st.markdown("**Dataset :** Brain Tumor MRI (Kaggle)")
    st.markdown("**Images d'entraînement :** 16 000")
    st.markdown("**Classes :**")
    for cls, info in CLASS_INFO.items():
        st.markdown(
            f"{info['emoji']} **{info['label']}**  \n"
            f"<small style='color:#64748b'>{info['desc']}</small>",
            unsafe_allow_html=True
        )
    st.markdown("---")
    st.markdown("**Architectures :** CNN · MobileNetV2 · VGG16")
    st.markdown("**Méthodes :** Transfer Learning · Fine Tuning")
    st.markdown("**Input size :** 224 × 224 px")
 
# ─── Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🧠 Détection des Tumeurs Cérébrales</h1>
    <p>PFE · Deep Learning · CNN + Transfer Learning + Fine Tuning &nbsp;|&nbsp; Classification IRM en 4 classes</p>
</div>
""", unsafe_allow_html=True)
 
# ─── Layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")
 
with col_left:
    # Model selector
    st.markdown('<div class="card"><h3>⚙️ Choix du Modèle</h3>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Modèle",
        list(MODEL_INFO.keys()),
        label_visibility="collapsed"
    )
    minfo = MODEL_INFO[selected_model]
    st.markdown(f"""
        <span class="badge {minfo['badge']}">{minfo['badge_txt']}</span>
        <small style="color:#94a3b8">{minfo['desc']}</small>
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-val">{minfo['acc']}</div>
                <div class="metric-lbl">Test Accuracy</div>
            </div>
            <div class="metric-item">
                <div class="metric-val">4</div>
                <div class="metric-lbl">Classes</div>
            </div>
            <div class="metric-item">
                <div class="metric-val">224px</div>
                <div class="metric-lbl">Input size</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
    # Upload
    st.markdown('<div class="card"><h3>📂 Image IRM</h3>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Glisse ou clique pour uploader une IRM",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
 
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="IRM uploadée", use_column_width=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:2rem; color:#4a5568;">
            <div style="font-size:3rem;">🩻</div>
            <div style="font-size:0.9rem; margin-top:0.5rem;">Aucune image sélectionnée</div>
            <div style="font-size:0.75rem; color:#374151;">Formats acceptés : JPG, PNG</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
with col_right:
    st.markdown('<div class="card"><h3>📊 Résultat de l\'analyse</h3>', unsafe_allow_html=True)
 
    if uploaded:
        with st.spinner("🔬 Analyse en cours..."):
            model = load_model(selected_model)
 
        if model is not None:
            img_array = preprocess_image(img)
            pred_class, probas = predict(model, img_array)
 
            info = CLASS_INFO[pred_class]
 
            # Result badge
            st.markdown(f"""
            <div class="result-box {info['css']}">
                {info['emoji']}  {info['label'].upper()}
            </div>
            """, unsafe_allow_html=True)
 
            st.markdown(
                f"<p style='color:#94a3b8; font-size:0.85rem; text-align:center'>{info['desc']}</p>",
                unsafe_allow_html=True
            )
 
            st.markdown("**Confiance par classe :**")
            for i, cls in enumerate(CLASSES):
                pct = float(probas[i]) * 100
                c_info = CLASS_INFO[cls]
                is_top = (cls == pred_class)
                label_style = "font-weight:700; color:#e2e8f0;" if is_top else "color:#94a3b8;"
                st.markdown(f"""
                <div style="margin-bottom:6px">
                    <div style="display:flex; justify-content:space-between; {label_style} font-size:0.85rem;">
                        <span>{c_info['emoji']} {c_info['label']}</span>
                        <span>{pct:.1f}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{pct:.1f}%; background:{c_info['color']};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
            # Confidence level
            top_conf = float(np.max(probas)) * 100
            if top_conf >= 90:
                level, level_color = "Très élevée ✅", "#22c55e"
            elif top_conf >= 70:
                level, level_color = "Élevée 🟡", "#eab308"
            else:
                level, level_color = "Faible ⚠️", "#ef4444"
 
            st.markdown(f"""
            <div style="margin-top:1rem; padding:0.8rem 1rem; background:#252b3b; border-radius:10px;">
                <span style="color:#64748b; font-size:0.8rem;">Confiance globale : </span>
                <span style="color:{level_color}; font-weight:700;">{level} ({top_conf:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)
 
    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 1rem; color:#4a5568;">
            <div style="font-size:3.5rem;">🔬</div>
            <div style="font-size:1rem; margin-top:1rem; color:#64748b;">
                Uploadez une image IRM<br>pour démarrer l'analyse
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown('</div>', unsafe_allow_html=True)
 
# ─── Disclaimer ───────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:1.5rem; padding:0.8rem 1.2rem; background:#1a1f2e;
     border-left:3px solid #3b82f6; border-radius:6px; font-size:0.8rem; color:#64748b;">
    ⚠️ <strong style="color:#94a3b8">Avertissement médical :</strong>
    Cette application est développée à des fins académiques (PFE).
    Elle ne remplace pas un diagnostic médical professionnel.
</div>
""", unsafe_allow_html=True)