import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# =========================
# Configuration page
# =========================
st.set_page_config(
    page_title="Évaluation du risque cardiovasculaire",
    layout="wide"
)

# =========================
# Charger le modèle
# =========================
model = joblib.load("chd_model.pkl")

# =========================
# HEADER AVEC IMAGE À GAUCHE
# =========================
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# =========================
# Charger l'image
# =========================
image = Image.open("medecine-globale.jpg")  # Nom exact du fichier

# Convertir l'image en base64 pour l'inclure dans le div
buffered = BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# =========================
# Header large avec image à gauche
# =========================
st.markdown(f"""
<div style="
    width:100%;
    padding:30px;
    border-radius:12px;
    background: linear-gradient(135deg, #2c3e50, #4b79a1);
    color:white;
    box-shadow:0px 5px 15px rgba(0,0,0,0.25);
    display:flex;
    align-items:center;
">
    <img src="data:image/png;base64,{img_str}" style="width:250px; margin-right:30px; border-radius:10px;"/>
    <div style="flex-grow:1;">
        <h1 style="margin:0; font-size:50px;">Évaluation du risque cardiovasculaire</h1>
        <p style="margin-top:8px; font-size:16px; opacity:0.9;">
           Outil d’évaluation du risque cardiovasculaire basé sur des indicateurs cliniques et un modèle prédictif avancé.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# Formulaire patient
# =========================
st.subheader("Données du patient")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Âge (années)", 20, 90, 30)
    sbp = st.number_input("Pression systolique (mmHg)", 90, 200, 120)
    chol = st.number_input("Cholestérol total (mg/dL)", 120, 350, 200)

with col2:
    ldl = st.number_input("LDL cholestérol (mg/dL)", 50, 250, 130)
    adiposity = st.number_input("Adiposité (%)", 10.0, 45.0, 20.0)
    bmi = st.number_input("Indice de masse corporelle (BMI)", 15.0, 50.0, 25.0)

with col3:
    tobacco = st.number_input("Tabac", 0.0, 40.0, 0.0)
    alcohol = st.number_input("Alcool", 0.0, 50.0, 0.0)
    typea = st.number_input("Stress (Type A)", 0, 100, 50)
    famhist = st.selectbox("Antécédents familiaux", ["Absent", "Present"])

# Préparer les données pour le modèle
data = pd.DataFrame([{
    "sbp": sbp,
    "tobacco": tobacco,
    "ldl": ldl,
    "adiposity": adiposity,
    "typea": typea,
    "obesity": bmi,
    "alcohol": alcohol,
    "age": age,
    "chol": chol,
    "famhist": famhist
}])

st.divider()

# =========================
# Bouton prédiction
# =========================
if st.button("Lancer l’évaluation du risque", use_container_width=True):

    pred = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]

    # Définir couleurs et labels dynamiques
    if proba < 0.30:
        color_metric = "green"
        risk_label = "Risque faible"
        alert_label = "Standard"
    elif proba < 0.60:
        color_metric = "orange"
        risk_label = "Risque modéré"
        alert_label = "Surveillance recommandée"
    else:
        color_metric = "red"
        risk_label = "Risque élevé"
        alert_label = "Surveillance renforcée"

    # =========================
    # Synthèse du résultat
    # =========================
    st.subheader("Synthèse du résultat")
    c1, c2, c3 = st.columns(3)

    # Probabilité détaillée à 2 décimales
    c1.metric(
        label="Probabilité estimée",
        value=f"{proba*100:.2f}%",
        delta=None
    )

    # Classification dynamique
    c2.markdown(
        f"<h3 style='color:{color_metric}; margin:0'>{risk_label}</h3>",
        unsafe_allow_html=True
    )

    # Niveau d’alerte dynamique
    c3.markdown(
        f"<h3 style='color:{color_metric}; margin:0'>{alert_label}</h3>",
        unsafe_allow_html=True
    )

    st.divider()

    # =========================
    # Pie chart du risque
    # =========================
    fig = go.Figure(go.Pie(
    labels=["Risque cardiovasculaire", "Risque faible"],
    values=[proba, 1 - proba],
    hole=0.6,
    marker=dict(colors=[color_metric, "#ecf0f1"])
))


    fig.update_layout(
        title="Distribution du risque",
        title_x=0.5,
        margin=dict(t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # =========================
    # Onglets
    # =========================
    tab1, tab2, tab3 = st.tabs(
        ["Interprétation clinique", "Facteurs de risque", "Recommandations"]
    )

    # -------- Tab 1: Interprétation
    with tab1:
        st.image("coeur.png", width=200, caption="prenez soin de votre coeur ")
        if proba < 0.30:
            st.success(
                "Le risque cardiovasculaire est faible. Les indicateurs cliniques sont rassurants."
            )
        elif proba < 0.60:
            st.warning(
                "Le risque cardiovasculaire est modéré. Une surveillance médicale régulière est recommandée."
            )
        else:
            st.error(
                "Le risque cardiovasculaire est élevé. Une consultation médicale est fortement conseillée."
            )

    # -------- Tab 2: Facteurs de risque
    with tab2:
        factors = {
            "Pression systolique": sbp / 200,
            "Cholestérol total": chol / 350,
            "LDL cholestérol": ldl / 250,
            "BMI": bmi / 50,
            "Adiposité": adiposity / 45,
            "Tabac": tobacco / 40,
            "Alcool": alcohol / 50,
            "Stress": typea / 100
        }

        fig_bar = px.bar(
            x=list(factors.keys()),
            y=list(factors.values()),
            text=[f"{v*100:.2f} %" for v in factors.values()],
            color=list(factors.values()),
            color_continuous_scale=["green", "orange", "red"]
        )

        fig_bar.update_layout(
            yaxis_title="Niveau relatif normalisé",
            xaxis_title="Facteurs contributifs",
            showlegend=False
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # -------- Tab 3: Recommandations
    with tab3:
        st.info(
            "- Surveillance régulière de la pression artérielle\n"
            "- Contrôle du cholestérol et du LDL\n"
            "- Activité physique adaptée\n"
            "- Réduction du tabac et de l’alcool\n"
            "- Suivi médical périodique"
        )

    st.divider()
    st.caption(
        "Cet outil fournit une estimation probabiliste basée sur un modèle "
        "de Machine Learning. Il ne remplace pas un avis médical professionnel."
    )
