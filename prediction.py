

import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# =========================
# Charger le modèle
# =========================
model = joblib.load("model.pkl")

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Santé Cardiaque ",
    page_icon="🫀",
    layout="wide"  # Utilise toute la largeur
)

# =========================
# HEADER LARGE
# =========================
st.markdown("""
<div style="
    width:100%;
    padding:30px;
    text-align:center;
    background: linear-gradient(to right, #e63946, #f1faee);
    color:white;
    border-radius:0px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.2);
    font-family: 'Arial';">
    <h1 style='margin:0; font-size:36px;'>🫀 Dashboard Santé Cardiaque</h1>
    <p style='margin:5px; font-size:16px; color:#f1faee;'>Analyse complète de votre risque cardiaque</p>
            
</div>
            
""", unsafe_allow_html=True)

st.write("---")

# =========================
# FORMULAIRE FIXE SUR TOUTE LA LARGEUR
# =========================
st.subheader("📝 Informations du patient")
col1, col2, col3 = st.columns([1,1,1])

with col1:
    age = st.number_input("Âge", 20, 90, 30)
    sbp = st.number_input("Pression systolique (SBP)", 90, 200, 120)
    chol = st.number_input("Cholestérol total", 120, 350, 200)

with col2:
    ldl = st.number_input("LDL cholestérol", 50, 250, 130)
    adiposity = st.number_input("Adiposity (%)", 10.0, 45.0, 20.0)
    obesity = st.number_input("BMI", 15.0, 50.0, 25.0)

with col3:
    tobacco = st.number_input("Tabac", 0.0, 40.0, 0.0)
    alcohol = st.number_input("Alcool", 0.0, 50.0, 0.0)
    typea = st.number_input("Type A (stress)", 0, 100, 50)
    famhist_text = st.selectbox("Antécédents familiaux", ["Absent", "Present"])

data = pd.DataFrame([{
    "sbp": sbp,
    "tobacco": tobacco,
    "ldl": ldl,
    "adiposity": adiposity,
    "typea": typea,
    "obesity": obesity,
    "alcohol": alcohol,
    "age": age,
    "chol": chol,
    "famhist": famhist_text
}])

st.write("---")

# =========================
# BOUTON PREDICTION FIXE
# =========================
if st.button(" Predire"):
    try:
        pred = model.predict(data)[0]
        proba = model.predict_proba(data)[0][1]

        # =========================
        # Section Résultat
        # =========================
        st.subheader("📊 Risque global")
        fig_risk = go.Figure(go.Pie(
            values=[proba, 1-proba],
            labels=["Risque", "Pas de risque"],
            hole=0.6,
            marker_colors=["#e63946", "#2a9d8f"],
            textinfo="percent"
        ))
        fig_risk.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_risk, use_container_width=True)

        # Carte résultat
        color_bg = "#ffe5e5" if pred==1 else "#e5ffe5"
        color_text = "#e63946" if pred==1 else "#2a9d8f"
        status_text = "⚠️ Risque détecté" if pred==1 else "✅ Aucun risque"

        st.markdown(f"""
        <div style="
            width:100%;
            padding:20px;
            border-radius:15px;
            background-color:{color_bg};
            color:{color_text};
            font-size:20px;
            text-align:center;
            box-shadow:0px 5px 20px rgba(0,0,0,0.15);">
            <b>{status_text}</b><br>
            Probabilité : <b>{proba:.2%}</b>
        </div>
        """, unsafe_allow_html=True)

        st.write("---")

        # =========================
        # Section Facteurs de risque en largeur
        # =========================
        st.subheader("🔎 Facteurs de risque")
        risk_factors = {
            "SBP": sbp / 200,
            "Chol": chol / 350,
            "LDL": ldl / 250,
            "BMI": obesity / 50,
            "Adiposity": adiposity / 45,
            "Tabac": tobacco / 40,
            "Alcool": alcohol / 50,
            "Type A": typea / 100
        }

        fig_bar = px.bar(
            x=list(risk_factors.keys()),
            y=list(risk_factors.values()),
            color=list(risk_factors.values()),
            color_continuous_scale=px.colors.sequential.Reds,
            text=[f"{v*100:.0f}%" for v in risk_factors.values()]
        )
        fig_bar.update_layout(
            showlegend=False,
            yaxis=dict(title="Niveau relatif"),
            xaxis=dict(title="Facteurs"),
            margin=dict(t=10,b=10,l=10,r=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.info("💡 Maintenez une alimentation saine et activité physique régulière.")

    except ValueError as e:
        st.error(f"Erreur : {e}")
        st.warning("⚠️ Vérifie toutes les valeurs.")

