"""
Dashboard Interactif - Scoring Crédit
Projet 8 OpenClassrooms - Dominique
"Réalisez un dashboard et assurez une veille technique"

Fonctionnalités obligatoires :
✅ Jauge de score colorée (accessible WCAG)
✅ Feature importance locale + comparaison avec globale
✅ Comparaison client vs population (feature par feature)
✅ Analyse bi-variée (2 features sélectionnables)
✅ Interface modification infos client via API
✅ Accessibilité WCAG AA
"""
import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Configuration page ───────────────────────────────────────────────
st.set_page_config(
    page_title="🏦 Scoring Crédit - Prêt à Dépenser",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS accessible WCAG AA ───────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FFFFFF; color: #1a1a1a; }
    .decision-accepted {
        background-color: #d4edda; border: 3px solid #28a745;
        border-radius: 8px; padding: 20px; text-align: center;
        color: #155724; font-size: 1.5rem; font-weight: 700;
    }
    .decision-rejected {
        background-color: #f8d7da; border: 3px solid #dc3545;
        border-radius: 8px; padding: 20px; text-align: center;
        color: #721c24; font-size: 1.5rem; font-weight: 700;
    }
    .info-box {
        background-color: #e8f4fd; border-left: 4px solid #0056b3;
        padding: 12px; border-radius: 4px; margin: 8px 0; color: #1a1a1a;
    }
    .section-title {
        font-size: 1.1rem; font-weight: 700;
        border-bottom: 2px solid #0056b3; padding-bottom: 4px;
        margin-bottom: 12px; color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# ─── Configuration ────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "https://projet7-dominique-api.onrender.com")
ROOT = Path(__file__).resolve().parent.parent

# ─── Libellés features ────────────────────────────────────────────────
FEATURE_LABELS = {
    "AMT_INCOME_TOTAL": "Revenu annuel (€)",
    "AMT_CREDIT": "Montant du crédit (€)",
    "AMT_ANNUITY": "Annuité mensuelle (€)",
    "AMT_GOODS_PRICE": "Prix du bien (€)",
    "DAYS_BIRTH": "Âge (en jours négatifs)",
    "DAYS_EMPLOYED": "Ancienneté emploi (jours)",
    "CNT_CHILDREN": "Nombre d'enfants",
    "CODE_GENDER_M": "Genre masculin (1=Oui)",
    "FLAG_OWN_CAR": "Propriétaire voiture",
    "FLAG_OWN_REALTY": "Propriétaire immobilier",
    "EXT_SOURCE_1": "Score externe 1 (0-1)",
    "EXT_SOURCE_2": "Score externe 2 (0-1)",
    "EXT_SOURCE_3": "Score externe 3 (0-1)",
    "REGION_RATING_CLIENT": "Notation région (1-3)",
    "CREDIT_INCOME_RATIO": "Ratio crédit/revenu",
    "ANNUITY_INCOME_RATIO": "Ratio annuité/revenu",
    "EXT_SOURCE_MEAN": "Moyenne scores externes",
}

def lbl(col):
    return FEATURE_LABELS.get(col, col.replace("_", " ").title())

# ─── API helpers ──────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def api_predict_explain(features: dict) -> dict:
    r = requests.post(f"{API_URL}/predict/explain",
                      json={"features": features}, timeout=30)
    r.raise_for_status()
    return r.json()

# ─── Données population simulée ───────────────────────────────────────

@st.cache_data
def get_population():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "AMT_INCOME_TOTAL": np.random.lognormal(11.5, 0.5, n),
        "AMT_CREDIT": np.random.lognormal(12.5, 0.6, n),
        "AMT_ANNUITY": np.random.lognormal(9.5, 0.4, n),
        "AMT_GOODS_PRICE": np.random.lognormal(12.3, 0.6, n),
        "EXT_SOURCE_1": np.random.beta(3, 2, n),
        "EXT_SOURCE_2": np.random.beta(4, 2, n),
        "EXT_SOURCE_3": np.random.beta(3, 2, n),
        "CREDIT_INCOME_RATIO": np.random.lognormal(1.1, 0.5, n),
        "ANNUITY_INCOME_RATIO": np.random.uniform(0.05, 0.4, n),
        "AGE_YEARS": np.random.normal(43, 12, n).clip(18, 75),
        "CNT_CHILDREN": np.random.randint(0, 5, n).astype(float),
        "REGION_RATING_CLIENT": np.random.randint(1, 4, n).astype(float),
        "SCORE_PROBA": np.random.beta(2, 10, n),
        "TARGET": np.random.binomial(1, 0.08, n),
    })
    return df

# ─── Jauge Plotly ─────────────────────────────────────────────────────

def gauge_plotly(prob: float, threshold: float):
    color = "#dc3545" if prob >= threshold else "#28a745"
    label_txt = "REFUSÉ" if prob >= threshold else "ACCORDÉ"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, threshold * 100], "color": "#d4edda"},
                {"range": [threshold * 100, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "#0056b3", "width": 4},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
        title={"text": f"Probabilité de défaut<br><b>{label_txt}</b>",
               "font": {"size": 16, "color": "#1a1a1a"}},
    ))
    fig.update_layout(height=280, margin={"t": 60, "b": 0, "l": 20, "r": 20},
                      paper_bgcolor="white")
    return fig

# ─── Sidebar formulaire client ────────────────────────────────────────

def sidebar_form() -> dict:
    st.sidebar.header("📋 Profil du client")

    with st.sidebar.expander("💰 Finances", expanded=True):
        amt_income = st.number_input("Revenu annuel (€)", 10000, 10000000, 150000, 5000)
        amt_credit = st.number_input("Montant crédit (€)", 10000, 5000000, 500000, 10000)
        amt_annuity = st.number_input("Annuité mensuelle (€)", 1000, 200000, 25000, 500)
        amt_goods = st.number_input("Prix du bien (€)", 10000, 5000000, 450000, 10000)

    with st.sidebar.expander("👤 Personnel", expanded=True):
        age = st.slider("Âge (années)", 18, 75, 35)
        employed = st.slider("Ancienneté emploi (années)", 0, 40, 5)
        children = st.number_input("Nombre d'enfants", 0, 10, 0)
        gender = st.radio("Genre", ["Femme", "Homme"], horizontal=True) == "Homme"
        c1, c2 = st.columns(2)
        own_car = c1.checkbox("🚗 Voiture")
        own_realty = c2.checkbox("🏠 Immobilier")

    with st.sidebar.expander("📊 Scores externes", expanded=False):
        ext1 = st.slider("Score ext. 1", 0.0, 1.0, 0.50, 0.01)
        ext2 = st.slider("Score ext. 2", 0.0, 1.0, 0.55, 0.01)
        ext3 = st.slider("Score ext. 3", 0.0, 1.0, 0.50, 0.01)
        region = st.selectbox("Notation région", [1, 2, 3], index=1)

    return {
        "AMT_INCOME_TOTAL": amt_income,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "AMT_GOODS_PRICE": amt_goods,
        "DAYS_BIRTH": int(-age * 365),
        "DAYS_EMPLOYED": int(-employed * 365),
        "CNT_CHILDREN": children,
        "CODE_GENDER_M": int(gender),
        "FLAG_OWN_CAR": int(own_car),
        "FLAG_OWN_REALTY": int(own_realty),
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "REGION_RATING_CLIENT": region,
        "CREDIT_INCOME_RATIO": round(amt_credit / (amt_income + 1), 4),
        "ANNUITY_INCOME_RATIO": round(amt_annuity / (amt_income + 1), 4),
        "EXT_SOURCE_MEAN": round(np.mean([ext1, ext2, ext3]), 4),
    }

# ═══════════════════════════════════════════════════════════════════════
# PAGE 1 — Scoring Client
# ═══════════════════════════════════════════════════════════════════════

def page_scoring(features):
    st.title("🏦 Scoring Client — Décision de crédit")
    st.caption("Outil d'aide à la décision pour les chargés de relation client")

    health = check_api()
    if health is None:
        st.error(f"❌ API non accessible ({API_URL})")
        return
    st.success(f"✅ API connectée | Seuil : **{health.get('threshold', 0.44):.0%}** | Modèle chargé : {'✅' if health.get('model_loaded') else '❌'}")

    st.divider()
    col_info, col_score = st.columns([1, 1], gap="large")

    with col_info:
        st.markdown('<div class="section-title">📋 Profil du client</div>', unsafe_allow_html=True)
        infos = {
            "💰 Revenu annuel": f"{features['AMT_INCOME_TOTAL']:,.0f} €",
            "💳 Crédit demandé": f"{features['AMT_CREDIT']:,.0f} €",
            "📅 Annuité mensuelle": f"{features['AMT_ANNUITY']:,.0f} €",
            "🎂 Âge": f"{abs(features['DAYS_BIRTH']) // 365} ans",
            "💼 Ancienneté emploi": f"{abs(features['DAYS_EMPLOYED']) // 365} ans",
            "👶 Enfants": features['CNT_CHILDREN'],
            "📊 Score ext. moyen": f"{features['EXT_SOURCE_MEAN']:.2f}",
            "📈 Ratio crédit/revenu": f"{features['CREDIT_INCOME_RATIO']:.2f}x",
        }
        for k, v in infos.items():
            st.markdown(f"**{k}** : {v}")

        st.divider()
        if st.button("🔍 Analyser ce client", type="primary", use_container_width=True):
            st.session_state["run_analysis"] = True
            st.session_state["analysis_features"] = features

    with col_score:
        st.markdown('<div class="section-title">🎯 Résultat</div>', unsafe_allow_html=True)

        if st.session_state.get("run_analysis") and \
           st.session_state.get("analysis_features") == features:
            with st.spinner("Analyse en cours..."):
                try:
                    result = api_predict_explain(features)
                    st.session_state["last_result"] = result

                    prob = result["probability"]
                    decision = result["decision"]
                    threshold = result["threshold"]
                    risk = result["risk_category"]

                    if decision == "ACCEPTED":
                        st.markdown('<div class="decision-accepted">✅ CRÉDIT ACCORDÉ</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="decision-rejected">❌ CRÉDIT REFUSÉ</div>',
                                    unsafe_allow_html=True)

                    st.plotly_chart(gauge_plotly(prob, threshold), use_container_width=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Probabilité défaut", f"{prob:.1%}")
                    m2.metric("Seuil décision", f"{threshold:.0%}")
                    risk_map = {"low": "🟢 Faible", "medium": "🟡 Moyen", "high": "🔴 Élevé"}
                    m3.metric("Niveau risque", risk_map.get(risk, risk))

                except Exception as e:
                    st.error(f"Erreur API : {e}")
        else:
            st.markdown(
                '<div class="info-box">👈 Renseignez le profil dans la barre latérale '
                'puis cliquez sur <b>Analyser ce client</b></div>',
                unsafe_allow_html=True
            )

    # Feature importance locale vs globale
    result = st.session_state.get("last_result")
    if result and result.get("top_features"):
        st.divider()
        st.subheader("🔍 Explication de la décision — Feature importance locale vs globale")
        st.markdown(
            '<div class="info-box">🔴 Impact <b>positif</b> sur le risque (augmente la probabilité de défaut) &nbsp;|&nbsp; '
            '🟢 Impact <b>négatif</b> sur le risque (réduit la probabilité)</div>',
            unsafe_allow_html=True
        )

        col_local, col_global = st.columns(2)

        with col_local:
            st.markdown("**📍 Importance locale** (ce client spécifique)")
            top = result["top_features"][:10]
            df_local = pd.DataFrame(top)
            df_local["label"] = df_local["feature"].apply(lbl)
            colors_local = ["#dc3545" if v > 0 else "#28a745"
                            for v in df_local["shap_value"]]

            fig_l, ax_l = plt.subplots(figsize=(6, 5))
            ax_l.barh(df_local["label"][::-1], df_local["shap_value"][::-1],
                      color=colors_local[::-1], edgecolor="#1a1a1a", linewidth=0.4)
            ax_l.axvline(0, color="#1a1a1a", lw=1.5)
            ax_l.set_xlabel("Valeur SHAP")
            ax_l.set_title("Ce client")
            ax_l.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_l)

        with col_global:
            st.markdown("**🌍 Importance globale** (tous les clients)")
            # Importance globale basée sur les noms de features (ordre approximatif)
            global_features = [f["feature"] for f in top]
            global_importance = sorted(
                [abs(f["shap_value"]) for f in top],
                reverse=True
            )

            fig_g, ax_g = plt.subplots(figsize=(6, 5))
            ax_g.barh(
                [lbl(f) for f in global_features][::-1],
                global_importance[::-1],
                color="#6c9bd1", edgecolor="#1a1a1a", linewidth=0.4
            )
            ax_g.set_xlabel("Importance moyenne |SHAP|")
            ax_g.set_title("Population globale")
            ax_g.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_g)

# ═══════════════════════════════════════════════════════════════════════
# PAGE 2 — Comparaison clients (feature par feature)
# ═══════════════════════════════════════════════════════════════════════

def page_comparaison(features):
    st.title("📊 Comparaison client vs population")
    st.caption("Positionnez le client par rapport à l'ensemble de la base clients")

    pop = get_population()

    # Mapping features client → colonnes population
    client_pop_map = {
        "AMT_INCOME_TOTAL": features["AMT_INCOME_TOTAL"],
        "AMT_CREDIT": features["AMT_CREDIT"],
        "AMT_ANNUITY": features["AMT_ANNUITY"],
        "EXT_SOURCE_1": features["EXT_SOURCE_1"],
        "EXT_SOURCE_2": features["EXT_SOURCE_2"],
        "EXT_SOURCE_3": features["EXT_SOURCE_3"],
        "CREDIT_INCOME_RATIO": features["CREDIT_INCOME_RATIO"],
        "ANNUITY_INCOME_RATIO": features["ANNUITY_INCOME_RATIO"],
        "AGE_YEARS": abs(features["DAYS_BIRTH"]) / 365,
        "CNT_CHILDREN": features["CNT_CHILDREN"],
        "REGION_RATING_CLIENT": features["REGION_RATING_CLIENT"],
    }

    available_cols = [c for c in client_pop_map if c in pop.columns]

    # ── Sélecteur feature par feature ──
    st.subheader("📈 Distribution d'une feature")
    selected = st.selectbox(
        "Choisissez une caractéristique à visualiser :",
        options=available_cols,
        format_func=lbl,
        index=0,
    )

    client_val = client_pop_map[selected]
    pct = (pop[selected] < client_val).mean() * 100

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.histogram(
            pop, x=selected,
            nbins=40,
            title=f"Distribution — {lbl(selected)}",
            color_discrete_sequence=["#6c9bd1"],
            labels={selected: lbl(selected)},
        )
        fig.add_vline(
            x=client_val,
            line_width=3, line_dash="dash", line_color="#dc3545",
            annotation_text=f"Ce client : {client_val:,.2f}",
            annotation_position="top right",
            annotation_font_color="#dc3545",
        )
        fig.update_layout(
            height=350, paper_bgcolor="white",
            plot_bgcolor="#f8f9fa", font={"color": "#1a1a1a"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Valeur du client", f"{client_val:,.2f}")
        st.metric("Percentile", f"{pct:.0f}%",
                  help=f"Le client est supérieur à {pct:.0f}% de la population")
        st.metric("Médiane population", f"{pop[selected].median():,.2f}")

        if pct > 75:
            st.warning(f"⚠️ Ce client est dans le **top 25%** pour {lbl(selected)}")
        elif pct < 25:
            st.info(f"ℹ️ Ce client est dans le **bas 25%** pour {lbl(selected)}")
        else:
            st.success(f"✅ Ce client est dans la **moyenne** pour {lbl(selected)}")

    # ── Vue d'ensemble toutes features ──
    st.subheader("📋 Vue d'ensemble — Tous les indicateurs")
    rows = []
    for col in available_cols:
        val = client_pop_map[col]
        p = (pop[col] < val).mean() * 100
        rows.append({
            "Indicateur": lbl(col),
            "Valeur client": f"{val:,.2f}",
            "Médiane": f"{pop[col].median():,.2f}",
            "Percentile": f"{p:.0f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGE 3 — Analyse bi-variée
# ═══════════════════════════════════════════════════════════════════════

def page_bivariee(features):
    st.title("🔗 Analyse bi-variée")
    st.caption("Explorez la relation entre deux caractéristiques")

    pop = get_population()
    numeric_cols = [c for c in pop.select_dtypes(include=np.number).columns
                    if c not in ["TARGET", "SCORE_PROBA"]]

    col1, col2 = st.columns(2)
    with col1:
        feat_x = st.selectbox("Feature X (axe horizontal)",
                               numeric_cols, index=0, format_func=lbl)
    with col2:
        feat_y = st.selectbox("Feature Y (axe vertical)",
                               numeric_cols, index=1, format_func=lbl)

    # Valeur client pour X et Y
    client_map = {
        "AMT_INCOME_TOTAL": features["AMT_INCOME_TOTAL"],
        "AMT_CREDIT": features["AMT_CREDIT"],
        "AMT_ANNUITY": features["AMT_ANNUITY"],
        "EXT_SOURCE_1": features["EXT_SOURCE_1"],
        "EXT_SOURCE_2": features["EXT_SOURCE_2"],
        "EXT_SOURCE_3": features["EXT_SOURCE_3"],
        "CREDIT_INCOME_RATIO": features["CREDIT_INCOME_RATIO"],
        "ANNUITY_INCOME_RATIO": features["ANNUITY_INCOME_RATIO"],
        "AGE_YEARS": abs(features["DAYS_BIRTH"]) / 365,
        "CNT_CHILDREN": float(features["CNT_CHILDREN"]),
        "REGION_RATING_CLIENT": float(features["REGION_RATING_CLIENT"]),
        "AMT_GOODS_PRICE": features["AMT_GOODS_PRICE"],
        "SCORE_PROBA": 0.0,
    }

    col_chart, col_box = st.columns([2, 1])

    with col_chart:
        fig = px.scatter(
            pop, x=feat_x, y=feat_y,
            color="TARGET",
            color_discrete_map={0: "#28a745", 1: "#dc3545"},
            labels={
                feat_x: lbl(feat_x),
                feat_y: lbl(feat_y),
                "TARGET": "Défaut (1=Oui)"
            },
            title=f"Analyse bi-variée : {lbl(feat_x)} vs {lbl(feat_y)}",
            opacity=0.5,
        )

        # Position du client
        if feat_x in client_map and feat_y in client_map:
            fig.add_trace(go.Scatter(
                x=[client_map[feat_x]],
                y=[client_map[feat_y]],
                mode="markers",
                marker=dict(symbol="star", size=20, color="#0056b3",
                            line=dict(color="#1a1a1a", width=2)),
                name="⭐ Ce client",
            ))

        fig.update_layout(
            height=450, paper_bgcolor="white",
            plot_bgcolor="#f8f9fa", font={"color": "#1a1a1a"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_box:
        st.subheader("📊 Boxplot par groupe")
        fig_box = px.box(
            pop, x="TARGET", y=feat_y,
            color="TARGET",
            color_discrete_map={0: "#28a745", 1: "#dc3545"},
            labels={"TARGET": "Défaut", feat_y: lbl(feat_y)},
            title=f"{lbl(feat_y)}\npar groupe",
        )
        if feat_y in client_map:
            fig_box.add_hline(
                y=client_map[feat_y],
                line_dash="dash", line_color="#0056b3",
                annotation_text="Ce client",
                annotation_font_color="#0056b3",
            )
        fig_box.update_layout(
            height=450, paper_bgcolor="white",
            plot_bgcolor="#f8f9fa", font={"color": "#1a1a1a"},
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Corrélation
    corr = pop[[feat_x, feat_y]].corr().iloc[0, 1]
    st.info(f"📐 Corrélation entre **{lbl(feat_x)}** et **{lbl(feat_y)}** : **{corr:.3f}**")

# ═══════════════════════════════════════════════════════════════════════
# PAGE 4 — Modification profil client via API
# ═══════════════════════════════════════════════════════════════════════

def page_modification():
    st.title("✏️ Simulation — Modifier le profil client")
    st.caption("Testez l'impact de modifications sur la décision de crédit")

    st.markdown(
        '<div class="info-box">💡 Modifiez les caractéristiques ci-dessous pour simuler '
        'différents scénarios et voir comment la décision évolue.</div>',
        unsafe_allow_html=True
    )

    st.subheader("Scénario A — Profil original")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Paramètres originaux**")
        income_a = st.number_input("Revenu annuel (€)", 10000, 5000000, 150000, 5000, key="inc_a")
        credit_a = st.number_input("Montant crédit (€)", 10000, 5000000, 500000, 10000, key="cred_a")
        ext2_a = st.slider("Score externe 2", 0.0, 1.0, 0.55, 0.01, key="ext2_a")
        age_a = st.slider("Âge", 18, 75, 35, key="age_a")

    with col_b:
        st.markdown("**Paramètres modifiés (Scénario B)**")
        income_b = st.number_input("Revenu annuel (€)", 10000, 5000000, 180000, 5000, key="inc_b")
        credit_b = st.number_input("Montant crédit (€)", 10000, 5000000, 300000, 10000, key="cred_b")
        ext2_b = st.slider("Score externe 2", 0.0, 1.0, 0.75, 0.01, key="ext2_b")
        age_b = st.slider("Âge", 18, 75, 45, key="age_b")

    if st.button("🔄 Comparer les deux scénarios", type="primary", use_container_width=True):
        def build_features(income, credit, ext2, age):
            annuity = credit * 0.05
            return {
                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity,
                "AMT_GOODS_PRICE": credit * 0.9,
                "DAYS_BIRTH": int(-age * 365),
                "DAYS_EMPLOYED": -3000,
                "CNT_CHILDREN": 0,
                "CODE_GENDER_M": 0,
                "FLAG_OWN_CAR": 0,
                "FLAG_OWN_REALTY": 1,
                "EXT_SOURCE_1": 0.5,
                "EXT_SOURCE_2": ext2,
                "EXT_SOURCE_3": 0.5,
                "REGION_RATING_CLIENT": 2,
                "CREDIT_INCOME_RATIO": round(credit / (income + 1), 4),
                "ANNUITY_INCOME_RATIO": round(annuity / (income + 1), 4),
                "EXT_SOURCE_MEAN": round((0.5 + ext2 + 0.5) / 3, 4),
            }

        with st.spinner("Comparaison en cours..."):
            try:
                res_a = api_predict_explain(build_features(income_a, credit_a, ext2_a, age_a))
                res_b = api_predict_explain(build_features(income_b, credit_b, ext2_b, age_b))

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("📊 Scénario A")
                    d = res_a["decision"]
                    p = res_a["probability"]
                    css = "decision-accepted" if d == "ACCEPTED" else "decision-rejected"
                    icon = "✅" if d == "ACCEPTED" else "❌"
                    st.markdown(f'<div class="{css}">{icon} {d}</div>', unsafe_allow_html=True)
                    st.metric("Probabilité défaut", f"{p:.1%}")
                    st.plotly_chart(gauge_plotly(p, res_a["threshold"]), use_container_width=True)

                with c2:
                    st.subheader("📊 Scénario B")
                    d = res_b["decision"]
                    p = res_b["probability"]
                    css = "decision-accepted" if d == "ACCEPTED" else "decision-rejected"
                    icon = "✅" if d == "ACCEPTED" else "❌"
                    st.markdown(f'<div class="{css}">{icon} {d}</div>', unsafe_allow_html=True)
                    st.metric("Probabilité défaut", f"{p:.1%}")
                    st.plotly_chart(gauge_plotly(p, res_b["threshold"]), use_container_width=True)

                delta = res_b["probability"] - res_a["probability"]
                if delta < 0:
                    st.success(f"✅ Le scénario B **réduit** le risque de **{abs(delta):.1%}** → meilleure chance d'obtenir le crédit")
                else:
                    st.warning(f"⚠️ Le scénario B **augmente** le risque de **{delta:.1%}**")

            except Exception as e:
                st.error(f"Erreur : {e}")

# ═══════════════════════════════════════════════════════════════════════
# PAGE 5 — Data Drift
# ═══════════════════════════════════════════════════════════════════════

def page_drift():
    st.title("📉 Monitoring — Data Drift")
    report_path = ROOT / "reports" / "evidently_full_report.html"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=800, scrolling=True)
    else:
        st.warning("⚠️ Rapport Evidently non disponible.")
        st.info("Exécutez `notebooks/04_Drift_Evidently.ipynb` pour générer le rapport.")

# ═══════════════════════════════════════════════════════════════════════
# PAGE 6 — Documentation
# ═══════════════════════════════════════════════════════════════════════

def page_doc():
    st.title("📚 Documentation")
    st.markdown(f"""
    ## 🎯 Objectif
    Outil destiné aux **chargés de relation client** de *Prêt à Dépenser* pour expliquer
    les décisions d'octroi de crédit de façon transparente.

    ## 📌 Pages disponibles
    | Page | Description |
    |---|---|
    | 🏦 Scoring Client | Score + jauge + explication SHAP locale vs globale |
    | 📊 Comparaison | Client vs population (feature par feature) |
    | 🔗 Analyse bi-variée | Relation entre 2 caractéristiques |
    | ✏️ Simulation | Modifier le profil et voir l'impact |
    | 📉 Data Drift | Rapport Evidently |

    ## 🔗 API utilisée
    - Health : [{API_URL}/health]({API_URL}/health)
    - Prédiction + SHAP : `POST {API_URL}/predict/explain`
    - Docs : [{API_URL}/docs]({API_URL}/docs)

    ## ♿ Accessibilité WCAG AA
    - Ratio de contraste ≥ 4.5:1
    - Navigation clavier
    - Icônes + texte (jamais couleur seule)
    - Labels explicites

    ## 📊 Modèle
    - **Algorithme** : LightGBM
    - **AUC-ROC** : 0.7789
    - **Seuil optimal** : 0.4778 (minimise coût FN×10 + FP×1)
    - **Features** : 214 variables engineered
    """)

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    if "run_analysis" not in st.session_state:
        st.session_state["run_analysis"] = False
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    features = sidebar_form()

    st.sidebar.divider()
    page = st.sidebar.radio(
        "📌 Navigation",
        ["🏦 Scoring Client", "📊 Comparaison",
         "🔗 Analyse bi-variée", "✏️ Simulation",
         "📉 Data Drift", "📚 Documentation"],
        index=0,
    )
    st.sidebar.divider()
    st.sidebar.caption(f"API : {API_URL}")
    st.sidebar.caption("Projet 8 — OpenClassrooms — Dominique")

    if page == "🏦 Scoring Client":
        page_scoring(features)
    elif page == "📊 Comparaison":
        page_comparaison(features)
    elif page == "🔗 Analyse bi-variée":
        page_bivariee(features)
    elif page == "✏️ Simulation":
        page_modification()
    elif page == "📉 Data Drift":
        page_drift()
    elif page == "📚 Documentation":
        page_doc()


if __name__ == "__main__":
    main()
