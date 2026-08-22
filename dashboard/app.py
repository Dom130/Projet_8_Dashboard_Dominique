"""
Dashboard Interactif - Scoring Crédit
Projet 8 OpenClassrooms - Dominique
"Réalisez un dashboard et assurez une veille technique"

Fonctionnalités :
- Jauge de score colorée (accessible WCAG)
- Explication locale SHAP
- Comparaison client vs population
- Saisie dynamique de nouveaux profils
"""
import os
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ─── Configuration page ───────────────────────────────────────────────
st.set_page_config(
    page_title="🏦 Scoring Crédit - Prêt à Dépenser",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS accessible WCAG ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Contraste élevé WCAG AA */
    .main { background-color: #FFFFFF; color: #1a1a1a; }
    .stMetric label { font-size: 1rem; font-weight: 600; color: #1a1a1a; }
    .stMetric value { font-size: 1.5rem; font-weight: 700; }
    .decision-accepted {
        background-color: #d4edda; border: 2px solid #28a745;
        border-radius: 8px; padding: 16px; text-align: center;
        color: #155724; font-size: 1.4rem; font-weight: 700;
    }
    .decision-rejected {
        background-color: #f8d7da; border: 2px solid #dc3545;
        border-radius: 8px; padding: 16px; text-align: center;
        color: #721c24; font-size: 1.4rem; font-weight: 700;
    }
    .info-box {
        background-color: #e8f4fd; border-left: 4px solid #0056b3;
        padding: 12px; border-radius: 4px; margin: 8px 0;
        color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# ─── Configuration API ────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "https://projet7-dominique-api.onrender.com")
ROOT = Path(__file__).resolve().parent.parent

# ─── Libellés des features ────────────────────────────────────────────
FEATURE_LABELS = {
    "AMT_INCOME_TOTAL": "Revenu annuel (€)",
    "AMT_CREDIT": "Montant du crédit (€)",
    "AMT_ANNUITY": "Annuité mensuelle (€)",
    "AMT_GOODS_PRICE": "Prix du bien (€)",
    "DAYS_BIRTH": "Âge (jours négatifs)",
    "DAYS_EMPLOYED": "Ancienneté emploi (jours négatifs)",
    "CNT_CHILDREN": "Nombre d'enfants",
    "CODE_GENDER_M": "Genre masculin (1=Oui)",
    "FLAG_OWN_CAR": "Propriétaire voiture",
    "FLAG_OWN_REALTY": "Propriétaire immobilier",
    "EXT_SOURCE_1": "Score externe 1",
    "EXT_SOURCE_2": "Score externe 2",
    "EXT_SOURCE_3": "Score externe 3",
    "REGION_RATING_CLIENT": "Notation région",
    "CREDIT_INCOME_RATIO": "Ratio crédit/revenu",
    "ANNUITY_INCOME_RATIO": "Ratio annuité/revenu",
    "EXT_SOURCE_MEAN": "Moyenne scores externes",
}


def label(col):
    return FEATURE_LABELS.get(col, col.replace("_", " ").title())


# ─── Helpers API ──────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def call_predict_explain(features: dict) -> dict:
    r = requests.post(
        f"{API_URL}/predict/explain",
        json={"features": features},
        timeout=30
    )
    r.raise_for_status()
    return r.json()


# ─── Jauge Plotly accessible ──────────────────────────────────────────

def draw_gauge_plotly(probability: float, threshold: float):
    """Jauge interactive accessible WCAG avec Plotly."""
    color = "#dc3545" if probability >= threshold else "#28a745"
    label_text = "REFUSÉ" if probability >= threshold else "ACCORDÉ"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        delta={
            "reference": threshold * 100,
            "decreasing": {"color": "#28a745"},
            "increasing": {"color": "#dc3545"},
        },
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#1a1a1a"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#1a1a1a",
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
        title={"text": f"Probabilité de défaut<br><b>{label_text}</b>",
               "font": {"size": 18, "color": "#1a1a1a"}},
    ))

    fig.update_layout(
        height=300,
        margin={"t": 60, "b": 20, "l": 30, "r": 30},
        paper_bgcolor="white",
        font={"color": "#1a1a1a"},
    )
    return fig


# ─── Données de population simulée ────────────────────────────────────

@st.cache_data
def get_population_data():
    """Simule une population de clients pour la comparaison."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "AMT_INCOME_TOTAL": np.random.lognormal(11.5, 0.5, n),
        "AMT_CREDIT": np.random.lognormal(12.5, 0.6, n),
        "AMT_ANNUITY": np.random.lognormal(9.5, 0.4, n),
        "EXT_SOURCE_1": np.random.beta(3, 2, n),
        "EXT_SOURCE_2": np.random.beta(4, 2, n),
        "EXT_SOURCE_3": np.random.beta(3, 2, n),
        "CREDIT_INCOME_RATIO": np.random.lognormal(1.1, 0.5, n),
        "AGE_YEARS": np.random.normal(43, 12, n).clip(18, 75),
        "TARGET": np.random.binomial(1, 0.08, n),
    })


# ─── Formulaire sidebar ───────────────────────────────────────────────

def sidebar_form() -> dict:
    st.sidebar.header("📋 Profil du client")
    st.sidebar.caption("Renseignez les informations du client")

    with st.sidebar.expander("💰 Informations financières", expanded=True):
        amt_income = st.number_input(
            "Revenu annuel (€)", 10000, 10000000, 150000, 5000,
            help="Revenu total annuel du client"
        )
        amt_credit = st.number_input(
            "Montant crédit demandé (€)", 10000, 5000000, 500000, 10000
        )
        amt_annuity = st.number_input(
            "Annuité mensuelle (€)", 1000, 200000, 25000, 500
        )
        amt_goods = st.number_input(
            "Prix du bien financé (€)", 10000, 5000000, 450000, 10000
        )

    with st.sidebar.expander("👤 Informations personnelles", expanded=True):
        age = st.slider("Âge (années)", 18, 75, 35,
                        help="Âge du client en années")
        employed = st.slider("Ancienneté emploi (années)", 0, 40, 5)
        children = st.number_input("Nombre d'enfants", 0, 10, 0)
        gender = st.radio("Genre", ["Femme", "Homme"],
                          horizontal=True) == "Homme"
        col1, col2 = st.columns(2)
        own_car = col1.checkbox("🚗 Voiture")
        own_realty = col2.checkbox("🏠 Immobilier")

    with st.sidebar.expander("📊 Scores externes", expanded=False):
        st.caption("Scores fournis par des bureaux de crédit externes (0 = risqué, 1 = sûr)")
        ext1 = st.slider("Score externe 1", 0.0, 1.0, 0.50, 0.01)
        ext2 = st.slider("Score externe 2", 0.0, 1.0, 0.55, 0.01)
        ext3 = st.slider("Score externe 3", 0.0, 1.0, 0.50, 0.01)
        region = st.selectbox("Notation de la région", [1, 2, 3],
                               index=1, help="1=bonne, 3=risquée")

    credit_income = amt_credit / (amt_income + 1)
    annuity_income = amt_annuity / (amt_income + 1)
    ext_mean = np.mean([ext1, ext2, ext3])

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
        "CREDIT_INCOME_RATIO": round(credit_income, 4),
        "ANNUITY_INCOME_RATIO": round(annuity_income, 4),
        "EXT_SOURCE_MEAN": round(ext_mean, 4),
    }


# ─── Page Scoring ─────────────────────────────────────────────────────

def page_scoring(features):
    st.title("🏦 Tableau de bord — Scoring Crédit")
    st.caption("Prêt à Dépenser · Outil d'aide à la décision pour les chargés de clientèle")

    health = check_api()
    if health is None:
        st.error(f"❌ API non accessible ({API_URL})")
        return

    status_color = "🟢" if health.get("model_loaded") else "🔴"
    st.info(f"{status_color} API connectée | Seuil de décision : **{health.get('threshold', 0.44):.0%}** | v{health.get('version', '1.0.0')}")

    st.divider()

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("📋 Profil client")
        data_display = {
            "💰 Revenu annuel": f"{features['AMT_INCOME_TOTAL']:,.0f} €",
            "💳 Crédit demandé": f"{features['AMT_CREDIT']:,.0f} €",
            "📅 Annuité": f"{features['AMT_ANNUITY']:,.0f} €",
            "🎂 Âge": f"{abs(features['DAYS_BIRTH']) // 365} ans",
            "💼 Ancienneté": f"{abs(features['DAYS_EMPLOYED']) // 365} ans",
            "👶 Enfants": features['CNT_CHILDREN'],
            "📊 Score ext. moyen": f"{features['EXT_SOURCE_MEAN']:.2f}",
            "📈 Ratio crédit/revenu": f"{features['CREDIT_INCOME_RATIO']:.2f}",
        }
        for k, v in data_display.items():
            st.markdown(f"**{k}** : {v}")

        st.divider()
        analyser = st.button(
            "🔍 Analyser ce client",
            type="primary",
            use_container_width=True,
            help="Cliquez pour obtenir la décision et l'explication"
        )

    with col_result:
        st.subheader("🎯 Décision de crédit")

        if analyser:
            with st.spinner("Analyse en cours..."):
                try:
                    result = call_predict_explain(features)
                    prob = result["probability"]
                    decision = result["decision"]
                    threshold = result["threshold"]
                    risk = result["risk_category"]

                    # Décision principale
                    if decision == "ACCEPTED":
                        st.markdown(
                            '<div class="decision-accepted">✅ CRÉDIT ACCORDÉ</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="decision-rejected">❌ CRÉDIT REFUSÉ</div>',
                            unsafe_allow_html=True
                        )

                    st.plotly_chart(
                        draw_gauge_plotly(prob, threshold),
                        use_container_width=True
                    )

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Probabilité défaut", f"{prob:.1%}")
                    m2.metric("Seuil", f"{threshold:.0%}")
                    risk_map = {"low": "🟢 Faible", "medium": "🟡 Moyen", "high": "🔴 Élevé"}
                    m3.metric("Risque", risk_map.get(risk, risk))

                    # Explication SHAP
                    if result.get("top_features"):
                        st.subheader("🔍 Facteurs influençant la décision")
                        st.markdown(
                            '<div class="info-box">🔴 Facteurs qui <b>augmentent</b> le risque &nbsp;|&nbsp; '
                            '🟢 Facteurs qui <b>diminuent</b> le risque</div>',
                            unsafe_allow_html=True
                        )

                        top = result["top_features"][:10]
                        df_shap = pd.DataFrame(top)
                        df_shap["label"] = df_shap["feature"].apply(label)
                        df_shap["color"] = df_shap["shap_value"].apply(
                            lambda x: "#dc3545" if x > 0 else "#28a745"
                        )

                        fig, ax = plt.subplots(figsize=(7, 5))
                        bars = ax.barh(
                            df_shap["label"][::-1],
                            df_shap["shap_value"][::-1],
                            color=df_shap["color"].tolist()[::-1],
                            edgecolor="#1a1a1a",
                            linewidth=0.5,
                        )
                        ax.axvline(0, color="#1a1a1a", linewidth=1.5)
                        ax.set_xlabel("Impact sur la décision (valeur SHAP)",
                                      fontsize=10, color="#1a1a1a")
                        ax.set_title("Explication locale de la décision",
                                     fontsize=12, fontweight="bold", color="#1a1a1a")
                        ax.tick_params(colors="#1a1a1a")
                        ax.grid(axis="x", alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)

                    # Stocker le résultat pour la comparaison
                    st.session_state["last_result"] = result
                    st.session_state["last_features"] = features

                except requests.exceptions.ConnectionError:
                    st.error("❌ Impossible de joindre l'API.")
                except Exception as e:
                    st.error(f"Erreur : {e}")
        else:
            st.markdown(
                '<div class="info-box">👈 Renseignez le profil client dans la barre latérale, '
                'puis cliquez sur <b>Analyser ce client</b></div>',
                unsafe_allow_html=True
            )


# ─── Page Comparaison ─────────────────────────────────────────────────

def page_comparaison(features):
    st.title("📊 Comparaison client vs population")
    st.caption("Positionnez le client par rapport à l'ensemble de la base clients")

    pop = get_population_data()
    age_client = abs(features["DAYS_BIRTH"]) / 365

    features_to_compare = {
        "Revenu annuel (€)": ("AMT_INCOME_TOTAL", features["AMT_INCOME_TOTAL"]),
        "Montant crédit (€)": ("AMT_CREDIT", features["AMT_CREDIT"]),
        "Score externe moyen": ("EXT_SOURCE_MEAN",
                                 np.mean([features["EXT_SOURCE_1"],
                                          features["EXT_SOURCE_2"],
                                          features["EXT_SOURCE_3"]])),
        "Ratio crédit/revenu": ("CREDIT_INCOME_RATIO", features["CREDIT_INCOME_RATIO"]),
        "Âge (années)": ("AGE_YEARS", age_client),
    }

    cols = st.columns(2)
    for i, (feat_label, (col_name, client_val)) in enumerate(features_to_compare.items()):
        if col_name not in pop.columns:
            continue
        with cols[i % 2]:
            fig = px.histogram(
                pop, x=col_name,
                nbins=30,
                title=f"Distribution — {feat_label}",
                color_discrete_sequence=["#6c9bd1"],
                labels={col_name: feat_label},
            )
            fig.add_vline(
                x=client_val,
                line_width=3,
                line_dash="dash",
                line_color="#dc3545",
                annotation_text=f"Ce client : {client_val:,.1f}",
                annotation_position="top right",
                annotation_font_color="#dc3545",
            )
            fig.update_layout(
                height=300,
                margin={"t": 50, "b": 30, "l": 30, "r": 30},
                paper_bgcolor="white",
                plot_bgcolor="#f8f9fa",
                font={"color": "#1a1a1a"},
                title_font={"size": 13},
            )
            st.plotly_chart(fig, use_container_width=True)

    # Percentiles
    st.subheader("📈 Position du client (percentiles)")
    rows = []
    for feat_label, (col_name, client_val) in features_to_compare.items():
        if col_name in pop.columns:
            pct = (pop[col_name] < client_val).mean() * 100
            rows.append({
                "Indicateur": feat_label,
                "Valeur client": f"{client_val:,.2f}",
                "Percentile": f"{pct:.0f}%",
                "Interprétation": f"Supérieur à {pct:.0f}% des clients"
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─── Page Drift ───────────────────────────────────────────────────────

def page_drift():
    st.title("📉 Monitoring — Data Drift")
    st.caption("Surveillance de la dérive des données en production")

    report_path = ROOT / "reports" / "evidently_full_report.html"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=800, scrolling=True)
    else:
        st.warning("⚠️ Rapport Evidently non disponible.")
        st.info(
            "Pour générer le rapport :\n"
            "1. Ouvrir `notebooks/04_Drift_Evidently.ipynb`\n"
            "2. Exécuter toutes les cellules\n"
            "3. Le rapport sera créé dans `reports/evidently_full_report.html`"
        )


# ─── Page Veille ──────────────────────────────────────────────────────

def page_veille():
    st.title("🔬 Veille Technique — XAI")
    st.caption("Explicabilité des modèles de Machine Learning : SHAP vs LIME")

    st.markdown("""
    ## Pourquoi l'explicabilité est-elle essentielle ?

    Dans le secteur bancaire, la réglementation (RGPD, directive crédit) impose de pouvoir
    **expliquer toute décision automatisée** à un client. L'explicabilité n'est pas un choix,
    c'est une obligation légale.

    ---

    ## SHAP (SHapley Additive exPlanations)

    **Principe :** Basé sur la théorie des jeux (valeurs de Shapley), SHAP mesure la
    contribution marginale de chaque feature à la prédiction.

    **Avantages :**
    - Consistant et localement précis
    - Supporte les modèles tree-based (LightGBM, XGBoost) nativement
    - Visualisations riches (summary plot, force plot, dependence plot)

    **Inconvénients :**
    - Peut être lent sur de grands datasets
    - Interprétation parfois complexe pour des non-experts

    ---

    ## LIME (Local Interpretable Model-agnostic Explanations)

    **Principe :** Approxime localement le modèle complexe par un modèle linéaire simple
    autour d'une instance à expliquer.

    **Avantages :**
    - Model-agnostic (fonctionne avec n'importe quel modèle)
    - Facile à comprendre pour des non-experts

    **Inconvénients :**
    - Instable (résultats peuvent varier entre deux runs)
    - Moins fidèle que SHAP pour les modèles tree-based

    ---

    ## Comparaison SHAP vs LIME

    | Critère | SHAP | LIME |
    |---|---|---|
    | Cohérence | ✅ Garantie mathématiquement | ⚠️ Non garantie |
    | Vitesse | ⚠️ Moyen | ✅ Rapide |
    | Model-agnostic | ✅ Oui | ✅ Oui |
    | Stabilité | ✅ Stable | ⚠️ Variable |
    | Adapté LightGBM | ✅ Natif | ✅ Compatible |

    ---

    ## Choix pour ce projet

    **SHAP** a été retenu pour ce projet car :
    1. LightGBM est un modèle tree-based → TreeExplainer très rapide
    2. Garanties mathématiques (valeurs de Shapley)
    3. Visualisations intégrées dans le dashboard
    4. Cohérence globale (feature importance) + locale (explication par client)

    Pour approfondir, consultez le notebook de veille :
    `notebooks/02_Veille_XAI_SHAP_LIME.ipynb`
    """)


# ─── Page Documentation ───────────────────────────────────────────────

def page_documentation():
    st.title("📚 Documentation")

    st.header("🎯 Contexte métier")
    st.markdown("""
    **Prêt à Dépenser** souhaite offrir plus de **transparence** dans ses décisions d'octroi de crédit.
    Ce dashboard permet aux chargés de relation client d'**expliquer simplement** pourquoi un crédit
    est accordé ou refusé, lors de rendez-vous clients.

    - **Coût d'un Faux Négatif** (accepter un mauvais payeur) : **10**
    - **Coût d'un Faux Positif** (refuser un bon client) : **1**
    - **Seuil optimal** : calibré pour minimiser ce coût asymétrique
    """)

    st.header("🔧 Architecture")
    st.code("""
    Projet_8/
    ├── dashboard/        # Ce dashboard Streamlit
    ├── notebooks/        # Veille XAI (SHAP vs LIME)
    ├── reports/          # Rapport Evidently (drift)
    ├── note_methodologique.md
    └── README.md
    
    → Connecté à l'API Projet 7 :
       https://projet7-dominique-api.onrender.com
    """, language="")

    st.header("📡 API utilisée")
    st.markdown(f"""
    - **Health** : [{API_URL}/health]({API_URL}/health)
    - **Predict + SHAP** : `POST {API_URL}/predict/explain`
    - **Documentation** : [{API_URL}/docs]({API_URL}/docs)
    """)

    st.header("♿ Accessibilité WCAG")
    st.markdown("""
    Ce dashboard respecte les critères WCAG AA :
    - Ratio de contraste texte/fond ≥ 4.5:1
    - Navigation au clavier possible
    - Labels explicites sur tous les champs
    - Couleurs non seules pour transmettre l'information (icônes + texte)
    """)


# ─── Navigation principale ────────────────────────────────────────────

def main():
    features = sidebar_form()

    st.sidebar.divider()
    page = st.sidebar.radio(
        "📌 Navigation",
        [
            "🎯 Scoring Client",
            "📊 Comparaison",
            "📉 Data Drift",
            "🔬 Veille XAI",
            "📚 Documentation",
        ],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.caption(f"🔗 API : {API_URL}")
    st.sidebar.caption("Projet 8 — OpenClassrooms — Dominique")

    if page == "🎯 Scoring Client":
        page_scoring(features)
    elif page == "📊 Comparaison":
        page_comparaison(features)
    elif page == "📉 Data Drift":
        page_drift()
    elif page == "🔬 Veille XAI":
        page_veille()
    elif page == "📚 Documentation":
        page_documentation()


if __name__ == "__main__":
    main()
