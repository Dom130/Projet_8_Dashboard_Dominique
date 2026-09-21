# 🏦 Projet 8 — Réalisez un dashboard et assurez une veille technique
**OpenClassrooms — Dominique (Dom130)**

## 📋 Description

Suite du Projet 7 (scoring crédit), ce projet développe :
1. Un **dashboard interactif** pour les chargés de relation client
2. Une **veille technique** sur l'explicabilité des modèles 

## 🎯 Contexte métier

Chez **Prêt à Dépenser**, les clients demandent plus de transparence sur les décisions de crédit.
Ce dashboard permet aux chargés de clientèle d'**expliquer simplement** chaque décision lors de rendez-vous.

## 🏗️ Structure

```
Projet_8_Dashboard/
├── dashboard/
│   └── app.py              # Dashboard Streamlit (5 pages)
├── notebooks/
│   └── notebook 1  # Veille technique
├── reports/
│   └── evidently_full_report.html      # Rapport drift
├── note_methodologique.md  
└── README.md
```

## 🚀 Fonctionnalités du dashboard

| Page | Description |
|---|---|
| 🎯 Scoring Client | Jauge de score + décision + explication SHAP |
| 📊 Comparaison | Client vs population (histogrammes, percentiles) |
| 📉 Data Drift | Rapport Evidently interactif |
| 🔬 Veille  
| 📚 Documentation | Guide d'utilisation |

## ♿ Accessibilité WCAG

- Ratio de contraste ≥ 4.5:1 (WCAG AA)
- Navigation clavier
- Labels explicites sur tous les éléments
- Couleurs + icônes + texte (jamais la couleur seule)

## 🔗 Liens

- **Projet 7 (API)** : https://projet7-dominique-api.onrender.com
- **Dashboard Projet 7** : https://projet7-dominique-dashboard.onrender.com
- **GitHub Projet 7** : https://github.com/Dom130/projet7_Open_Dominique

## 📦 Installation locale

```bash
pip install streamlit requests pandas plotly matplotlib
streamlit run dashboard/app.py
```

## 📚 Veille technique

Voir `note_methodologique.md` et `notebooks/02_Veille_XAI_SHAP_LIME.ipynb`

**Méthode retenue : SHAP** — fondement mathématique solide (valeurs de Shapley),
TreeExplainer natif pour LightGBM, déterministe et cohérent.

---
*Projet 8 OpenClassrooms — "Réalisez un dashboard et assurez une veille technique"*
