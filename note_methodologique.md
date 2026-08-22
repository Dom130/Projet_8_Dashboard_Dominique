# Note Méthodologique — Veille Technique XAI
## Projet 8 OpenClassrooms — Dominique
### "Réalisez un dashboard et assurez une veille technique"

---

## 1. Contexte et objectif

Dans le cadre du projet de scoring crédit pour **Prêt à Dépenser**, nous avons développé
un modèle LightGBM capable de prédire la probabilité de défaut de paiement d'un client.

Cependant, la réglementation bancaire (RGPD, Art. 22) et les attentes des clients exigent
que toute décision automatisée soit **explicable**. Cette note présente la veille technique
réalisée sur les méthodes d'explicabilité des modèles de Machine Learning (XAI).

---

## 2. Méthodes étudiées

### 2.1 SHAP (SHapley Additive exPlanations)

**Référence :** Lundberg & Lee, 2017 — "A Unified Approach to Interpreting Model Predictions"
*Source : https://arxiv.org/abs/1705.07874*

**Principe mathématique :**
Les valeurs SHAP sont basées sur la théorie des jeux coopératifs.
La valeur de Shapley φᵢ d'une feature i est définie par :

```
φᵢ(f) = Σ [|S|!(|F|-|S|-1)!/|F|!] × [f(S∪{i}) - f(S)]
```

où S est un sous-ensemble de features, F l'ensemble complet.

**Propriétés garanties :**
- **Efficacité** : la somme des SHAP values = différence entre prédiction et valeur moyenne
- **Symétrie** : deux features avec le même impact ont la même valeur
- **Nullité** : une feature sans impact a une valeur SHAP = 0
- **Additivité** : les contributions s'additionnent de façon cohérente

**Implémentation utilisée :**
```python
import shap
explainer = shap.TreeExplainer(model)  # Optimisé pour LightGBM/XGBoost
shap_values = explainer.shap_values(X)
```

---

### 2.2 LIME (Local Interpretable Model-agnostic Explanations)

**Référence :** Ribeiro et al., 2016 — "Why Should I Trust You?"
*Source : https://arxiv.org/abs/1602.04938*

**Principe :**
LIME approxime localement le modèle complexe par un modèle linéaire simple.
Pour un point x à expliquer, LIME résout :

```
ξ(x) = argmin L(f, g, πₓ) + Ω(g)
```

où f est le modèle complexe, g le modèle simple, πₓ la pondération de proximité.

**Inconvénients identifiés :**
- Instabilité : les résultats varient entre deux runs (perturbation aléatoire)
- Moins fidèle pour les modèles tree-based que SHAP

---

## 3. Comparaison expérimentale

| Critère | SHAP | LIME |
|---|---|---|
| Fondement mathématique | Théorie des jeux (Shapley) | Approximation locale linéaire |
| Cohérence globale | ✅ Garantie | ⚠️ Non garantie |
| Stabilité | ✅ Déterministe | ⚠️ Variable (aléatoire) |
| Vitesse (LightGBM) | ✅ TreeExplainer rapide | ✅ Rapide |
| Model-agnostic | ✅ Oui | ✅ Oui |
| Explication globale | ✅ Summary plot | ❌ Local uniquement |
| Explication locale | ✅ Force plot | ✅ Coefficients linéaires |
| Adoption industrielle | ✅ Très large | ✅ Large |

---

## 4. Résultats sur notre modèle

Sur le modèle LightGBM entraîné (AUC = 0.7789, seuil = 0.4778) :

**Top 5 features les plus importantes (SHAP global) :**
1. `EXT_SOURCE_2` — Score externe 2 (impact négatif sur le risque)
2. `EXT_SOURCE_3` — Score externe 3
3. `EXT_SOURCE_1` — Score externe 1
4. `CREDIT_INCOME_RATIO` — Ratio crédit/revenu (impact positif sur le risque)
5. `DAYS_BIRTH` — Âge du client

**Interprétation :**
- Les scores externes sont les meilleurs prédicteurs : un score élevé → risque faible
- Un ratio crédit/revenu élevé → risque élevé (client sur-endetté)
- Les clients plus âgés ont statistiquement moins de défauts

---

## 5. Recommandation

**SHAP est retenu** comme méthode d'explicabilité principale pour ce projet car :

1. **Fondement mathématique solide** (valeurs de Shapley)
2. **TreeExplainer** optimisé nativement pour LightGBM → très rapide en production
3. **Cohérence** entre explications globales et locales
4. **Déterministe** → même input = même explication (important pour la conformité réglementaire)
5. **Visualisations** intégrées dans le dashboard pour les chargés de clientèle

---

## 6. Références bibliographiques

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
   *Advances in Neural Information Processing Systems*, 30.
   https://arxiv.org/abs/1705.07874

2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?":
   Explaining the predictions of any classifier.
   *KDD 2016*. https://arxiv.org/abs/1602.04938

3. Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.).
   https://christophm.github.io/interpretable-ml-book/

4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
   *KDD 2016*. https://arxiv.org/abs/1603.02754

---

*Document rédigé dans le cadre du Projet 8 OpenClassrooms*
*"Réalisez un dashboard et assurez une veille technique"*
*Auteur : Dominique (Dom130)*
