"""Predictive Modeling Using Machine Learning
Part A: Classification - predict whether a tumour is malignant or benign (Breast Cancer Wisconsin dataset)
Part B: Regression     - predict diabetes disease progression (Diabetes dataset)
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)

sns.set_theme(style="whitegrid")
OUT, SEED = "outputs/", 42
report = []

# =============== PART A: CLASSIFICATION ===============
bc = load_breast_cancer(as_frame=True)
X, y = bc.data, bc.target                       # target: 0 = malignant, 1 = benign
pd.concat([X, y.rename("target")], axis=1).to_csv("data/breast_cancer.csv", index=False)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
}
rows, fitted, probas = [], {}, {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    pred, proba = m.predict(X_te), m.predict_proba(X_te)[:, 1]
    fitted[name], probas[name] = m, proba
    rows.append({"Model": name,
                 "CV Accuracy (5-fold)": cross_val_score(m, X_tr, y_tr, cv=5).mean(),
                 "Test Accuracy": accuracy_score(y_te, pred),
                 "Precision": precision_score(y_te, pred),
                 "Recall": recall_score(y_te, pred),
                 "F1": f1_score(y_te, pred),
                 "ROC-AUC": roc_auc_score(y_te, proba)})
res = pd.DataFrame(rows).set_index("Model").round(4)
res.to_csv(OUT + "classification_results.csv")
best = res["ROC-AUC"].idxmax()
report.append("PART A - CLASSIFICATION (Breast Cancer)\n" + res.to_string())
report.append(f"\nBest model by ROC-AUC: {best}\n\n" + classification_report(
    y_te, fitted[best].predict(X_te), target_names=["malignant", "benign"]))

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, m) in zip(axes, fitted.items()):
    cm = confusion_matrix(y_te, m.predict(X_te))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold"); plt.tight_layout()
plt.savefig(OUT + "01_confusion_matrices.png", dpi=150); plt.close()

# ROC curves
plt.figure(figsize=(7, 6))
for name, p in probas.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_te, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves")
plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(OUT + "02_roc_curves.png", dpi=150); plt.close()

# Model comparison
res[["Test Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].plot.bar(figsize=(9, 5), ylim=(0.85, 1.0), colormap="viridis")
plt.title("Model Comparison"); plt.xticks(rotation=0); plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(OUT + "03_model_comparison.png", dpi=150); plt.close()

# Feature importance (Random Forest)
imp = pd.Series(fitted["Random Forest"].feature_importances_, index=X.columns).nlargest(10)
plt.figure(figsize=(8, 5)); sns.barplot(x=imp.values, y=imp.index, color="#3b528b")
plt.title("Top 10 Features - Random Forest"); plt.xlabel("Importance"); plt.tight_layout()
plt.savefig(OUT + "04_feature_importance.png", dpi=150); plt.close()
report.append("\nTop 5 features (Random Forest): " + ", ".join(imp.index[:5]))

# =============== PART B: REGRESSION ===============
db = load_diabetes(as_frame=True)
Xr, yr = db.data, db.target
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)
reg_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=SEED),
}
rrows, rpreds = [], {}
for name, m in reg_models.items():
    m.fit(Xr_tr, yr_tr); p = m.predict(Xr_te); rpreds[name] = p
    rrows.append({"Model": name, "MAE": mean_absolute_error(yr_te, p),
                  "RMSE": np.sqrt(mean_squared_error(yr_te, p)), "R2": r2_score(yr_te, p)})
rres = pd.DataFrame(rrows).set_index("Model").round(3)
rres.to_csv(OUT + "regression_results.csv")
report.append("\n\nPART B - REGRESSION (Diabetes progression)\n" + rres.to_string())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, (name, p) in zip(axes, rpreds.items()):
    ax.scatter(yr_te, p, alpha=.6, color="#21918c")
    lim = [yr_te.min(), yr_te.max()]; ax.plot(lim, lim, "r--")
    ax.set_title(f"{name}  (R2 = {r2_score(yr_te, p):.2f})"); ax.set_xlabel("Actual")
axes[0].set_ylabel("Predicted"); plt.suptitle("Regression: Actual vs Predicted", fontsize=15, fontweight="bold")
plt.tight_layout(); plt.savefig(OUT + "05_regression_actual_vs_predicted.png", dpi=150); plt.close()

text = "\n".join(report)
open(OUT + "model_report.txt", "w").write(text)
print(text)
