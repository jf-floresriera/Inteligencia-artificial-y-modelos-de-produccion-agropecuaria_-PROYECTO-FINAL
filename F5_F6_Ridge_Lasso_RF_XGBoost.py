#!/usr/bin/env python3
"""
FASE 5 y 6: Regresion regularizada (Ridge/Lasso) + Modelado supervisado
(Random Forest / XGBoost tuneado con Optuna) - Cultivo de CAUCHO, unicamente Lote 5.

Target: LAI4, LAI5, %Cob (ground truth de campo)
Predictores: NDVI, SAVI, EVI, NDRE, GNDVI x 3 fechas (20160913, 20161222, 20170814)

Entrada: 11_dataset_LIMPIO_sin_atipicos.csv (generado en Fase 3c)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import optuna
import shap
import joblib

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"]})
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATASET_CSV = os.path.expanduser("~/analisis_gulupa/11_dataset_LIMPIO_sin_atipicos.csv")
OUT_DIR = os.path.expanduser("~/analisis_gulupa/30_fase5_6_modelado")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATASET_CSV)
predictor_cols = [c for c in df.columns if any(idx in c for idx in ["NDVI_","SAVI_","EVI_","NDRE_","GNDVI_"])]
targets = [c for c in ["LAI4", "LAI5", "%Cob"] if c in df.columns]
print(f"Predictores ({len(predictor_cols)}): {predictor_cols}")
print(f"Targets: {targets}")

resultados_finales = []

for target in targets:
    print(f"\n{'='*60}\nTARGET: {target}\n{'='*60}")
    data = df.dropna(subset=predictor_cols + [target]).copy()
    X = data[predictor_cols].values
    y = data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    modelos = {}

    # --- Ridge ---
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_s, y_train)
    modelos["Ridge"] = ridge

    # --- Lasso ---
    lasso = Lasso(alpha=0.1, random_state=42, max_iter=5000)
    lasso.fit(X_train_s, y_train)
    modelos["Lasso"] = lasso

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    modelos["RandomForest"] = rf

    # --- XGBoost tuneado con Optuna ---
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    print(f"Mejores hiperparametros XGBoost ({target}): {study.best_params}")

    xgb_final = xgb.XGBRegressor(**study.best_params, random_state=42, verbosity=0)
    xgb_final.fit(X_train, y_train)
    modelos["XGBoost"] = xgb_final

    for nombre, modelo in modelos.items():
        X_te = X_test_s if nombre in ["Ridge", "Lasso"] else X_test
        y_pred = modelo.predict(X_te)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        resultados_finales.append({"Target": target, "Modelo": nombre, "R2": round(r2,3),
                                    "MAE": round(mae,3), "RMSE": round(rmse,3)})
        print(f"  {nombre:15s} R2={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")

    joblib.dump(modelos["XGBoost"], os.path.join(OUT_DIR, f"modelo_XGBoost_{target.replace('%','pct')}.pkl"))
    joblib.dump(scaler, os.path.join(OUT_DIR, f"scaler_{target.replace('%','pct')}.pkl"))

    # --- SHAP para el mejor modelo (XGBoost) ---
    explainer = shap.TreeExplainer(xgb_final)
    shap_values = explainer.shap_values(X_test)
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, feature_names=predictor_cols, show=False)
    plt.title(f"SHAP - Importancia de variables para {target}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"Fig_SHAP_{target.replace('%','pct')}.png"), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()

    # --- Coeficientes Ridge/Lasso (interpretabilidad) ---
    coefs = pd.DataFrame({
        "Predictor": predictor_cols,
        "Ridge_coef": modelos["Ridge"].coef_,
        "Lasso_coef": modelos["Lasso"].coef_,
    }).sort_values("Lasso_coef", key=abs, ascending=False)
    coefs.to_csv(os.path.join(OUT_DIR, f"Coeficientes_Ridge_Lasso_{target.replace('%','pct')}.csv"), index=False)
    n_vars_lasso = (coefs["Lasso_coef"] != 0).sum()
    print(f"  Lasso selecciono {n_vars_lasso}/{len(predictor_cols)} variables (coef != 0)")

tabla_resultados = pd.DataFrame(resultados_finales)
tabla_resultados.to_csv(os.path.join(OUT_DIR, "Tabla_comparacion_modelos.csv"), index=False)
print(f"\n{'='*60}\n=== COMPARACION FINAL DE MODELOS ===")
print(tabla_resultados.to_string(index=False))
print(f"\nSalida completa en: {OUT_DIR}")
