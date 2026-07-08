#!/usr/bin/env python3
"""
FASE 5 y 6 (v3 - OPCION A, CORREGIDO): fix de tipo de dato en Imagen_asignada
que causaba 0 registros retenidos (el CSV guarda "20160913" como texto pero
pandas al releerlo lo interpretaba como entero, rompiendo el merge con el
diccionario de fechas de imagen).
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

DATASET_CSV = os.path.expanduser("~/analisis_gulupa/10_dataset_modelado_POR_MES.csv")
OUT_DIR = os.path.expanduser("~/analisis_gulupa/30_fase5_6_modelado_OPCIONA")
os.makedirs(OUT_DIR, exist_ok=True)

# --- LECTURA CORREGIDA: forzar texto en Imagen_asignada ---
df = pd.read_csv(DATASET_CSV, dtype={"Imagen_asignada": str})
df["Imagen_asignada"] = df["Imagen_asignada"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)

print("Valores unicos en Imagen_asignada tras correccion:", df["Imagen_asignada"].unique())

df["Fecha_dt"] = pd.to_datetime(df["Fecha"])
df["AnioMes_campo"] = df["Fecha_dt"].dt.to_period("M")

FECHAS_IMAGEN = {"20160913": "2016-09", "20161222": "2016-12", "20170814": "2017-08"}
df["AnioMes_imagen"] = df["Imagen_asignada"].map(FECHAS_IMAGEN)
print("Valores unicos en AnioMes_imagen tras mapeo:", df["AnioMes_imagen"].unique())

df["AnioMes_imagen"] = pd.PeriodIndex(df["AnioMes_imagen"], freq="M")

mismo_mes = df["AnioMes_campo"] == df["AnioMes_imagen"]
df_A = df[mismo_mes].copy()
print(f"\nOPCION A: {len(df_A)}/{len(df)} registros retenidos (mismo mes campo-imagen)")

if len(df_A) == 0:
    raise SystemExit("Aun 0 registros: revisa manualmente los valores impresos arriba de "
                      "Imagen_asignada y AnioMes_campo para verificar formato de fecha.")

predictor_cols = [c for c in df_A.columns if c.endswith("_asignado")]
targets = [c for c in ["LAI4", "LAI5", "%Cob"] if c in df_A.columns]
print(f"Predictores ({len(predictor_cols)}): {predictor_cols}")
print(f"Targets: {targets}")

resultados_finales = []

for target in targets:
    print(f"\n{'='*60}\nTARGET: {target}\n{'='*60}")
    data = df_A.dropna(subset=predictor_cols + [target]).copy()
    print(f"Registros disponibles para {target}: {len(data)}")
    if len(data) < 10:
        print(f"[SKIP] Muy pocos registros ({len(data)}) para entrenar/evaluar {target}.")
        continue

    X = data[predictor_cols].values
    y = data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    modelos = {}
    ridge = Ridge(alpha=1.0, random_state=42); ridge.fit(X_train_s, y_train); modelos["Ridge"] = ridge
    lasso = Lasso(alpha=0.1, random_state=42, max_iter=5000); lasso.fit(X_train_s, y_train); modelos["Lasso"] = lasso
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train); modelos["RandomForest"] = rf

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        kf = KFold(n_splits=min(5, len(X_train)), shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    print(f"Mejores hiperparametros XGBoost ({target}): {study.best_params}")
    xgb_final = xgb.XGBRegressor(**study.best_params, random_state=42, verbosity=0)
    xgb_final.fit(X_train, y_train); modelos["XGBoost"] = xgb_final

    for nombre, modelo in modelos.items():
        X_te = X_test_s if nombre in ["Ridge", "Lasso"] else X_test
        y_pred = modelo.predict(X_te)
        r2 = r2_score(y_test, y_pred); mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        resultados_finales.append({"Target": target, "Modelo": nombre, "R2": round(r2,3),
                                    "MAE": round(mae,3), "RMSE": round(rmse,3), "N": len(data)})
        print(f"  {nombre:15s} R2={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")

    joblib.dump(modelos["XGBoost"], os.path.join(OUT_DIR, f"modelo_XGBoost_{target.replace('%','pct')}.pkl"))
    joblib.dump(scaler, os.path.join(OUT_DIR, f"scaler_{target.replace('%','pct')}.pkl"))

    explainer = shap.TreeExplainer(xgb_final)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, X_test, feature_names=predictor_cols, show=False)
    plt.title(f"SHAP - {target} (Opcion A, n={len(data)})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"Fig_SHAP_{target.replace('%','pct')}.png"), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()

    coefs = pd.DataFrame({"Predictor": predictor_cols, "Ridge_coef": modelos["Ridge"].coef_,
                          "Lasso_coef": modelos["Lasso"].coef_}).sort_values("Lasso_coef", key=abs, ascending=False)
    coefs.to_csv(os.path.join(OUT_DIR, f"Coeficientes_{target.replace('%','pct')}.csv"), index=False)
    print(f"  Coeficientes:\n{coefs.to_string(index=False)}")

tabla_resultados = pd.DataFrame(resultados_finales)
tabla_resultados.to_csv(os.path.join(OUT_DIR, "Tabla_comparacion_modelos_OPCIONA.csv"), index=False)
print(f"\n{'='*60}\n=== COMPARACION FINAL (OPCION A, solo mismo mes) ===")
print(tabla_resultados.to_string(index=False))
print(f"\nSalida: {OUT_DIR}")
