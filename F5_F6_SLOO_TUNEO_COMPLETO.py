#!/usr/bin/env python3
"""
FASE 5-6 (v5 - DEFINITIVA CON TUNEO COMPLETO): Spatial Leave-One-Out (SLOO)
+ Optuna anidado para LOS 4 MODELOS (Ridge, Lasso, Random Forest, XGBoost).

Corrige la limitacion anterior donde solo XGBoost era tuneado. Ahora los 4
modelos son optimizados con la misma tecnica de busqueda bayesiana (Optuna),
usando K-Fold interno (5-fold) SOLO sobre el set de entrenamiento de cada
iteracion SLOO externa (nested cross-validation), garantizando comparacion
justa y evitando fuga de informacion.

Hiperparametros tuneados por modelo:
  Ridge          -> alpha
  Lasso          -> alpha
  Random Forest  -> n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features
  XGBoost        -> n_estimators, max_depth, learning_rate, subsample,
                     colsample_bytree, reg_alpha, reg_lambda

Entrada: 10_dataset_modelado_POR_MES.csv (con columnas '_asignado', Latitud, Longitud)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import optuna
from scipy.spatial.distance import cdist

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"]})
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATASET_CSV = os.path.expanduser("~/analisis_gulupa/10_dataset_modelado_POR_MES.csv")
OUT_DIR = os.path.expanduser("~/analisis_gulupa/30_fase5_6_SLOO_TUNEO_COMPLETO")
os.makedirs(OUT_DIR, exist_ok=True)

RADIO_EXCLUSION_M = 20
N_TRIALS_OPTUNA = 20

df = pd.read_csv(DATASET_CSV, dtype={"Imagen_asignada": str})
df["Fecha_dt"] = pd.to_datetime(df["Fecha"])
df["AnioMes_campo"] = df["Fecha_dt"].dt.to_period("M")
FECHAS_IMAGEN = {"20160913": "2016-09", "20161222": "2016-12", "20170814": "2017-08"}
df["AnioMes_imagen"] = pd.PeriodIndex(df["Imagen_asignada"].map(FECHAS_IMAGEN), freq="M")
df_A = df[df["AnioMes_campo"] == df["AnioMes_imagen"]].copy()
print(f"Registros retenidos (mismo mes campo-imagen): {len(df_A)}/{len(df)}")

predictor_cols = [c for c in df_A.columns if c.endswith("_asignado")]
targets = [c for c in ["LAI4", "LAI5", "%Cob"] if c in df_A.columns]

def coords_a_metros(lat, lon):
    x = lon * 111320 * np.cos(np.radians(lat.mean()))
    y = lat * 110540
    return np.column_stack([x, y])

def tunear_ridge(X_train, y_train):
    def obj(trial):
        alpha = trial.suggest_float("alpha", 0.001, 100, log=True)
        model = Ridge(alpha=alpha, random_state=42)
        kf = KFold(n_splits=min(5, len(X_train)), shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=N_TRIALS_OPTUNA, show_progress_bar=False)
    return Ridge(alpha=study.best_params["alpha"], random_state=42)

def tunear_lasso(X_train, y_train):
    def obj(trial):
        alpha = trial.suggest_float("alpha", 0.0001, 10, log=True)
        model = Lasso(alpha=alpha, random_state=42, max_iter=5000)
        kf = KFold(n_splits=min(5, len(X_train)), shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=N_TRIALS_OPTUNA, show_progress_bar=False)
    return Lasso(alpha=study.best_params["alpha"], random_state=42, max_iter=5000)

def tunear_rf(X_train, y_train):
    def obj(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        kf = KFold(n_splits=min(5, len(X_train)), shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=N_TRIALS_OPTUNA, show_progress_bar=False)
    return RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)

def tunear_xgb(X_train, y_train):
    def obj(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        kf = KFold(n_splits=min(5, len(X_train)), shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=N_TRIALS_OPTUNA, show_progress_bar=False)
    return xgb.XGBRegressor(**study.best_params, random_state=42, verbosity=0)

resultados_finales = []
hiperparametros_log = []

for target in targets:
    print(f"\n{'='*60}\nTARGET: {target}  (SLOO, radio={RADIO_EXCLUSION_M}m, tuneo completo)\n{'='*60}")
    data = df_A.dropna(subset=predictor_cols + [target, "Latitud", "Longitud"]).copy().reset_index(drop=True)
    n = len(data)
    print(f"Registros disponibles: {n}")
    if n < 15:
        print(f"[SKIP] Muy pocos registros ({n}).")
        continue

    X_all = data[predictor_cols].values
    y_all = data[target].values
    coords_m = coords_a_metros(data["Latitud"], data["Longitud"])
    dist_matrix = cdist(coords_m, coords_m)

    predicciones = {"Ridge": [], "Lasso": [], "RandomForest": [], "XGBoost": []}
    y_true_list = []

    for i in range(n):
        vecinos_excluidos = np.where(dist_matrix[i] <= RADIO_EXCLUSION_M)[0]
        idx_train = np.setdiff1d(np.arange(n), vecinos_excluidos)
        idx_test = np.array([i])
        if len(idx_train) < 10:
            continue

        X_train, X_test = X_all[idx_train], X_all[idx_test]
        y_train, y_test = y_all[idx_train], y_all[idx_test]
        y_true_list.append(y_test[0])

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ridge_tuned = tunear_ridge(X_train_s, y_train); ridge_tuned.fit(X_train_s, y_train)
        predicciones["Ridge"].append(ridge_tuned.predict(X_test_s)[0])

        lasso_tuned = tunear_lasso(X_train_s, y_train); lasso_tuned.fit(X_train_s, y_train)
        predicciones["Lasso"].append(lasso_tuned.predict(X_test_s)[0])

        rf_tuned = tunear_rf(X_train, y_train); rf_tuned.fit(X_train, y_train)
        predicciones["RandomForest"].append(rf_tuned.predict(X_test)[0])

        xgb_tuned = tunear_xgb(X_train, y_train); xgb_tuned.fit(X_train, y_train)
        predicciones["XGBoost"].append(xgb_tuned.predict(X_test)[0])

        if i == 0:
            hiperparametros_log.append({"Target": target, "Fold": i, "Ridge_alpha": ridge_tuned.alpha,
                "Lasso_alpha": lasso_tuned.alpha, "RF_params": rf_tuned.get_params(),
                "XGB_params": xgb_tuned.get_params()})

        if i % 20 == 0:
            print(f"  Fold {i+1}/{n} completado...")

    y_true_arr = np.array(y_true_list)
    for nombre, preds in predicciones.items():
        preds_arr = np.array(preds)
        r2 = r2_score(y_true_arr, preds_arr)
        mae = mean_absolute_error(y_true_arr, preds_arr)
        rmse = np.sqrt(mean_squared_error(y_true_arr, preds_arr))
        resultados_finales.append({"Target": target, "Modelo": nombre, "R2_SLOO": round(r2,3),
                                    "MAE_SLOO": round(mae,3), "RMSE_SLOO": round(rmse,3), "N_folds": len(y_true_arr)})
        print(f"  {nombre:15s} R2={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  (n_folds={len(y_true_arr)})")

tabla_final = pd.DataFrame(resultados_finales)
tabla_final.to_csv(os.path.join(OUT_DIR, "Tabla_SLOO_TUNEO_COMPLETO.csv"), index=False)
print(f"\n{'='*60}\n=== RESULTADOS FINALES (4 modelos tuneados, SLOO) ===")
print(tabla_final.to_string(index=False))
print(f"\nSalida completa en: {OUT_DIR}")
print("\nNOTA: este script tunea los 4 modelos dentro de cada uno de los 90 folds SLOO,")
print("por lo que el tiempo de ejecucion sera considerablemente mayor que la version anterior.")
