"""
train_model.py (v9.0 - XGBoost + LightGBM + Isolation Forest Feature)
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import optuna

warnings.filterwarnings("ignore")

OPTUNA_TRIALS = 100
N_SPLITS = 5
RANDOM_STATE = 42

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models/saved")
REPORTS_DIR = Path("outputs/reports")

for d in [MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROCESSED_DIR / "train_balanced.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
FEAT_COLS_PATH = PROCESSED_DIR / "feature_cols.json"


def load_and_preprocess():
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)

    cat_features = train.select_dtypes(include=['object']).columns.tolist()
    for col in cat_features:
        if col not in ["user_id"]:
            le = LabelEncoder()
            combined = pd.concat([train[col], test[col]], axis=0).astype(str)
            le.fit(combined)
            train[col] = le.transform(train[col].astype(str))
            test[col] = le.transform(test[col].astype(str))

    if FEAT_COLS_PATH.exists():
        with open(FEAT_COLS_PATH) as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [c for c in train.columns if c not in ["user_id", "label"]]

    X_train = train[feature_cols].values
    y_train = train["label"].values
    X_test = test[feature_cols].values
    y_test = test["label"].values
    
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    train_spw = neg_count / pos_count if pos_count > 0 else 1.0
    
    print("\n【特徵增強】訓練 Isolation Forest 異常檢測")
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=pos_count / len(y_train),
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    
    train_anomaly_scores = iso_forest.decision_function(X_train)
    test_anomaly_scores = iso_forest.decision_function(X_test)
    
    train_min, train_max = train_anomaly_scores.min(), train_anomaly_scores.max()
    train_anomaly_norm = 1 - (train_anomaly_scores - train_min) / (train_max - train_min)
    test_anomaly_norm = 1 - (test_anomaly_scores - train_min) / (train_max - train_min)
    
    X_train = np.column_stack([X_train, train_anomaly_norm])
    X_test = np.column_stack([X_test, test_anomaly_norm])
    feature_cols.append("isolation_forest_score")
    
    with open(MODELS_DIR / "isolationforest.pkl", "wb") as f:
        pickle.dump(iso_forest, f)
    print(f"✅ Isolation Forest 已訓練並加入特徵")

    return X_train, y_train, X_test, y_test, feature_cols, train_spw, iso_forest


def find_best_threshold_refined(y_true, y_prob, name="Model", verbose=True):
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    if verbose:
        print(f" 🔍 [{name}] 最佳 F1: {best_f1:.4f} | 最佳閾值: {best_thresh:.2f}")
    return best_thresh, best_f1


def tune_hyperparams_v2(X_train, y_train, train_spw, model_type="lgbm"):
    def obj(trial):
        if model_type == "lgbm":
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", train_spw * 0.5, train_spw * 1.5),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            }
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", train_spw * 0.5, train_spw * 1.5),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            if model_type == "lgbm":
                m = lgb.LGBMClassifier(**params)
            else:
                m = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
            
            m.fit(X_train[tr_idx], y_train[tr_idx])
            prob = m.predict_proba(X_train[val_idx])[:, 1]
            _, current_f1 = find_best_threshold_refined(y_train[val_idx], prob, verbose=False)
            scores.append(current_f1)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(obj, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    print(f"\n✅ [{model_type.upper()}] 最佳參數: {study.best_params}")
    print(f"   最佳 CV F1: {study.best_value:.4f}")
    return study.best_params


def main():
    print("=== BitoGuard v9.0 (XGB + LGBM + IsoForest Feature) ===")
    X_train, y_train, X_test, y_test, feat_cols, train_spw, iso_forest = load_and_preprocess()

    print("\n【階段 1】超參數調優")
    print("-" * 60)
    lgbm_params = tune_hyperparams_v2(X_train, y_train, train_spw, "lgbm")
    xgb_params = tune_hyperparams_v2(X_train, y_train, train_spw, "xgb")

    print("\n【階段 2】5-Fold 交叉驗證訓練")
    print("-" * 60)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    lgbm_oof = np.zeros(len(y_train))
    xgb_oof = np.zeros(len(y_train))
    lgbm_test_probs = []
    xgb_test_probs = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n  Fold {fold_idx}/{N_SPLITS}")
        
        m_lgb = lgb.LGBMClassifier(**lgbm_params)
        m_lgb.fit(X_train[tr_idx], y_train[tr_idx])
        lgbm_oof[val_idx] = m_lgb.predict_proba(X_train[val_idx])[:, 1]
        lgbm_test_probs.append(m_lgb.predict_proba(X_test)[:, 1])
        
        m_xgb = xgb.XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric='logloss')
        m_xgb.fit(X_train[tr_idx], y_train[tr_idx])
        xgb_oof[val_idx] = m_xgb.predict_proba(X_train[val_idx])[:, 1]
        xgb_test_probs.append(m_xgb.predict_proba(X_test)[:, 1])

    print("\n【階段 3】各模型 OOF 表現")
    print("-" * 60)
    for name, oof_prob in [("LightGBM", lgbm_oof), ("XGBoost", xgb_oof)]:
        find_best_threshold_refined(y_train, oof_prob, name)

    print("\n【階段 4】Ensemble 集成")
    print("-" * 60)
    
    oof_ensemble_simple = (lgbm_oof + xgb_oof) / 2
    test_ensemble_simple = (np.mean(lgbm_test_probs, axis=0) + np.mean(xgb_test_probs, axis=0)) / 2
    
    X_stack_train = np.column_stack([lgbm_oof, xgb_oof])
    X_stack_test = np.column_stack([np.mean(lgbm_test_probs, axis=0), np.mean(xgb_test_probs, axis=0)])
    
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    meta_model.fit(X_stack_train, y_train)
    oof_ensemble_stacking = meta_model.predict_proba(X_stack_train)[:, 1]
    test_ensemble_stacking = meta_model.predict_proba(X_stack_test)[:, 1]
    
    _, f1_simple = find_best_threshold_refined(y_train, oof_ensemble_simple, "簡單平均")
    _, f1_stacking = find_best_threshold_refined(y_train, oof_ensemble_stacking, "Stacking")
    
    if f1_stacking > f1_simple:
        print(f"\n✅ 選擇 Stacking 方法（F1: {f1_stacking:.4f} > {f1_simple:.4f}）")
        oof_final = oof_ensemble_stacking
        test_final = test_ensemble_stacking
        ensemble_method = "stacking"
    else:
        print(f"\n✅ 選擇簡單平均方法（F1: {f1_simple:.4f} >= {f1_stacking:.4f}）")
        oof_final = oof_ensemble_simple
        test_final = test_ensemble_simple
        ensemble_method = "simple_average"
        meta_model = None
    
    best_thresh, oof_f1 = find_best_threshold_refined(y_train, oof_final, "Final Ensemble (OOF)")
    
    print("\n【階段 5】測試集最終評估")
    print("-" * 60)
    y_pred = (test_final >= best_thresh).astype(int)
    
    test_f1 = f1_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩陣:")
    print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    
    print("\n【階段 6】儲存模型")
    print("-" * 60)
    
    final_model_data = {
        "meta": meta_model,
        "ensemble_method": ensemble_method,
        "thresh": best_thresh,
        "lgbm_params": lgbm_params,
        "xgb_params": xgb_params,
        "isolation_forest": iso_forest,
        "feature_cols": feat_cols,
        "oof_f1": oof_f1,
        "test_f1": test_f1,
    }
    
    with open(MODELS_DIR / "f1_optimized_model.pkl", "wb") as f:
        pickle.dump(final_model_data, f)
    
    threshold_info = {
        "ensemble_method": ensemble_method,
        "best_threshold": float(best_thresh),
        "oof_f1": float(oof_f1),
        "test_f1": float(test_f1),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
    }
    with open(MODELS_DIR / "best_thresholds.json", "w") as f:
        json.dump(threshold_info, f, indent=2)
    
    print(f"✅ 模型已儲存至: {MODELS_DIR / 'f1_optimized_model.pkl'}")
    print(f"✅ 閾值資訊已儲存至: {MODELS_DIR / 'best_thresholds.json'}")
    print(f"\n🎯 最終架構: XGBoost + LightGBM + Isolation Forest 特徵 + {ensemble_method.upper()}")


if __name__ == "__main__":
    main()
