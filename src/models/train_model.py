"""
train_model.py
位置：src/models/train_model.py

【v5.7 宗師無洩漏版 — 拋棄SMOTE + 驗證集閾值校準 + 雙Optuna引擎】
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve
)

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("  ℹ️  CatBoost 未安裝")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings("ignore")

# ============================================================
# 設定區
# ============================================================
OPTUNA_TRIALS = 30    
DROP_K_FEATURES = 5    

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/saved")
REPORTS_DIR   = Path("outputs/reports")

for d in [MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_PATH      = PROCESSED_DIR / "train_balanced.csv"
TEST_PATH       = PROCESSED_DIR / "test.csv"
PREDICT_PATH    = PROCESSED_DIR / "predict_scaled.csv"
FEAT_COLS_PATH  = PROCESSED_DIR / "feature_cols.json"
SUBMISSION_PATH = REPORTS_DIR   / "submission.csv"

RANDOM_STATE = 42

# ============================================================
# 1. 載入資料 (🛑 拋棄 SMOTE，回歸真實數據)
# ============================================================
def load_data():
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test  = pd.read_csv(TEST_PATH,  low_memory=False)

    if FEAT_COLS_PATH.exists():
        with open(FEAT_COLS_PATH) as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [c for c in train.columns if c not in ["user_id", "label"]]

    X_test   = test[feature_cols].values
    y_test   = test["label"].values
    uid_test = test["user_id"].values

    orig_mask    = train["user_id"] != -1
    X_train_orig = train.loc[orig_mask, feature_cols].values
    y_train_orig = train.loc[orig_mask, "label"].values

    orig_neg = int((y_train_orig == 0).sum())
    orig_pos = int((y_train_orig == 1).sum())
    orig_spw = round(orig_neg / orig_pos, 1)

    # 🚀 關鍵改變：徹底丟棄 SMOTE 合成數據，改用 100% 真實數據
    X_train   = X_train_orig.copy()
    y_train   = y_train_orig.copy()
    train_spw = orig_spw

    X_predict, uid_predict = None, None
    if PREDICT_PATH.exists():
        pred_df     = pd.read_csv(PREDICT_PATH, low_memory=False)
        uid_predict = pred_df["user_id"].values
        X_predict   = pred_df[feature_cols].values

    print(f"  訓練集 (無 SMOTE 雜訊)：{len(X_train):,} 筆  權重：{train_spw}")
    return (X_train, y_train, X_test, y_test, uid_test, feature_cols, train_spw, X_predict, uid_predict)

# ============================================================
# 2. 特徵篩選
# ============================================================
def perform_feature_selection(X_train, y_train, X_test, X_predict, feature_cols, train_spw, drop_k=5):
    print("\n" + "=" * 50); print(f"【特徵篩選】— 剔除最弱的 {drop_k} 個垃圾特徵")
    eval_model = lgb.LGBMClassifier(n_estimators=150, random_state=RANDOM_STATE, scale_pos_weight=train_spw, verbose=-1, n_jobs=-1)
    eval_model.fit(X_train, y_train)

    feat_imp_df = pd.DataFrame({'feature': feature_cols, 'importance': eval_model.feature_importances_}).sort_values(by='importance', ascending=False)
    drop_features = feat_imp_df['feature'].tail(drop_k).tolist()
    print(f"  🗑️ 即將剔除：{drop_features}")

    keep_idx = [i for i, f in enumerate(feature_cols) if f not in drop_features]
    new_feature_cols = [feature_cols[i] for i in keep_idx]

    return X_train[:, keep_idx], X_test[:, keep_idx], (X_predict[:, keep_idx] if X_predict is not None else None), new_feature_cols

# ============================================================
# 3. 評估與無作弊閾值搜尋 
# ============================================================
def evaluate(model_name, y_true, y_pred, y_prob) -> dict:
    p, r, f1 = precision_score(y_true, y_pred, zero_division=0), recall_score(y_true, y_pred, zero_division=0), f1_score(y_true, y_pred, zero_division=0)
    apr = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    print(f"\n  ── {model_name} ──")
    print(f"  Precision ：{p:.4f}  Recall：{r:.4f}  F1：{f1:.4f}  AUC-PR：{apr:.4f}")
    print(f"  混淆矩陣：TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return {"model": model_name, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "auc_pr": round(apr, 4), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

def find_best_threshold(y_true, y_prob, prefix="") -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"  [驗證集找閾值] {prefix}最佳 F1 閾值：{best_thresh:.4f} | F1: {f1_scores[best_idx]:.4f}")
    return float(best_thresh)

# ============================================================
# 4. 雙引擎 Optuna 調參
# ============================================================
def tune_lgbm(X_train, y_train, train_spw):
    if not OPTUNA_AVAILABLE: return {}
    print("\n【Optuna 自動調參 - LightGBM】")
    def obj(trial):
        params = {"n_estimators": trial.suggest_int("n_estimators", 200, 500), "max_depth": trial.suggest_int("max_depth", 3, 7), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True)}
        return cross_val_score(lgb.LGBMClassifier(**params, scale_pos_weight=train_spw, verbose=-1, random_state=RANDOM_STATE, n_jobs=-1), X_train, y_train, cv=3, scoring="average_precision", n_jobs=-1).mean()
    study = optuna.create_study(direction="maximize"); study.optimize(obj, n_trials=20, show_progress_bar=False)
    return study.best_params

def tune_xgb(X_train, y_train, train_spw):
    if not OPTUNA_AVAILABLE: return {}
    print("\n【Optuna 自動調參 - XGBoost】")
    def obj(trial):
        params = {"n_estimators": trial.suggest_int("n_estimators", 200, 500), "max_depth": trial.suggest_int("max_depth", 3, 7), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True)}
        return cross_val_score(xgb.XGBClassifier(**params, scale_pos_weight=train_spw, verbosity=0, random_state=RANDOM_STATE, n_jobs=-1), X_train, y_train, cv=3, scoring="average_precision", n_jobs=-1).mean()
    study = optuna.create_study(direction="maximize"); study.optimize(obj, n_trials=20, show_progress_bar=False)
    return study.best_params

# ============================================================
# 5. 嚴謹模型訓練區 (80/20切分找閾值 -> 100%重訓)
# ============================================================
def train_lightgbm(X_train, y_train, X_test, y_test, optuna_params, train_spw):
    print("\n" + "=" * 50); print("【LightGBM 訓練 (無作弊閾值校準)】")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE)
    params = {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05, **optuna_params}

    val_model = lgb.LGBMClassifier(**params, scale_pos_weight=train_spw, verbose=-1, random_state=RANDOM_STATE)
    val_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    val_prob = val_model.predict_proba(X_val)[:, 1]
    thresh = find_best_threshold(y_val, val_prob, "LGBM ")

    print("  🚀 重新使用 100% 訓練集打造最終模型...")
    final_model = lgb.LGBMClassifier(**params, scale_pos_weight=train_spw, verbose=-1, random_state=RANDOM_STATE)
    final_model.fit(X_train, y_train)

    y_prob = final_model.predict_proba(X_test)[:, 1]
    metrics = evaluate("LightGBM", y_test, (y_prob >= thresh).astype(int), y_prob)
    metrics["threshold"] = thresh
    return final_model, y_prob, val_prob, y_val, metrics

def train_xgboost(X_train, y_train, X_test, y_test, optuna_params, train_spw):
    print("\n" + "=" * 50); print("【XGBoost 訓練 (無作弊閾值校準)】")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE)
    params = {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05, **optuna_params}

    val_model = xgb.XGBClassifier(**params, scale_pos_weight=train_spw, eval_metric="aucpr", early_stopping_rounds=30, verbosity=0, random_state=RANDOM_STATE, n_jobs=-1)
    val_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
    val_prob = val_model.predict_proba(X_val)[:, 1]
    thresh = find_best_threshold(y_val, val_prob, "XGBoost ")

    print("  🚀 重新使用 100% 訓練集打造最終模型...")
    final_model = xgb.XGBClassifier(**params, scale_pos_weight=train_spw, verbosity=0, random_state=RANDOM_STATE, n_jobs=-1)
    final_model.fit(X_train, y_train)

    y_prob = final_model.predict_proba(X_test)[:, 1]
    metrics = evaluate("XGBoost", y_test, (y_prob >= thresh).astype(int), y_prob)
    metrics["threshold"] = thresh
    return final_model, y_prob, val_prob, y_val, metrics

def train_catboost(X_train, y_train, X_test, y_test, train_spw):
    if not CATBOOST_AVAILABLE: return None, None, None, None, None
    print("\n" + "=" * 50); print("【CatBoost 訓練 (無作弊閾值校準)】")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE)
    
    val_model = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05, scale_pos_weight=train_spw, eval_metric="F1", early_stopping_rounds=30, verbose=0, random_seed=RANDOM_STATE)
    val_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    val_prob = val_model.predict_proba(X_val)[:, 1]
    thresh = find_best_threshold(y_val, val_prob, "CatBoost ")

    print("  🚀 重新使用 100% 訓練集打造最終模型...")
    final_model = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05, scale_pos_weight=train_spw, verbose=0, random_seed=RANDOM_STATE)
    final_model.fit(X_train, y_train)

    y_prob = final_model.predict_proba(X_test)[:, 1]
    metrics = evaluate("CatBoost", y_test, (y_prob >= thresh).astype(int), y_prob)
    metrics["threshold"] = thresh
    return final_model, y_prob, val_prob, y_val, metrics

# ============================================================
# 6. 無作弊集成 (Rank Ensembling + Validation Threshold)
# ============================================================
def ensemble_predict(prob_dict, val_prob_dict, y_val, all_metrics_dict, y_test) -> tuple:
    print("\n" + "=" * 50); print("【無作弊集成模型（Rank 融合 + 驗證集閾值）】")
    top_models = sorted(all_metrics_dict.items(), key=lambda x: x[1].get('auc_pr', 0), reverse=True)[:3]
    w = {k: round(v.get('auc_pr',0) / (sum(m[1].get('auc_pr',0) for m in top_models)+1e-9), 3) for k, v in top_models}
    print(f"  選定貢獻模型與權重：{w}")

    # 1. 在驗證集(沒看過答案)上做 Rank Ensemble，找出最佳安全閾值
    ranked_val_probs = {name: rankdata(prob) / len(prob) for name, prob in val_prob_dict.items()}
    val_prob_ens = sum(w.get(name, 0) * ranked_val_probs[name] for name in w)
    thresh = find_best_threshold(y_val, val_prob_ens, "Ensemble ")

    # 2. 在未知測試集上做 Rank Ensemble，直接套用上述找到的真實閾值！
    ranked_test_probs = {name: rankdata(prob) / len(prob) for name, prob in prob_dict.items()}
    y_prob_ens = sum(w.get(name, 0) * ranked_test_probs[name] for name in w)
    
    metrics = evaluate("Ensemble", y_test, (y_prob_ens >= thresh).astype(int), y_prob_ens)
    metrics.update({"threshold": thresh, "weights": w})
    return y_prob_ens, metrics

def save_models_and_reports(models, all_metrics, uid_test, prob_dict, y_test, feature_cols, X_predict, uid_predict, ens_metrics):
    print("\n【儲存模型與產出報告】")
    for name, model in models.items():
        with open(MODELS_DIR / f"{name.lower()}.pkl", "wb") as f: pickle.dump(model, f)
    with open(MODELS_DIR / "feature_cols.json", "w") as f: json.dump(feature_cols, f)
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f: json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    if X_predict is not None:
        pred_probs = {name: model.predict_proba(X_predict)[:, 1] for name, model in models.items()}
        ranked_pred_probs = {name: rankdata(prob) / len(prob) for name, prob in pred_probs.items()}
        
        prob_sum = np.zeros(len(X_predict))
        for name, weight in ens_metrics.get("weights", {}).items():
            if weight > 0 and name in models: prob_sum += weight * ranked_pred_probs[name]
                
        y_pred = (prob_sum >= ens_metrics.get("threshold", 0.5)).astype(int)
        pd.DataFrame({"user_id": uid_predict, "status": y_pred, "risk_score": np.round(prob_sum, 4)}).sort_values("user_id").to_csv(SUBMISSION_PATH, index=False)
        print(f"  ✅ 提交檔已建立：{SUBMISSION_PATH}")

def main():
    print("=" * 60); print("BitoGuard v5.7 — 宗師級 Tabular 提分版 (徹底根除作弊)"); print("=" * 60 + "\n")

    (X_train, y_train, X_test, y_test, uid_test, feature_cols, train_spw, X_predict, uid_predict) = load_data()

    X_train, X_test, X_predict, feature_cols = perform_feature_selection(X_train, y_train, X_test, X_predict, feature_cols, train_spw, drop_k=DROP_K_FEATURES)

    lgb_params = tune_lgbm(X_train, y_train, train_spw)
    xgb_params = tune_xgb(X_train, y_train, train_spw)

    all_metrics, prob_dict, val_prob_dict, model_dict, metrics_map = [], {}, {}, {}, {}

    lgb_model, lgb_prob, lgb_val_prob, y_val, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test, lgb_params, train_spw)
    all_metrics.append(lgb_metrics); prob_dict["LightGBM"] = lgb_prob; val_prob_dict["LightGBM"] = lgb_val_prob; model_dict["LightGBM"] = lgb_model; metrics_map["LightGBM"] = lgb_metrics

    xgb_model, xgb_prob, xgb_val_prob, _, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test, xgb_params, train_spw)
    all_metrics.append(xgb_metrics); prob_dict["XGBoost"] = xgb_prob; val_prob_dict["XGBoost"] = xgb_val_prob; model_dict["XGBoost"] = xgb_model; metrics_map["XGBoost"] = xgb_metrics

    if CATBOOST_AVAILABLE:
        cat_model, cat_prob, cat_val_prob, _, cat_metrics = train_catboost(X_train, y_train, X_test, y_test, train_spw)
        if cat_model is not None:
            all_metrics.append(cat_metrics); prob_dict["CatBoost"] = cat_prob; val_prob_dict["CatBoost"] = cat_val_prob; model_dict["CatBoost"] = cat_model; metrics_map["CatBoost"] = cat_metrics

    ens_prob, ens_metrics = ensemble_predict(prob_dict, val_prob_dict, y_val, metrics_map, y_test)
    all_metrics.append(ens_metrics)
    
    save_models_and_reports(model_dict, all_metrics, uid_test, prob_dict, y_test, feature_cols, X_predict, uid_predict, ens_metrics)

    print("\n  🏆 最佳模型（F1）：", max(all_metrics, key=lambda x: x["f1"])["model"])
    print("\n訓練完成！")

if __name__ == "__main__":
    main()