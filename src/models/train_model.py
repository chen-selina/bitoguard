import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder # 引入類別編碼
import lightgbm as lgb
import xgboost as xgb
import optuna

warnings.filterwarnings("ignore")

# ============================================================
# 設定區
# ============================================================
OPTUNA_TRIALS = 50  
N_SPLITS      = 5   
RANDOM_STATE  = 42

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/saved")
REPORTS_DIR   = Path("outputs/reports")

for d in [MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_PATH      = PROCESSED_DIR / "train_balanced.csv"
TEST_PATH       = PROCESSED_DIR / "test.csv"
FEAT_COLS_PATH  = PROCESSED_DIR / "feature_cols.json"

# ============================================================
# 1. 增強版資料處理 (加入類別編碼)
# ============================================================
def load_and_preprocess():
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test  = pd.read_csv(TEST_PATH,  low_memory=False)

    # 自動識別類別特徵並編碼 (參考第二段代碼優點)
    cat_features = train.select_dtypes(include=['object']).columns.tolist()
    for col in cat_features:
        if col not in ["user_id"]:
            le = LabelEncoder()
            # 結合訓練與測試集確保編碼一致
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
    X_test  = test[feature_cols].values
    y_test  = test["label"].values
    
    # 計算正負樣本比 (scale_pos_weight 關鍵)
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    train_spw = neg_count / pos_count if pos_count > 0 else 1.0

    return X_train, y_train, X_test, y_test, feature_cols, train_spw

# ============================================================
# 2. 核心：細緻化閾值搜尋 (參考第一段代碼優點)
# ============================================================
def find_best_threshold_refined(y_true, y_prob, name="Model"):
    """使用更細的步進尋找最佳 F1 閾值"""
    thresholds = np.arange(0.1, 0.9, 0.01) # 0.01 步進比預設更精準
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f" 🔍 [{name}] 最佳 F1: {best_f1:.4f} | 最佳閾值: {best_thresh:.2f}")
    return best_thresh, best_f1

# ============================================================
# 3. Optuna 調參：直接鎖定 F1 最大化
# ============================================================
def tune_hyperparams_v2(X_train, y_train, train_spw, model_type="lgbm"):
    def obj(trial):
        if model_type == "lgbm":
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", 500, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 128),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, train_spw),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            }
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 500, 1000),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, train_spw),
            }
        
        # 使用 StratifiedKFold 確保 F1 的穩定性
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            m = lgb.LGBMClassifier(**params) if model_type=="lgbm" else xgb.XGBClassifier(**params)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            prob = m.predict_proba(X_train[val_idx])[:, 1]
            # 關鍵：評估時使用「該組資料的最佳 F1」而非 0.5
            _, current_f1 = find_best_threshold_refined(y_train[val_idx], prob)
            scores.append(current_f1)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=OPTUNA_TRIALS)
    return study.best_params

# ============================================================
# 4. 主流程 (Stacking + Refined Threshold)
# ============================================================
def main():
    print("=== BitoGuard v8.0 (F1 Optimization) ===")
    X_train, y_train, X_test, y_test, feat_cols, train_spw = load_and_preprocess()

    # 1. 調參 (鎖定最佳 F1)
    lgbm_params = tune_hyperparams_v2(X_train, y_train, train_spw, "lgbm")
    xgb_params  = tune_hyperparams_v2(X_train, y_train, train_spw, "xgb")

    # 2. 獲取 OOF (使用 5-Fold 增加穩定性)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    lgbm_oof = np.zeros(len(y_train))
    xgb_oof = np.zeros(len(y_train))
    
    lgbm_test_probs = []
    xgb_test_probs = []

    print("\n--- 5-Fold Training ---")
    for tr_idx, val_idx in skf.split(X_train, y_train):
        # LightGBM
        m_lgb = lgb.LGBMClassifier(**lgbm_params)
        m_lgb.fit(X_train[tr_idx], y_train[tr_idx])
        lgbm_oof[val_idx] = m_lgb.predict_proba(X_train[val_idx])[:, 1]
        lgbm_test_probs.append(m_lgb.predict_proba(X_test)[:, 1])
        
        # XGBoost
        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X_train[tr_idx], y_train[tr_idx])
        xgb_oof[val_idx] = m_xgb.predict_proba(X_train[val_idx])[:, 1]
        xgb_test_probs.append(m_xgb.predict_proba(X_test)[:, 1])

    # 3. Meta-Ensemble (Stacking)
    X_stack_train = np.column_stack([lgbm_oof, xgb_oof])
    X_stack_test  = np.column_stack([np.mean(lgbm_test_probs, axis=0), np.mean(xgb_test_probs, axis=0)])
    
    meta_model = LogisticRegression()
    meta_model.fit(X_stack_train, y_train)
    final_prob = meta_model.predict_proba(X_stack_test)[:, 1]
    oof_final_prob = meta_model.predict_proba(X_stack_train)[:, 1]

    # 4. 關鍵步驟：尋找 Meta 模型的最佳閾值
    best_thresh, _ = find_best_threshold_refined(y_train, oof_final_prob, "Final Stacking")

    # 5. 最終評估
    y_pred = (final_prob >= best_thresh).astype(int)
    print("\n--- Final Test Report ---")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    
    # 儲存模型
    with open(MODELS_DIR / "f1_optimized_model.pkl", "wb") as f:
        pickle.dump({"meta": meta_model, "thresh": best_thresh, "params": lgbm_params}, f)

if __name__ == "__main__":
    main()