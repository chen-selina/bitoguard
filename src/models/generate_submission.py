"""
generate_submission.py
位置：src/models/generate_submission.py

功能：載入訓練好的模型，對 predict_label.csv 中的用戶進行預測，
      生成最終提交檔案 submission.csv（只包含 user_id 和 status）

使用方式：python src/models/generate_submission.py
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# 路徑設定
# ============================================================
PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
MODELS_DIR = Path("models/saved")
REPORTS_DIR = Path("outputs/reports")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 輸入檔案
PREDICT_SCALED_PATH = PROCESSED_DIR / "predict_scaled.csv"
PREDICT_LABEL_PATH = RAW_DIR / "predict_label.csv"
FEAT_COLS_PATH = PROCESSED_DIR / "feature_cols.json"

# 模型檔案（優先使用 ensemble，若無則用單一模型）
ENSEMBLE_MODEL_PATH = MODELS_DIR / "final_ensemble.pkl"
F1_MODEL_PATH = MODELS_DIR / "f1_optimized_model.pkl"
LIGHTGBM_PATH = MODELS_DIR / "lightgbm.pkl"
XGBOOST_PATH = MODELS_DIR / "xgboost.pkl"

# 輸出檔案
SUBMISSION_PATH = REPORTS_DIR / "submission.csv"


# ============================================================
# 1. 載入模型與資料
# ============================================================

def load_model():
    """依優先順序載入模型"""
    model_paths = [
        (ENSEMBLE_MODEL_PATH, "Ensemble 模型"),
        (F1_MODEL_PATH, "F1 優化模型"),
        (LIGHTGBM_PATH, "LightGBM 模型"),
        (XGBOOST_PATH, "XGBoost 模型"),
    ]
    
    for path, name in model_paths:
        if path.exists():
            print(f"  載入模型：{name} ({path})")
            with open(path, "rb") as f:
                model_data = pickle.load(f)
            
            # 處理不同的模型格式
            if isinstance(model_data, dict):
                # Ensemble 模型格式：{"models": {"LightGBM": ..., "XGBoost": ..., "stacking": ...}, "columns": [...]}
                if "models" in model_data and isinstance(model_data["models"], dict):
                    models_dict = model_data["models"]
                    # 使用 stacking 模型（meta learner）
                    if "stacking" in models_dict:
                        model = models_dict["stacking"]
                        threshold = 0.5
                        print(f"  使用 Stacking Meta Learner")
                        return model, threshold, name, "stacking", models_dict
                    else:
                        # 如果沒有 stacking，使用所有基礎模型
                        models = list(models_dict.values())
                        threshold = 0.5
                        return models, threshold, name, "ensemble", models_dict
                # F1 優化模型格式：{"meta": model, "thresh": 0.5, ...}
                elif "meta" in model_data:
                    model = model_data["meta"]
                    threshold = model_data.get("thresh", 0.5)
                    return model, threshold, name, "single", None
                # 其他字典格式
                else:
                    model = model_data.get("model")
                    threshold = model_data.get("thresh", 0.5)
                    return model, threshold, name, "single", None
            else:
                # 直接是模型物件
                model = model_data
                threshold = 0.5
                return model, threshold, name, "single", None
    
    raise FileNotFoundError("找不到任何訓練好的模型檔案！")


def load_predict_data():
    """載入預測集資料"""
    # 載入特徵欄位名稱
    if not FEAT_COLS_PATH.exists():
        raise FileNotFoundError(f"找不到特徵欄位檔案：{FEAT_COLS_PATH}")
    
    with open(FEAT_COLS_PATH, "r") as f:
        feature_cols = json.load(f)
    
    # 載入預測集（已縮放）
    if not PREDICT_SCALED_PATH.exists():
        raise FileNotFoundError(f"找不到預測集檔案：{PREDICT_SCALED_PATH}")
    
    predict_df = pd.read_csv(PREDICT_SCALED_PATH, low_memory=False)
    
    # 載入原始 predict_label（取得完整的 user_id 清單）
    if PREDICT_LABEL_PATH.exists():
        predict_label = pd.read_csv(PREDICT_LABEL_PATH)
        print(f"  預測目標用戶數：{len(predict_label):,} 位")
    else:
        predict_label = None
        print(f"  預測集用戶數：{len(predict_df):,} 位")
    
    X_predict = predict_df[feature_cols].values
    uid_predict = predict_df["user_id"].values
    
    print(f"  特徵數：{len(feature_cols)}")
    
    return X_predict, uid_predict, feature_cols, predict_label


# ============================================================
# 2. 進行預測
# ============================================================

def make_predictions(model, X_predict, threshold: float, model_type: str, models_dict=None):
    """使用模型進行預測"""
    print(f"\n  使用閾值：{threshold:.4f}")
    
    # Stacking 模型：需要先用基礎模型生成特徵，再用 meta learner 預測
    if model_type == "stacking" and models_dict:
        print(f"  使用 Stacking 架構（基礎模型 → Meta Learner）...")
        
        # 取得基礎模型的預測機率
        base_probs = []
        for model_name, base_model in models_dict.items():
            if model_name == "stacking":
                continue  # 跳過 meta learner
            
            if hasattr(base_model, "predict_proba"):
                prob = base_model.predict_proba(X_predict)[:, 1]
            else:
                prob = base_model.predict(X_predict)
            base_probs.append(prob)
            print(f"    - {model_name} 預測完成")
        
        # 將基礎模型的預測作為 meta learner 的輸入
        X_meta = np.column_stack(base_probs)
        
        # 使用 meta learner（stacking model）進行最終預測
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_meta)[:, 1]
        else:
            y_prob = model.predict(X_meta)
    
    # Ensemble 模型：多個模型投票
    elif model_type == "ensemble":
        print(f"  使用 {len(model)} 個模型進行 Ensemble 預測...")
        all_probs = []
        for m in model:
            if hasattr(m, "predict_proba"):
                prob = m.predict_proba(X_predict)[:, 1]
            else:
                prob = m.predict(X_predict)
            all_probs.append(prob)
        
        # 平均所有模型的機率
        y_prob = np.mean(all_probs, axis=0)
    
    # 單一模型
    else:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_predict)[:, 1]
        else:
            y_prob = model.predict(X_predict)
    
    # 根據閾值轉換為 0/1
    y_pred = (y_prob >= threshold).astype(int)
    
    # 統計
    total = len(y_pred)
    blacklist_count = np.sum(y_pred == 1)
    normal_count = np.sum(y_pred == 0)
    
    print(f"  預測結果：")
    print(f"    正常用戶：{normal_count:,} 位 ({normal_count/total*100:.2f}%)")
    print(f"    黑名單  ：{blacklist_count:,} 位 ({blacklist_count/total*100:.2f}%)")
    
    return y_pred, y_prob


# ============================================================
# 3. 生成提交檔案
# ============================================================

def generate_submission(uid_predict, y_pred, predict_label=None):
    """
    生成最終提交檔案，格式：user_id, status
    只包含 predict_label.csv 中的用戶
    """
    # 建立預測結果 DataFrame
    submission = pd.DataFrame({
        "user_id": uid_predict,
        "status": y_pred
    })
    
    # 如果有 predict_label，確保只包含這些用戶
    if predict_label is not None:
        target_uids = set(predict_label["user_id"])
        submission = submission[submission["user_id"].isin(target_uids)]
        
        # 確認是否有遺漏的用戶
        missing = target_uids - set(submission["user_id"])
        if missing:
            print(f"  ⚠️  警告：{len(missing)} 位用戶沒有預測結果，將標記為正常（status=0）")
            missing_df = pd.DataFrame({
                "user_id": list(missing),
                "status": 0
            })
            submission = pd.concat([submission, missing_df], ignore_index=True)
    
    # 排序並存檔（只保留 user_id 和 status）
    submission = submission[["user_id", "status"]].sort_values("user_id")
    submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ 提交檔案已生成：{SUBMISSION_PATH}")
    print(f"   總用戶數：{len(submission):,}")
    print(f"   格式：user_id, status")
    
    # 顯示前幾筆
    print(f"\n  前 5 筆預測結果：")
    print(submission.head().to_string(index=False))
    
    return submission


# ============================================================
# 4. 主程式
# ============================================================

def main():
    print("=" * 60)
    print("BitoGuard — 生成最終提交檔案")
    print("=" * 60 + "\n")
    
    print("【載入模型】")
    model, threshold, model_name, model_type, models_dict = load_model()
    print()
    
    print("【載入預測集】")
    X_predict, uid_predict, feature_cols, predict_label = load_predict_data()
    print()
    
    print("【進行預測】")
    y_pred, y_prob = make_predictions(model, X_predict, threshold, model_type, models_dict)
    print()
    
    print("【生成提交檔案】")
    submission = generate_submission(uid_predict, y_pred, predict_label)
    
    print("\n" + "=" * 60)
    print("✅ 完成！可以提交 submission.csv 了")
    print("=" * 60)


if __name__ == "__main__":
    main()
