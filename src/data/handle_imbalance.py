"""
handle_imbalance.py (v4 穩健版 - 移除 SMOTE)
核心變更：
  1. 移除預處理階段的 SMOTE，避免資料洩漏 (Data Leakage)。
  2. 僅保留「切分資料」與「特徵縮放 (RobustScaler)」。
  3. 產出 class_weights.csv 供訓練模型時使用 scale_pos_weight 參數。
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# 設定路徑
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH          = PROCESSED_DIR / "features.csv"
PREDICT_INPUT_PATH  = PROCESSED_DIR / "predict_features.csv"
TRAIN_OUTPUT_PATH   = PROCESSED_DIR / "train_balanced.csv" # 保持檔名一致以相容 Pipeline
TEST_OUTPUT_PATH    = PROCESSED_DIR / "test.csv"
PREDICT_OUTPUT_PATH = PROCESSED_DIR / "predict_scaled.csv"
SCALER_INFO_PATH    = PROCESSED_DIR / "scaler_info.csv"
WEIGHTS_PATH        = PROCESSED_DIR / "class_weights.csv"
FEAT_COLS_PATH      = PROCESSED_DIR / "feature_cols.json"

TEST_SIZE    = 0.2
RANDOM_STATE = 42

def analyze_imbalance(y: np.ndarray, stage: str = "") -> None:
    counter = Counter(y)
    total   = len(y)
    normal  = counter.get(0, 0)
    black   = counter.get(1, 0)
    ratio   = black / total * 100 if total > 0 else 0
    tag     = f"[{stage}] " if stage else ""
    print(f"  {tag}正常：{normal:,}  黑名單：{black:,}  比例：{ratio:.4f}%")

def main():
    print("=" * 60)
    print("BitoGuard v4 — 穩健資料處理（移除 SMOTE，改採權重法）")
    print("=" * 60 + "\n")

    # 1. 載入資料
    if not INPUT_PATH.exists():
        print(f"❌ 找不到特徵檔：{INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"【1】載入原始特徵：{len(df):,} 筆")
    
    feature_cols = [c for c in df.columns if c not in ["user_id", "label"]]
    X = df[feature_cols].values
    y = df["label"].values
    analyze_imbalance(y, "原始全量")

    # 2. 切分訓練/測試集 (Stratified Split 確保黑名單比例一致)
    user_ids = df["user_id"].values
    X_train, X_test, y_train, y_test, uid_train, uid_test = train_test_split(
        X, y, user_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"\n【2】資料切分完成：")
    analyze_imbalance(y_train, "訓練集")
    analyze_imbalance(y_test,  "測試集")

    # 3. 特徵縮放 (僅用訓練集 Fit)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # 存下 Scaler 參數備查
    scaler_info = pd.DataFrame({"feature": feature_cols, "center": scaler.center_, "scale": scaler.scale_})
    scaler_info.to_csv(SCALER_INFO_PATH, index=False)
    print(f"\n【3】特徵縮放完成，參數已存：{SCALER_INFO_PATH}")

    # 4. 處理預測集 (如果有)
    X_predict_scaled = None
    uid_predict = None
    if PREDICT_INPUT_PATH.exists():
        pred_df = pd.read_csv(PREDICT_INPUT_PATH, low_memory=False)
        uid_predict = pred_df["user_id"].values
        # 補齊預測集缺少的欄位
        for c in feature_cols:
            if c not in pred_df.columns:
                pred_df[c] = 0
        X_predict_scaled = scaler.transform(pred_df[feature_cols].values)
        print(f"【4】預測集處理完成：{len(pred_df):,} 筆")

    # 5. 計算類別權重 (Class Weights)
    # 此權重會傳給 XGBoost/LightGBM 的 scale_pos_weight
    counter = Counter(y_train)
    weight_val = round(counter[0] / counter[1], 2)
    weights = {0: 1.0, 1: weight_val}
    pd.DataFrame([{"class": k, "weight": v} for k, v in weights.items()]).to_csv(WEIGHTS_PATH, index=False)
    print(f"\n【5】建議權重已計算：{weights} (1:{weight_val})")

    # 6. 存檔
    # 訓練集
    train_out = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_out.insert(0, "user_id", uid_train)
    train_out["label"] = y_train
    train_out.to_csv(TRAIN_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # 測試集
    test_out = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_out.insert(0, "user_id", uid_test)
    test_out["label"] = y_test
    test_out.to_csv(TEST_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # 預測集
    if X_predict_scaled is not None:
        pred_out = pd.DataFrame(X_predict_scaled, columns=feature_cols)
        pred_out.insert(0, "user_id", uid_predict)
        pred_out.to_csv(PREDICT_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    with open(FEAT_COLS_PATH, "w") as f:
        json.dump(feature_cols, f)

    print(f"\n✅ 處理完成！")
    print(f"   - 訓練集：{TRAIN_OUTPUT_PATH}")
    print(f"   - 測試集：{TEST_OUTPUT_PATH}")
    print(f"\n下一步請執行：python src/models/train_model.py")

if __name__ == "__main__":
    main()