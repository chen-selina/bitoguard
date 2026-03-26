"""
handle_imbalance.py
位置：src/data/handle_imbalance.py

【v3 升級說明】
  1. SMOTE ratio 從 0.3 降至 0.15
     - 舊版把黑名單合成到 23%（比例過高，噪音太多）
     - 新版合成到 ~13%，品質更好、噪音更少
     - 業界詐騙偵測建議：合成到 10-15% 最佳

  2. 保留 v2 的所有修正：
     - 先切分再 Scaler（避免洩漏）
     - CV 用原始分布資料（非 SMOTE 後）
     - 預測集 transform

使用方式：python src/data/handle_imbalance.py
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH          = PROCESSED_DIR / "features.csv"
PREDICT_INPUT_PATH  = PROCESSED_DIR / "predict_features.csv"
TRAIN_OUTPUT_PATH   = PROCESSED_DIR / "train_balanced.csv"
TEST_OUTPUT_PATH    = PROCESSED_DIR / "test.csv"
PREDICT_OUTPUT_PATH = PROCESSED_DIR / "predict_scaled.csv"
SCALER_INFO_PATH    = PROCESSED_DIR / "scaler_info.csv"
WEIGHTS_PATH        = PROCESSED_DIR / "class_weights.csv"

TEST_SIZE    = 0.2
RANDOM_STATE = 42

# 【v3 修正】SMOTE ratio 從 0.3 降至 0.15
# 合成到 ~13% 而非 23%，減少噪音、提升模型泛化能力
SMOTE_RATIO = 0.15


# ============================================================
# 1. 載入訓練特徵
# ============================================================

def load_features():
    print(f"  讀取：{INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"  總資料：{len(df):,} 筆，{len(df.columns)} 欄")

    if "label" not in df.columns:
        raise ValueError("找不到 label 欄位，請先執行 feature_engineering.py")

    feature_cols = [c for c in df.columns if c not in ["user_id", "label"]]
    X = df[feature_cols].values
    y = df["label"].values
    return df, X, y, feature_cols


def analyze_imbalance(y: np.ndarray, stage: str = "") -> None:
    counter = Counter(y)
    total   = len(y)
    normal  = counter.get(0, 0)
    black   = counter.get(1, 0)
    ratio   = black / total * 100 if total > 0 else 0
    tag     = f"[{stage}] " if stage else ""
    print(f"  {tag}正常：{normal:,}  黑名單：{black:,}  比例：{ratio:.4f}%")
    if ratio < 1:
        print("  ⚠️  極度不平衡（< 1%）")
    elif ratio < 5:
        print("  ⚠️  嚴重不平衡（< 5%），建議使用 SMOTE")


# ============================================================
# 2. 切分訓練集 / 測試集
# ============================================================

def split_data(df, X, y, feature_cols):
    user_ids = df["user_id"].values
    (X_train, X_test,
     y_train, y_test,
     uid_train, uid_test) = train_test_split(
        X, y, user_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  訓練集：{len(X_train):,} 筆")
    print(f"  測試集：{len(X_test):,} 筆（保留給最終評估，不參與任何訓練）")
    analyze_imbalance(y_train, "訓練集")
    analyze_imbalance(y_test,  "測試集")
    return X_train, X_test, y_train, y_test, uid_train, uid_test


# ============================================================
# 3. 特徵縮放
# ============================================================

def scale_features(X_train, X_test, feature_cols, X_predict=None):
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print(f"  Scaler 已用訓練集 fit（{len(X_train):,} 筆）")

    X_test_scaled = scaler.transform(X_test)
    print(f"  測試集 transform 完成（{len(X_test):,} 筆）")

    X_predict_scaled = None
    if X_predict is not None:
        X_predict_scaled = scaler.transform(X_predict)
        print(f"  預測集 transform 完成（{len(X_predict):,} 筆）")

    scaler_info = pd.DataFrame({
        "feature": feature_cols,
        "center":  scaler.center_,
        "scale":   scaler.scale_,
    })
    scaler_info.to_csv(SCALER_INFO_PATH, index=False)
    print(f"  Scaler 參數已存：{SCALER_INFO_PATH}")

    return X_train_scaled, X_test_scaled, X_predict_scaled, scaler


# ============================================================
# 4. SMOTE（v3：ratio 0.3 → 0.15）
# ============================================================

def choose_and_apply_smote(X_train, y_train, feature_cols):
    black_count = int((y_train == 1).sum())
    print(f"  訓練集黑名單數：{black_count}")
    print(f"  目標合成比率：{SMOTE_RATIO:.0%}（舊版 0.3，新版降低避免過多噪音）")

    if black_count < 6:
        print("  ⚠️  黑名單樣本數過少（< 6），跳過 SMOTE")
        return X_train, y_train

    binary_flags = [
        "career_high_risk", "income_no_source", "sex_undisclosed",
        "twd_withdraw_gt_deposit", "age_missing",
    ]
    cat_indices = [i for i, col in enumerate(feature_cols) if col in binary_flags]
    k = min(5, black_count - 1)

    try:
        if len(cat_indices) > 0:
            print(f"  策略：SMOTENC（含 {len(cat_indices)} 個類別型特徵）")
            sampler = SMOTENC(
                categorical_features=cat_indices,
                sampling_strategy=SMOTE_RATIO,
                random_state=RANDOM_STATE,
                k_neighbors=k,
            )
        else:
            print("  策略：SMOTE（純數值特徵）")
            sampler = SMOTE(
                sampling_strategy=SMOTE_RATIO,
                random_state=RANDOM_STATE,
                k_neighbors=k,
            )

        X_res, y_res = sampler.fit_resample(X_train, y_train)
        print(f"  SMOTE 前：{Counter(y_train)}")
        print(f"  SMOTE 後：{Counter(y_res)}")
        analyze_imbalance(y_res, "SMOTE 後")
        return X_res, y_res

    except Exception as e:
        print(f"  ⚠️  SMOTE 失敗：{e}，退回原始訓練集")
        return X_train, y_train


# ============================================================
# 5. 計算 class_weight
# ============================================================

def compute_class_weight(y_original: np.ndarray) -> dict:
    """
    【v3 重要】class_weight 用 SMOTE 前的原始訓練集計算
    讓模型知道原始的不平衡程度（1:30），而不是 SMOTE 後的假比例
    """
    counter      = Counter(y_original)
    normal_count = counter.get(0, 1)
    black_count  = counter.get(1, 1)
    weight_black = normal_count / black_count
    weights      = {0: 1.0, 1: round(weight_black, 2)}
    print(f"  建議 class_weight（基於原始分布）：{weights}")
    print(f"  （原始黑名單佔 {black_count / (normal_count + black_count) * 100:.2f}%）")
    return weights


# ============================================================
# 6. 存檔
# ============================================================

def save_datasets(X_train_res, y_train_res,
                  X_test_scaled, y_test,
                  uid_train, uid_test,
                  X_predict_scaled, uid_predict,
                  feature_cols) -> None:
    n_orig  = len(uid_train)
    n_synth = len(X_train_res) - n_orig
    uid_col = list(uid_train) + [-1] * n_synth

    train_df = pd.DataFrame(X_train_res, columns=feature_cols)
    train_df.insert(0, "user_id", uid_col)
    train_df["label"] = y_train_res
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"  訓練集：{TRAIN_OUTPUT_PATH}（{len(train_df):,} 筆，含 {n_synth} 筆合成）")

    test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_df.insert(0, "user_id", uid_test)
    test_df["label"] = y_test
    test_df.to_csv(TEST_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"  測試集：{TEST_OUTPUT_PATH}（{len(test_df):,} 筆）")

    if X_predict_scaled is not None and uid_predict is not None:
        pred_df = pd.DataFrame(X_predict_scaled, columns=feature_cols)
        pred_df.insert(0, "user_id", uid_predict)
        pred_df.to_csv(PREDICT_OUTPUT_PATH, index=False, encoding="utf-8-sig")
        print(f"  預測集：{PREDICT_OUTPUT_PATH}（{len(pred_df):,} 筆）")


# ============================================================
# 7. 主程式
# ============================================================

def main():
    print("=" * 55)
    print("BitoGuard v3 — 不平衡資料處理（SMOTE ratio 0.15）")
    print("=" * 55 + "\n")

    print("【載入訓練特徵】")
    df, X, y, feature_cols = load_features()
    analyze_imbalance(y, "原始資料")
    print()

    X_predict, uid_predict = None, None
    if PREDICT_INPUT_PATH.exists():
        print("【載入預測特徵】")
        pred_df    = pd.read_csv(PREDICT_INPUT_PATH, low_memory=False)
        uid_predict = pred_df["user_id"].values
        # 只取 feature_cols 中存在的欄位
        valid_cols = [c for c in feature_cols if c in pred_df.columns]
        missing    = [c for c in feature_cols if c not in pred_df.columns]
        if missing:
            print(f"  ⚠️  預測集缺少欄位（補 0）：{missing}")
            for c in missing:
                pred_df[c] = 0
        X_predict = pred_df[feature_cols].values
        print(f"  預測集：{len(pred_df):,} 筆")
        print()

    print("【切分訓練集 / 測試集（先於 Scaler）】")
    (X_train, X_test,
     y_train, y_test,
     uid_train, uid_test) = split_data(df, X, y, feature_cols)
    print()

    # 保留原始 y_train 供 class_weight 計算
    y_train_original = y_train.copy()

    print("【特徵縮放（只用訓練集 fit）】")
    (X_train_scaled, X_test_scaled,
     X_predict_scaled, scaler) = scale_features(
        X_train, X_test, feature_cols, X_predict
    )
    print()

    print("【SMOTE（ratio=0.15，比舊版 0.3 更保守）】")
    X_train_res, y_train_res = choose_and_apply_smote(
        X_train_scaled, y_train, feature_cols
    )
    print()

    print("【計算 class_weight（基於原始訓練集分布）】")
    weights = compute_class_weight(y_train_original)
    weight_df = pd.DataFrame([{"class": k, "weight": v} for k, v in weights.items()])
    weight_df.to_csv(WEIGHTS_PATH, index=False)
    print(f"  已存：{WEIGHTS_PATH}")
    print()

    print("【存檔】")
    save_datasets(
        X_train_res, y_train_res,
        X_test_scaled, y_test,
        uid_train, uid_test,
        X_predict_scaled, uid_predict,
        feature_cols,
    )

    import json
    feat_path = PROCESSED_DIR / "feature_cols.json"
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"  特徵欄名：{feat_path}")

    print("\n不平衡處理完成！")
    print("下一步：執行 src/models/train_model.py")


if __name__ == "__main__":
    main()