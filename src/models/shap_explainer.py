"""
shap_explainer.py
位置：src/models/shap_explainer.py

功能：載入訓練好的 XGBoost 模型，用 SHAP 解釋每位用戶的風險因子，
      自動生成「風險診斷書」（自然語言 + 數值），
      輸出至 outputs/reports/ 與 outputs/plots/

使用方式：python src/models/shap_explainer.py
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # 不開視窗，直接存圖

# ============================================================
# 路徑設定
# ============================================================
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/saved")
REPORTS_DIR   = Path("outputs/reports")
PLOTS_DIR     = Path("outputs/plots")

for d in [REPORTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TEST_PATH        = PROCESSED_DIR / "test.csv"
MODEL_PATH       = MODELS_DIR / "lightgbm.pkl"
FEAT_COLS_PATH   = MODELS_DIR / "feature_cols.json"
RISK_SCORES_PATH = REPORTS_DIR / "user_risk_scores.csv"

# 診斷書輸出
DIAGNOSIS_CSV  = REPORTS_DIR / "risk_diagnosis.csv"
DIAGNOSIS_JSON = REPORTS_DIR / "risk_diagnosis.json"

# 圖表輸出
SUMMARY_PLOT     = PLOTS_DIR / "shap_summary.png"
IMPORTANCE_PLOT  = PLOTS_DIR / "shap_importance.png"

# 分析前幾名高風險用戶
TOP_N_USERS = 20

# 風險閾值（高於此分數才產生診斷書）
RISK_THRESHOLD = 0.5


# ============================================================
# 特徵中文說明對照表
# ============================================================
FEATURE_LABELS = {
    # 法幣行為
    "twd_deposit_count":       "台幣入金次數",
    "twd_deposit_total":       "台幣入金總額",
    "twd_deposit_mean":        "台幣入金平均金額",
    "twd_deposit_max":         "台幣入金單筆最高金額",
    "twd_deposit_std":         "台幣入金金額標準差",
    "twd_withdraw_count":      "台幣出金次數",
    "twd_withdraw_total":      "台幣出金總額",
    "twd_withdraw_mean":       "台幣出金平均金額",
    "twd_withdraw_max":        "台幣出金單筆最高金額",
    "twd_withdraw_std":        "台幣出金金額標準差",
    "twd_inout_ratio":         "台幣快進快出比率",
    "twd_withdraw_gt_deposit": "出金次數超過入金次數",
    "twd_stay_mean":           "台幣平均滯留時間（秒）",
    "twd_stay_min":            "台幣最短滯留時間（秒）",
    "twd_night_ratio":         "台幣深夜交易比率",
    # IP 行為
    "ip_unique_count":         "使用不同 IP 數量",
    "ip_total_count":          "IP 操作總次數",
    "ip_diversity_ratio":      "IP 多樣性比率",
    "ip_subnet_count":         "使用不同子網段數量",
    # KYC 屬性
    "kyc_level":               "KYC 驗證等級",
    "kyc_l1_to_l2_days":       "KYC L1 到 L2 完成天數",
    "career_high_risk":        "高風險職業旗標",
    "career_code":             "職業代碼",
    "income_no_source":        "無收入來源旗標",
    "income_code":             "收入來源代碼",
    "sex_undisclosed":         "性別未填旗標",
    "user_source":             "帳號註冊管道",
    # 虛擬幣行為
    "crypto_deposit_count":    "虛擬幣入金次數",
    "crypto_deposit_total":    "虛擬幣入金總量",
    "crypto_withdraw_count":   "虛擬幣出金次數",
    "crypto_withdraw_total":   "虛擬幣出金總量",
    "crypto_inout_ratio":      "虛擬幣快進快出比率",
    "crypto_internal_count":   "交易所內轉次數",
    "crypto_internal_ratio":   "內轉比率",
    "trc20_count":             "TRC20 鏈使用次數",
    "trc20_usage_ratio":       "TRC20 使用比率",
    "crypto_wallet_diversity": "使用不同錢包地址數",
    "crypto_currency_diversity":"交易幣種多樣性",
    "crypto_night_ratio":      "虛擬幣深夜交易比率",
    # 交易行為
    "trade_count":             "掛單交易次數",
    "trade_total_usdt":        "掛單 USDT 總量",
    "trade_mean_usdt":         "掛單平均 USDT 數量",
    "trade_buy_ratio":         "掛單買單比率",
    "trade_market_ratio":      "市價單比率",
    "swap_count":              "一鍵買賣次數",
    "swap_total_twd":          "一鍵買賣台幣總額",
    "swap_mean_twd":           "一鍵買賣平均台幣金額",
    "swap_sell_ratio":         "一鍵賣幣比率",
}

# ============================================================
# 風險因子的自然語言模板
# 格式：(特徵名, 方向, 門檻) → 說明文字
# 方向 "high" = 數值高時有風險，"low" = 數值低時有風險
# ============================================================
RISK_TEMPLATES = [
    ("twd_inout_ratio",         "high", 0.8,
     "台幣出入金比率高達 {val:.0%}，資金幾乎全數快進快出"),
    ("twd_stay_min",            "low",  3600,
     "最短滯留時間僅 {val:.0f} 秒，疑似存入後立即提領"),
    ("twd_night_ratio",         "high", 0.3,
     "深夜（23:00–05:00）交易比率達 {val:.0%}，異常時段活躍"),
    ("ip_unique_count",         "high", 5,
     "使用了 {val:.0f} 個不同 IP 位址進行操作"),
    ("ip_diversity_ratio",      "high", 0.5,
     "IP 多樣性比率達 {val:.0%}，可能使用 VPN 或多設備操作"),
    ("trc20_usage_ratio",       "high", 0.3,
     "TRC20 鏈使用比率達 {val:.0%}（常見詐騙鏈路）"),
    ("crypto_internal_ratio",   "high", 0.5,
     "交易所內轉比率達 {val:.0%}，疑似資金拆分清洗"),
    ("crypto_inout_ratio",      "high", 0.8,
     "虛擬幣快進快出比率達 {val:.0%}"),
    ("career_high_risk",        "high", 0.5,
     "職業屬於高風險類別（學生／無業／自由業），但交易量異常龐大"),
    ("income_no_source",        "high", 0.5,
     "收入來源登記為「無」，卻持續進行大額交易"),
    ("kyc_level",               "low",  2,
     "KYC 驗證未完成（等級 {val:.0f}），僅完成基本驗證"),
    ("trade_market_ratio",      "high", 0.7,
     "市價單比率達 {val:.0%}，交易追蹤難度較高"),
    ("crypto_wallet_diversity", "high", 5,
     "使用了 {val:.0f} 個不同虛擬錢包地址"),
]


# ============================================================
# 1. 載入資料與模型
# ============================================================

def load_assets():
    print(f"  模型：{MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"  特徵欄名：{FEAT_COLS_PATH}")
    with open(FEAT_COLS_PATH, "r") as f:
        feature_cols = json.load(f)

    print(f"  測試集：{TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    # 讀取風險分數（train_model.py 產出）
    risk_df = None
    if RISK_SCORES_PATH.exists():
        risk_df = pd.read_csv(RISK_SCORES_PATH)

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values if "label" in test_df.columns else None
    uid_test = test_df["user_id"].values

    print(f"  測試集：{len(test_df):,} 筆，特徵數：{len(feature_cols)}")
    return model, feature_cols, test_df, X_test, y_test, uid_test, risk_df


# ============================================================
# 2. 計算 SHAP 值
# ============================================================

def compute_shap_values(model, X_test: np.ndarray, feature_cols: list):
    """
    TreeExplainer 專為樹狀模型（XGBoost/LightGBM）設計，速度最快。
    回傳每位用戶、每個特徵的 SHAP 值矩陣（shape: n_users × n_features）
    """
    print("  建立 TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    print("  計算 SHAP 值（可能需要 1~2 分鐘）...")
    shap_values = explainer.shap_values(X_test)

    # XGBoost 二元分類：shap_values 可能是 list，取正類（黑名單）那組
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print(f"  SHAP 矩陣：{shap_values.shape}（用戶數 × 特徵數）")
    return explainer, shap_values


# ============================================================
# 3. 全局特徵重要性圖（Summary Plot）
# ============================================================

def plot_global_importance(shap_values: np.ndarray,
                            X_test: np.ndarray,
                            feature_cols: list) -> None:
    """
    兩張圖：
    1. Beeswarm（蜂群圖）— 顯示每個特徵在所有用戶的 SHAP 分布
       → 橫軸 = SHAP 值大小，顏色 = 原始特徵值（紅=高，藍=低）
    2. Bar Chart — 全局平均 |SHAP| 排名
    """
    # 把特徵名換成中文
    display_names = [FEATURE_LABELS.get(c, c) for c in feature_cols]

    # --- 圖1：Beeswarm ---
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=display_names,
        show=False,
        max_display=20,
    )
    plt.title("SHAP Beeswarm — 各特徵對黑名單判定的影響", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Beeswarm 圖：{SUMMARY_PLOT}")

    # --- 圖2：Bar Chart ---
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=display_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title("SHAP Feature Importance — 全局平均貢獻度", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  重要性圖：{IMPORTANCE_PLOT}")


# ============================================================
# 4. 單用戶 Waterfall 圖
# ============================================================

def plot_user_waterfall(explainer, shap_values: np.ndarray,
                         X_test: np.ndarray, feature_cols: list,
                         user_idx: int, user_id, risk_score: float) -> str:
    """
    Waterfall 圖：顯示這位用戶每個特徵如何把風險分數
    從「基準值」一步一步推高或拉低。
    直覺易懂，適合給合規人員看。
    """
    display_names = [FEATURE_LABELS.get(c, c) for c in feature_cols]

    explanation = shap.Explanation(
        values=shap_values[user_idx],
        base_values=explainer.expected_value
                    if not isinstance(explainer.expected_value, list)
                    else explainer.expected_value[1],
        data=X_test[user_idx],
        feature_names=display_names,
    )

    plt.figure(figsize=(12, 7))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(
        f"用戶 {user_id} 風險診斷  |  風險分數：{risk_score:.3f}",
        fontsize=12, pad=10,
    )
    plt.tight_layout()

    plot_path = PLOTS_DIR / f"waterfall_user_{user_id}.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    return str(plot_path)


# ============================================================
# 5. 生成自然語言風險診斷書
# ============================================================

def generate_diagnosis(user_id,
                        risk_score: float,
                        shap_values_row: np.ndarray,
                        feature_vals_row: np.ndarray,
                        feature_cols: list,
                        true_label: int = None) -> dict:
    """
    對單一用戶生成風險診斷書，包含：
    - 風險評級（低/中/高/極高）
    - Top 5 風險因子（特徵名 + SHAP 貢獻 + 數值）
    - 自然語言說明
    """
    # 風險評級
    if risk_score >= 0.85:
        risk_level = "極高"
        risk_emoji = "🔴"
    elif risk_score >= 0.65:
        risk_level = "高"
        risk_emoji = "🟠"
    elif risk_score >= 0.40:
        risk_level = "中"
        risk_emoji = "🟡"
    else:
        risk_level = "低"
        risk_emoji = "🟢"

    # 取前 5 個 |SHAP| 最大的特徵（對判斷貢獻最大）
    top_idx = np.argsort(np.abs(shap_values_row))[::-1][:5]

    top_factors = []
    for idx in top_idx:
        fname  = feature_cols[idx]
        fval   = feature_vals_row[idx]
        sv     = shap_values_row[idx]
        label  = FEATURE_LABELS.get(fname, fname)
        direction = "↑ 拉高風險" if sv > 0 else "↓ 降低風險"

        top_factors.append({
            "rank":       len(top_factors) + 1,
            "feature":    fname,
            "label":      label,
            "value":      round(float(fval), 4),
            "shap":       round(float(sv), 4),
            "direction":  direction,
        })

    # 自然語言說明（觸發條件型）
    nl_reasons = []
    feat_val_map = dict(zip(feature_cols, feature_vals_row))

    for (fname, direction, threshold, template) in RISK_TEMPLATES:
        val = feat_val_map.get(fname, 0)
        triggered = (direction == "high" and val > threshold) or \
                    (direction == "low"  and val < threshold)
        if triggered:
            try:
                nl_reasons.append(template.format(val=val))
            except Exception:
                nl_reasons.append(template)

    # 若沒有觸發任何規則，用 SHAP Top1 特徵補充說明
    if not nl_reasons and top_factors:
        t = top_factors[0]
        nl_reasons.append(
            f"主要風險因子為「{t['label']}」（值：{t['value']}，SHAP：{t['shap']:+.3f}）"
        )

    diagnosis = {
        "user_id":      str(user_id),
        "risk_score":   round(float(risk_score), 4),
        "risk_level":   risk_level,
        "risk_emoji":   risk_emoji,
        "true_label":   int(true_label) if true_label is not None else None,
        "top_factors":  top_factors,
        "nl_summary":   "；".join(nl_reasons) if nl_reasons else "無明顯異常訊號",
        "waterfall_plot": None,  # 之後由 plot_user_waterfall 填入
    }

    return diagnosis


# ============================================================
# 6. 主程式
# ============================================================

def main():
    print("=" * 55)
    print("BitoGuard - SHAP 解釋性分析開始")
    print("=" * 55 + "\n")

    # 載入
    print("【載入資產】")
    (model, feature_cols, test_df,
     X_test, y_test, uid_test, risk_df) = load_assets()
    print()

    # 計算 SHAP 值
    print("【計算 SHAP 值】")
    explainer, shap_values = compute_shap_values(model, X_test, feature_cols)
    print()

    # 全局圖表
    print("【生成全局特徵重要性圖】")
    plot_global_importance(shap_values, X_test, feature_cols)
    print()

    # 決定要分析哪些用戶
    # 優先用 risk_scores.csv 的排名，若沒有則用模型預測機率
    if risk_df is not None and "risk_score" in risk_df.columns:
        high_risk_uids = (
            risk_df[risk_df["risk_score"] >= RISK_THRESHOLD]
            .sort_values("risk_score", ascending=False)
            .head(TOP_N_USERS)["user_id"]
            .values
        )
        uid_to_score = dict(zip(risk_df["user_id"], risk_df["risk_score"]))
    else:
        # fallback：用模型直接預測
        y_prob = model.predict_proba(X_test)[:, 1]
        top_idx_global = np.argsort(y_prob)[::-1][:TOP_N_USERS]
        high_risk_uids = uid_test[top_idx_global]
        uid_to_score   = dict(zip(uid_test, y_prob))

    print(f"【生成前 {len(high_risk_uids)} 名高風險用戶診斷書】")

    # 建立 uid → 測試集 index 的對應
    uid_to_idx = {uid: i for i, uid in enumerate(uid_test)}

    all_diagnoses = []

    for rank, uid in enumerate(high_risk_uids, 1):
        if uid not in uid_to_idx:
            continue

        idx        = uid_to_idx[uid]
        risk_score = uid_to_score.get(uid, 0.0)
        true_label = int(y_test[idx]) if y_test is not None else None

        # 生成診斷書
        diagnosis = generate_diagnosis(
            user_id=uid,
            risk_score=risk_score,
            shap_values_row=shap_values[idx],
            feature_vals_row=X_test[idx],
            feature_cols=feature_cols,
            true_label=true_label,
        )

        # 生成 Waterfall 圖
        plot_path = plot_user_waterfall(
            explainer, shap_values, X_test, feature_cols,
            idx, uid, risk_score,
        )
        diagnosis["waterfall_plot"] = plot_path

        all_diagnoses.append(diagnosis)

        # 印出簡要摘要
        label_tag = ""
        if true_label is not None:
            label_tag = "【真實黑名單✓】" if true_label == 1 else "【正常用戶】"
        print(f"  #{rank:02d}  用戶 {uid}  {diagnosis['risk_emoji']} {diagnosis['risk_level']}風險  "
              f"分數：{risk_score:.3f}  {label_tag}")
        print(f"       {diagnosis['nl_summary'][:60]}{'...' if len(diagnosis['nl_summary']) > 60 else ''}")

    # 存成 CSV（供 dashboard 讀取）
    flat_rows = []
    for d in all_diagnoses:
        top_factor_labels = [f["label"] for f in d["top_factors"]]
        top_factor_shaps  = [f["shap"]  for f in d["top_factors"]]
        flat_rows.append({
            "user_id":         d["user_id"],
            "risk_score":      d["risk_score"],
            "risk_level":      d["risk_level"],
            "true_label":      d["true_label"],
            "nl_summary":      d["nl_summary"],
            "top_factor_1":    top_factor_labels[0] if len(top_factor_labels) > 0 else "",
            "top_factor_2":    top_factor_labels[1] if len(top_factor_labels) > 1 else "",
            "top_factor_3":    top_factor_labels[2] if len(top_factor_labels) > 2 else "",
            "shap_factor_1":   top_factor_shaps[0]  if len(top_factor_shaps) > 0 else 0,
            "shap_factor_2":   top_factor_shaps[1]  if len(top_factor_shaps) > 1 else 0,
            "shap_factor_3":   top_factor_shaps[2]  if len(top_factor_shaps) > 2 else 0,
            "waterfall_plot":  d["waterfall_plot"],
        })

    diag_df = pd.DataFrame(flat_rows)
    diag_df.to_csv(DIAGNOSIS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  診斷書 CSV：{DIAGNOSIS_CSV}")

    # 存成 JSON（完整版，含 top_factors 詳細資料）
    with open(DIAGNOSIS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_diagnoses, f, ensure_ascii=False, indent=2)
    print(f"  診斷書 JSON：{DIAGNOSIS_JSON}")

    # 印出一份完整範例
    if all_diagnoses:
        print("\n" + "=" * 55)
        print("【診斷書範例（第一名高風險用戶）】")
        print("=" * 55)
        ex = all_diagnoses[0]
        print(f"  用戶 ID    ：{ex['user_id']}")
        print(f"  風險評級   ：{ex['risk_emoji']} {ex['risk_level']}風險")
        print(f"  風險分數   ：{ex['risk_score']}")
        print(f"  真實標籤   ：{'黑名單' if ex['true_label'] == 1 else '正常' if ex['true_label'] == 0 else '未知'}")
        print(f"\n  風險摘要：")
        print(f"    {ex['nl_summary']}")
        print(f"\n  Top 5 風險因子：")
        for fac in ex["top_factors"]:
            print(f"    {fac['rank']}. {fac['label']:<20}  "
                  f"數值={fac['value']:<10}  SHAP={fac['shap']:+.4f}  {fac['direction']}")
        print(f"\n  Waterfall 圖：{ex['waterfall_plot']}")

    print("\nSHAP 解釋分析完成！")
    print("下一步：執行 src/models/graph_analysis.py")


if __name__ == "__main__":
    main()