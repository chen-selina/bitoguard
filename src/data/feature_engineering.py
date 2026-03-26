"""
feature_engineering.py
位置：src/data/feature_engineering.py

【v4.0 修正版 — 解釋性優先 + 完整時間特徵】
主要修正：
  1. build_kyc_features：補上時間差衍生欄位
       - account_age_days    （帳號存活天數）
       - kyc_l1_to_l2_days   （KYC 升級速度，詐騙帳號往往極快）
       - confirmed_to_l2_days（從註冊到完成 Level2 的天數）
       - kyc_rushed          （KYC 升級 < 1 天的 binary flag）
  2. build_trading_features：補上 updated_at parse 與交易時間特徵
       - trade_night_ratio   （深夜交易比例）
       - trade_active_days   （交易活躍天數）
       - swap_twd_total      （兌換 TWD 總金額）
  3. main()：加入白名單篩選，控制輸出 feature 在 25–28 個
     （高解釋性目標：少而精，每個 feature 可用白話說明）
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUTPUT_PATH   = PROCESSED_DIR / "features.csv"
PREDICT_OUTPUT_PATH = PROCESSED_DIR / "predict_features.csv"

# ============================================================
# 白名單：最終保留的 25–28 個 feature（依解釋性優先排序）
# 若某欄位不存在（如 crypto 表缺失）會自動略過
# ============================================================
SELECTED_FEATURES = [
    # --- 用戶基本資料（3 個）---
    "age",                      # 年齡：年輕低收帳號風險較高
    "career_code",              # 職業代碼：特定職業與詐騙相關
    "kyc_level",                # KYC 完成等級（0/1/2）

    # --- KYC 時間特徵（4 個）---
    "account_age_days",         # 帳號存活天數：新帳號風險高
    "kyc_l1_to_l2_days",        # Level1→Level2 升級天數：< 1 天異常
    "confirmed_to_l2_days",     # 註冊→Level2 完成天數
    "kyc_rushed",               # KYC 升級 < 1 天：binary flag（強詐騙訊號）

    # --- TWD 轉帳行為（5 個）---
    "twd_deposit_count",        # 入金次數
    "twd_deposit_total",        # 入金總額
    "twd_withdraw_total",       # 出金總額
    "twd_inout_ratio",          # 出/入金比（接近 1 = 快進快出）
    "twd_fast_exit_ratio",      # 1 小時內快速出金的比例

    # --- TWD 異常行為（3 個）---
    "twd_night_ratio",          # 深夜交易比例（23:00–05:59）
    "twd_round_amount_ratio",   # 整數金額比率（10000 的倍數，疑似分拆）
    "twd_active_days",          # 活躍天數（越少 + 量越大越可疑）

    # --- Crypto 特徵（3 個）---
    "crypto_total_count",       # Crypto 出入金總次數
    "crypto_inout_ratio",       # Crypto 出/入金比
    "crypto_night_ratio",       # Crypto 深夜操作比例

    # --- 交易行為（4 個）---
    "trade_count",              # USDT/TWD 交易次數
    "trade_buy_ratio",          # 買入比例（偏向單向操作是異常訊號）
    "trade_night_ratio",        # 深夜交易比例
    "swap_count",               # USDT 兌換次數

    # --- IP 特徵（2 個）---
    "ip_unique_count",          # 使用過的不重複 IP 數（多 IP = 多帳號跡象）
    "ip_diversity_ratio",       # IP 多樣性比率

    # --- 跨表複合特徵（4 個）---
    "twd_to_crypto_gap_hr",     # TWD 入金→Crypto 出金最短時間差（小時）
    "activity_density",         # 活躍密度（操作次數 / 活躍天數）
    "feat_instant_wash_risk",   # 秒洗風險：滯留 < 5 分鐘且快出比 > 0.8
    "feat_low_kyc_high_vol",    # 低 KYC 卻有高額入金（人頭戶強訊號）
]


# ============================================================
# 1. 載入與工具
# ============================================================

def load_tables() -> dict:
    tables = {}
    files = {
        "user_info":     "user_info.csv",
        "twd":           "twd_transfer.csv",
        "crypto":        "crypto_transfer.csv",
        "usdt_trade":    "usdt_twd_trading.csv",
        "usdt_swap":     "usdt_swap.csv",
        "train_label":   "train_label.csv",
        "predict_label": "predict_label.csv",
    }
    for key, fname in files.items():
        path = RAW_DIR / fname
        if path.exists():
            tables[key] = pd.read_csv(path, low_memory=False)
            print(f"  載入 {fname}：{len(tables[key]):,} 筆")
        else:
            print(f"  警告：找不到 {fname}，跳過")
            tables[key] = pd.DataFrame()
    return tables


def parse_datetime(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """將指定欄位轉換為 datetime，無法解析的設為 NaT。"""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ============================================================
# 2. 用戶基本資料 & KYC 時間特徵
# ============================================================

def build_kyc_features(user: pd.DataFrame) -> pd.DataFrame:
    """
    來源：user_info.csv
    產出：
      - age, career_code, kyc_level          （基本資料）
      - account_age_days                      （帳號存活天數）
      - kyc_l1_to_l2_days                     （KYC Level1→Level2 升級速度）
      - confirmed_to_l2_days                  （從確認帳號到完成 Level2）
      - kyc_rushed                            （KYC < 1 天：binary 詐騙訊號）
    """
    if user.empty:
        return pd.DataFrame()

    user = user.copy()
    user = parse_datetime(user, ["confirmed_at", "level1_finished_at", "level2_finished_at"])

    feat = user[["user_id"]].copy()

    # --- 基本資料 ---
    feat["age"] = pd.to_numeric(user.get("age"), errors="coerce").fillna(0)
    feat["career_code"] = user.get("career", pd.Series(0, index=user.index)).fillna(0).astype(int)

    # --- KYC 完成等級 ---
    feat["kyc_level"] = 0
    feat.loc[user["confirmed_at"].notna(), "kyc_level"] = 1
    feat.loc[user["level2_finished_at"].notna(), "kyc_level"] = 2

    # --- 【新增】帳號存活天數：from confirmed_at to now ---
    now = pd.Timestamp.now(tz=None)  # naive timestamp，配合資料
    feat["account_age_days"] = (
        (now - user["confirmed_at"]).dt.total_seconds() / 86400
    ).clip(lower=0).fillna(0)

    # --- 【新增】KYC 升級速度 ---
    # Level1 → Level2 的天數（越短越可疑）
    feat["kyc_l1_to_l2_days"] = (
        (user["level2_finished_at"] - user["level1_finished_at"])
        .dt.total_seconds() / 86400
    ).clip(lower=0).fillna(-1)   # -1 表示未完成 Level2

    # 從確認帳號到完成 Level2 的天數
    feat["confirmed_to_l2_days"] = (
        (user["level2_finished_at"] - user["confirmed_at"])
        .dt.total_seconds() / 86400
    ).clip(lower=0).fillna(-1)

    # --- 【新增】KYC 極速升級 flag（< 1 天 = 異常）---
    feat["kyc_rushed"] = (
        feat["kyc_l1_to_l2_days"].between(0, 1, inclusive="neither")
    ).astype(int)

    print(f"  KYC 特徵：{len(feat)} 位用戶，"
          f"kyc_rushed 比例：{feat['kyc_rushed'].mean():.4f}")
    return feat


# ============================================================
# 3. 法幣轉帳特徵（twd_transfer）
# ============================================================

def build_twd_features(twd: pd.DataFrame) -> pd.DataFrame:
    """
    來源：twd_transfer.csv（kind=0 入金，kind=1 出金）
    產出：入出金統計、快進快出比、深夜比、整數金額比、活躍天數
    """
    if twd.empty:
        return pd.DataFrame()

    twd = twd.copy()
    twd = parse_datetime(twd, ["created_at"])
    amount_col = "ori_amount" if "ori_amount" in twd.columns else "ori_samount"

    deposit  = twd[twd["kind"] == 0].copy()
    withdraw = twd[twd["kind"] == 1].copy()

    dep_stats = deposit.groupby("user_id")[amount_col].agg(
        twd_deposit_count="count",
        twd_deposit_total="sum",
    ).reset_index()

    wd_stats = withdraw.groupby("user_id")[amount_col].agg(
        twd_withdraw_count="count",
        twd_withdraw_total="sum",
    ).reset_index()

    feat = dep_stats.merge(wd_stats, on="user_id", how="outer").fillna(0)
    feat["twd_inout_ratio"] = (
        feat["twd_withdraw_total"] / (feat["twd_deposit_total"] + 1e-9)
    )

    # A. 快進快出分析（入金 → 出金時間差）
    if not deposit.empty and not withdraw.empty:
        dep_t = (deposit[["user_id", "created_at"]]
                 .rename(columns={"created_at": "dep_t"}))
        wd_t  = (withdraw[["user_id", "created_at"]]
                 .rename(columns={"created_at": "wd_t"}))
        paired = dep_t.merge(wd_t, on="user_id")
        paired = paired[paired["wd_t"] >= paired["dep_t"]]
        paired["stay_sec"] = (paired["wd_t"] - paired["dep_t"]).dt.total_seconds()

        stay = (paired.groupby("user_id")["stay_sec"]
                .agg(twd_stay_mean="mean", twd_stay_min="min")
                .reset_index())
        paired["is_fast"] = (paired["stay_sec"] < 3600).astype(int)
        fast_ratio = (paired.groupby("user_id")["is_fast"]
                      .mean().reset_index(name="twd_fast_exit_ratio"))

        feat = (feat
                .merge(stay, on="user_id", how="left")
                .merge(fast_ratio, on="user_id", how="left"))

    # B. 深夜交易比例（23:00–05:59）
    twd["hour"] = twd["created_at"].dt.hour
    twd["is_night"] = (twd["hour"] >= 23) | (twd["hour"] <= 5)
    night_stats = (twd.groupby("user_id")["is_night"]
                   .mean().reset_index(name="twd_night_ratio"))
    feat = feat.merge(night_stats, on="user_id", how="left")

    # C. 活躍天數 & 爆量比（前三天的交易量佔比）
    twd["date"] = twd["created_at"].dt.date
    daily = twd.groupby(["user_id", "date"]).size().reset_index(name="cnt")
    active_days = (daily.groupby("user_id")["date"]
                   .nunique().reset_index(name="twd_active_days"))
    burst_ratio = (daily.groupby("user_id")["cnt"]
                   .apply(lambda x: x.nlargest(3).sum() / (x.sum() + 1e-9))
                   .reset_index(name="twd_burst_ratio"))
    feat = feat.merge(active_days, on="user_id", how="left")
    feat = feat.merge(burst_ratio, on="user_id", how="left")

    # D. 整數金額比率（10000 的倍數，疑似分拆洗錢）
    round_r = (twd.groupby("user_id")
               .apply(lambda x: ((x[amount_col] % 10000 == 0) & (x[amount_col] > 0)).mean())
               .reset_index(name="twd_round_amount_ratio"))
    feat = feat.merge(round_r, on="user_id", how="left")

    print(f"  TWD 特徵：{len(feat)} 位用戶")
    return feat


# ============================================================
# 4. IP 多樣性特徵（跨三張交易表）
# ============================================================

def build_ip_features(twd: pd.DataFrame,
                      crypto: pd.DataFrame,
                      trade: pd.DataFrame) -> pd.DataFrame:
    """
    來源：twd_transfer、crypto_transfer、usdt_twd_trading 的 source_ip_hash
    產出：
      - ip_unique_count     不重複 IP 數（多 = 多裝置/多帳號跡象）
      - ip_diversity_ratio  IP 多樣性比率
    """
    ip_col = "source_ip_hash"
    dfs = [
        df[["user_id", ip_col]]
        for df in [twd, crypto, trade]
        if not df.empty and ip_col in df.columns
    ]
    if not dfs:
        return pd.DataFrame()

    all_ip = pd.concat(dfs).dropna()
    feat = (all_ip.groupby("user_id")[ip_col]
            .agg(ip_unique_count="nunique", ip_total_count="count")
            .reset_index())
    feat["ip_diversity_ratio"] = (
        feat["ip_unique_count"] / (feat["ip_total_count"] + 1e-9)
    )
    print(f"  IP 特徵：{len(feat)} 位用戶")
    return feat


# ============================================================
# 5. Crypto 出入金特徵（crypto_transfer）
# ============================================================

def build_crypto_features(crypto: pd.DataFrame) -> pd.DataFrame:
    """
    來源：crypto_transfer.csv（kind=0 入，kind=1 出）
    產出：次數、出入比、深夜比
    """
    if crypto.empty:
        return pd.DataFrame()

    crypto = crypto.copy()
    crypto = parse_datetime(crypto, ["created_at"])
    amt_col = "ori_amount" if "ori_amount" in crypto.columns else "ori_samount"

    # 分別計算出入金
    if "kind" in crypto.columns:
        dep_total = (crypto[crypto["kind"] == 0]
                     .groupby("user_id")[amt_col].sum()
                     .reset_index(name="crypto_deposit_total"))
        wd_total  = (crypto[crypto["kind"] == 1]
                     .groupby("user_id")[amt_col].sum()
                     .reset_index(name="crypto_withdraw_total"))
    else:
        dep_total = pd.DataFrame(columns=["user_id", "crypto_deposit_total"])
        wd_total  = pd.DataFrame(columns=["user_id", "crypto_withdraw_total"])

    crypto["is_night"] = (
        crypto["created_at"].dt.hour.isin([23, 0, 1, 2, 3, 4, 5])
    )
    base = (crypto.groupby("user_id")
            .agg(
                crypto_total_count=("user_id", "count"),
                crypto_night_ratio=("is_night", "mean"),
            ).reset_index())

    feat = (base
            .merge(dep_total, on="user_id", how="left")
            .merge(wd_total,  on="user_id", how="left")
            .fillna(0))
    feat["crypto_inout_ratio"] = (
        feat["crypto_withdraw_total"] / (feat["crypto_deposit_total"] + 1e-9)
    )

    print(f"  Crypto 特徵：{len(feat)} 位用戶")
    return feat


# ============================================================
# 6. 交易行為特徵（usdt_twd_trading + usdt_swap）
# ============================================================

def build_trading_features(usdt_trade: pd.DataFrame,
                            usdt_swap: pd.DataFrame) -> pd.DataFrame:
    """
    來源：usdt_twd_trading.csv、usdt_swap.csv
    產出：
      - trade_count         交易次數
      - trade_buy_ratio     買入比例
      - trade_night_ratio   【新增】深夜交易比例（23:00–05:59）
      - trade_active_days   【新增】交易活躍天數
      - swap_count          兌換次數
      - swap_twd_total      【新增】兌換 TWD 總金額
    """
    frames = []

    if not usdt_trade.empty:
        trade = usdt_trade.copy()
        # 【新增】parse updated_at
        trade = parse_datetime(trade, ["updated_at"])

        trade["is_night"] = (
            trade["updated_at"].dt.hour.isin([23, 0, 1, 2, 3, 4, 5])
        )
        trade["date"] = trade["updated_at"].dt.date

        trade_feat = (trade.groupby("user_id")
                      .agg(
                          trade_count=("user_id", "count"),
                          trade_buy_ratio=("is_buy", "mean"),
                          trade_night_ratio=("is_night", "mean"),    # 【新增】
                      ).reset_index())

        # 【新增】活躍天數
        active = (trade.groupby("user_id")["date"]
                  .nunique().reset_index(name="trade_active_days"))
        trade_feat = trade_feat.merge(active, on="user_id", how="left")
        frames.append(trade_feat)

    if not usdt_swap.empty:
        swap = usdt_swap.copy()
        twd_amt = "twd_amount" if "twd_amount" in swap.columns else "twd_samount"

        swap_feat = (swap.groupby("user_id")
                     .agg(
                         swap_count=("user_id", "count"),
                         swap_twd_total=(twd_amt, "sum"),             # 【新增】
                     ).reset_index())
        frames.append(swap_feat)

    if not frames:
        return pd.DataFrame()

    feat = frames[0]
    for f in frames[1:]:
        feat = feat.merge(f, on="user_id", how="outer")

    print(f"  交易特徵：{len(feat)} 位用戶")
    return feat.fillna(0)


# ============================================================
# 7. 跨表複合特徵（秒洗偵測、低KYC高流量、爆發強度）
# ============================================================

def build_cross_table_features(twd: pd.DataFrame,
                                crypto: pd.DataFrame,
                                user: pd.DataFrame,
                                twd_feat: pd.DataFrame) -> pd.DataFrame:
    """
    來源：跨多張表
    產出：
      - twd_to_crypto_gap_hr   TWD 入金→Crypto 出金最短時間差（小時）
      - activity_density       所有操作的活躍密度（次數 / 天數跨度）
      - feat_instant_wash_risk 秒洗 binary flag
      - feat_low_kyc_high_vol  低 KYC 高額入金 binary flag
    """
    results = []

    # A. TWD 入金 → Crypto 出金最短時間差
    if not twd.empty and not crypto.empty:
        twd_t  = parse_datetime(twd.copy(),    ["created_at"])
        cry_t  = parse_datetime(crypto.copy(), ["created_at"])
        dep    = (twd_t[twd_t["kind"] == 0][["user_id", "created_at"]]
                  .rename(columns={"created_at": "t_in"}))
        wdc    = (cry_t[cry_t["kind"] == 1][["user_id", "created_at"]]
                  .rename(columns={"created_at": "t_out"}))
        merged = dep.merge(wdc, on="user_id")
        merged = merged[merged["t_out"] >= merged["t_in"]]
        merged["gap"] = (merged["t_out"] - merged["t_in"]).dt.total_seconds() / 3600
        cross_gap = (merged.groupby("user_id")["gap"]
                     .min().reset_index(name="twd_to_crypto_gap_hr"))
        results.append(cross_gap)

    # B. 整體活躍密度（TWD + Crypto 的時間跨度）
    parts = []
    if not twd.empty and "created_at" in twd.columns:
        parts.append(twd[["user_id", "created_at"]])
    if not crypto.empty and "created_at" in crypto.columns:
        parts.append(crypto[["user_id", "created_at"]])

    if parts:
        all_t = pd.concat(parts).dropna()
        all_t["created_at"] = pd.to_datetime(all_t["created_at"], errors="coerce")
        span = (all_t.groupby("user_id")["created_at"]
                .agg(["min", "max", "count"]).reset_index())
        span["first_to_last_days"] = (
            (span["max"] - span["min"]).dt.total_seconds() / 86400
        )
        span["activity_density"] = span["count"] / (span["first_to_last_days"] + 1)
        results.append(span[["user_id", "first_to_last_days", "activity_density"]])

    # 合併 A + B
    if not results:
        return pd.DataFrame()
    feat = results[0]
    for r in results[1:]:
        feat = feat.merge(r, on="user_id", how="outer")

    # C. 秒洗風險：滯留時間 < 5 分鐘 且 快出比 > 80%
    if ("twd_stay_min" in twd_feat.columns and
            "twd_fast_exit_ratio" in twd_feat.columns):
        feat = feat.merge(
            twd_feat[["user_id", "twd_stay_min", "twd_fast_exit_ratio"]],
            on="user_id", how="left"
        )
        feat["feat_instant_wash_risk"] = (
            (feat["twd_stay_min"] < 300) & (feat["twd_fast_exit_ratio"] > 0.8)
        ).astype(int)
        feat.drop(columns=["twd_stay_min", "twd_fast_exit_ratio"], inplace=True)

    # D. 低 KYC 卻有高額入金（人頭戶強訊號）
    user_kyc = build_kyc_features(user)
    if not user_kyc.empty and "twd_deposit_total" in twd_feat.columns:
        tmp = (twd_feat[["user_id", "twd_deposit_total"]]
               .merge(user_kyc[["user_id", "kyc_level"]], on="user_id"))
        tmp["feat_low_kyc_high_vol"] = (
            (tmp["kyc_level"] < 2) & (tmp["twd_deposit_total"] > 200000)
        ).astype(int)
        feat = feat.merge(tmp[["user_id", "feat_low_kyc_high_vol"]],
                          on="user_id", how="left")

    print(f"  跨表特徵：{len(feat)} 位用戶")
    return feat.fillna(0)


# ============================================================
# 8. 主流程
# ============================================================

def main():
    print("=" * 60)
    print("BitoGuard v4.0 — 解釋性優先：完整時間特徵 + 精選 25–28 個 feature")
    print("=" * 60)

    tables = load_tables()
    train_label   = tables.get("train_label",   pd.DataFrame())
    predict_label = tables.get("predict_label", pd.DataFrame())

    # --- 建立各表特徵 ---
    print("\n【建立 KYC 特徵】")
    kyc_f = build_kyc_features(tables["user_info"])

    print("\n【建立 TWD 法幣特徵】")
    twd_f = build_twd_features(tables["twd"])

    print("\n【建立 IP 特徵】")
    ip_f = build_ip_features(tables["twd"], tables["crypto"], tables["usdt_trade"])

    print("\n【建立 Crypto 特徵】")
    cry_f = build_crypto_features(tables["crypto"])

    print("\n【建立交易特徵（含時間）】")
    td_f = build_trading_features(tables["usdt_trade"], tables["usdt_swap"])

    print("\n【建立跨表複合特徵】")
    cross_f = build_cross_table_features(
        tables["twd"], tables["crypto"], tables["user_info"], twd_f
    )

    # --- 合併所有特徵 ---
    print("\n【合併特徵表】")
    all_feat = (kyc_f
                .merge(twd_f,   on="user_id", how="left")
                .merge(ip_f,    on="user_id", how="left")
                .merge(cry_f,   on="user_id", how="left")
                .merge(td_f,    on="user_id", how="left")
                .merge(cross_f, on="user_id", how="left"))

    # 填充數值空值為 0
    num_cols = all_feat.select_dtypes(include=[np.number]).columns
    all_feat[num_cols] = all_feat[num_cols].fillna(0)

    # --- 【白名單篩選】只保留指定的 25–28 個 feature ---
    existing = [f for f in SELECTED_FEATURES if f in all_feat.columns]
    missing  = [f for f in SELECTED_FEATURES if f not in all_feat.columns]
    if missing:
        print(f"\n  ⚠️  以下 feature 不存在（可能對應資料表缺失），已略過：")
        for m in missing:
            print(f"       - {m}")

    keep_cols = ["user_id"] + existing
    all_feat  = all_feat[keep_cols]

    print(f"\n  ✅ 最終保留 feature 數：{len(existing)}")
    print(f"  Feature 清單：{existing}")

    # --- 分離訓練集 & 預測集 ---
    train_feat = (all_feat[all_feat["user_id"].isin(train_label["user_id"])]
                  .copy()
                  .merge(
                      train_label[["user_id", "status"]].rename(columns={"status": "label"}),
                      on="user_id"
                  ))
    train_feat.to_csv(TRAIN_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    pred_feat = all_feat[all_feat["user_id"].isin(predict_label["user_id"])].copy()
    pred_feat.to_csv(PREDICT_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n✅ 特徵工程完成！")
    print(f"   訓練集：{len(train_feat):,} 筆，{len(train_feat.columns)-2} 個 feature")
    print(f"   預測集：{len(pred_feat):,} 筆")
    print(f"   已存檔：{TRAIN_OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()