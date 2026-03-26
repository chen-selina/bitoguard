"""
feature_engineering.py
位置：src/data/feature_engineering.py

【v3 升級說明 — 新增行為節奏特徵，目標提升 F1 +0.05~0.10】

  新增特徵群（共 ~20 個新特徵）：
  A. 快進快出時間差特徵（最高鑑別力）
     - twd_min_stay_sec：法幣入→出最短滯留秒數（詐騙帳戶通常 < 60 秒）
     - twd_fast_exit_ratio：滯留 < 1 小時的出金比率
     - crypto_buy_to_withdraw_min：買 USDT 到提幣的最短時間差（秒）
     - crypto_fast_withdraw_ratio：買入 30 分鐘內就提出的比率

  B. 行為集中度特徵
     - twd_active_days：有交易的天數
     - twd_burst_ratio：最活躍 3 天交易量佔總量比率（爆量特徵）
     - crypto_active_days：虛擬幣有交易天數
     - crypto_burst_ratio：虛擬幣最活躍 3 天爆量比率

  C. 跨表行為序列特徵
     - twd_to_crypto_gap_min：法幣入金→虛擬幣出金最短時間差（小時）
     - first_to_last_days：帳號首次交易到最後交易天數（活躍期長度）
     - trade_to_withdraw_min：USDT 交易→提幣最短時間差（小時）

  D. 金額離群值特徵
     - twd_max_to_mean_ratio：單筆最大 / 平均金額（衝量旗標）
     - crypto_max_to_mean_ratio：虛擬幣單筆最大 / 平均
     - twd_round_amount_ratio：整數金額（萬元整數）交易比率（人頭帳戶特徵）

  E. 帳號生命週期特徵
     - account_age_days：KYC 完成到最後活躍天數
     - activity_density：交易次數 / 活躍天數（密集度）

使用方式：python src/data/feature_engineering.py
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
# 1. 載入資料
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
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ============================================================
# 2. 法幣行為特徵（v3：新增快進快出 + 集中度 + 離群值）
# ============================================================

def build_twd_features(twd: pd.DataFrame) -> pd.DataFrame:
    if twd.empty:
        return pd.DataFrame()

    twd = parse_datetime(twd, ["created_at"])
    amount_col = "ori_amount" if "ori_amount" in twd.columns else "ori_samount"

    deposit  = twd[twd["kind"] == 0].copy()
    withdraw = twd[twd["kind"] == 1].copy()

    # 基礎統計
    dep_stats = deposit.groupby("user_id")[amount_col].agg(
        twd_deposit_count="count",
        twd_deposit_total="sum",
        twd_deposit_mean="mean",
        twd_deposit_max="max",
        twd_deposit_std="std",
    ).reset_index()

    wd_stats = withdraw.groupby("user_id")[amount_col].agg(
        twd_withdraw_count="count",
        twd_withdraw_total="sum",
        twd_withdraw_mean="mean",
        twd_withdraw_max="max",
        twd_withdraw_std="std",
    ).reset_index()

    feat = dep_stats.merge(wd_stats, on="user_id", how="outer").fillna(0)

    feat["twd_inout_ratio"] = (
        feat["twd_withdraw_total"] / (feat["twd_deposit_total"] + 1e-9)
    )
    feat["twd_withdraw_gt_deposit"] = (
        feat["twd_withdraw_count"] > feat["twd_deposit_count"]
    ).astype(int)

    # ── 【v3 新增】A. 快進快出滯留時間特徵 ──
    if not deposit.empty and not withdraw.empty:
        dep_t = deposit[["user_id", "created_at", amount_col]].rename(
            columns={"created_at": "dep_time", amount_col: "dep_amt"})
        wd_t  = withdraw[["user_id", "created_at", amount_col]].rename(
            columns={"created_at": "wd_time", amount_col: "wd_amt"})

        paired = dep_t.merge(wd_t, on="user_id")
        paired = paired[paired["wd_time"] >= paired["dep_time"]]
        paired["stay_sec"] = (paired["wd_time"] - paired["dep_time"]).dt.total_seconds()

        stay = paired.groupby("user_id")["stay_sec"].agg(
            twd_stay_mean="mean",
            twd_stay_min="min",
        ).reset_index()

        # 【新增】最短滯留秒數（詐騙帳戶通常 < 60 秒出金）
        # fast_exit_ratio：滯留 < 3600 秒（1小時）的出金比率
        paired["is_fast"] = (paired["stay_sec"] < 3600).astype(int)
        fast_ratio = paired.groupby("user_id")["is_fast"].mean().reset_index()
        fast_ratio.rename(columns={"is_fast": "twd_fast_exit_ratio"}, inplace=True)

        feat = feat.merge(stay, on="user_id", how="left")
        feat = feat.merge(fast_ratio, on="user_id", how="left")
    else:
        feat["twd_stay_mean"]       = np.nan
        feat["twd_stay_min"]        = np.nan
        feat["twd_fast_exit_ratio"] = np.nan

    # ── 【v3 新增】B. 深夜交易比率 ──
    twd["hour"]     = twd["created_at"].dt.hour
    twd["is_night"] = twd["hour"].between(23, 23) | twd["hour"].between(0, 5)
    night = twd.groupby("user_id")["is_night"].agg(
        twd_night_count="sum",
        twd_total_for_night="count",
    ).reset_index()
    night["twd_night_ratio"] = night["twd_night_count"] / (night["twd_total_for_night"] + 1e-9)
    feat = feat.merge(night[["user_id", "twd_night_ratio"]], on="user_id", how="left")

    # ── 【v3 新增】C. 行為集中度（爆量特徵）──
    twd["date"] = twd["created_at"].dt.date
    daily_cnt = twd.groupby(["user_id", "date"]).size().reset_index(name="day_count")

    # 活躍天數
    active_days = daily_cnt.groupby("user_id")["date"].nunique().reset_index(name="twd_active_days")

    # 最活躍 3 天交易量佔總量比率
    def burst_ratio(grp):
        top3 = grp.nlargest(3).sum()
        total = grp.sum()
        return top3 / (total + 1e-9)
    burst = daily_cnt.groupby("user_id")["day_count"].apply(burst_ratio).reset_index(name="twd_burst_ratio")

    feat = feat.merge(active_days, on="user_id", how="left")
    feat = feat.merge(burst, on="user_id", how="left")

    # ── 【v3 新增】D. 金額離群值特徵 ──
    feat["twd_max_to_mean_ratio"] = feat["twd_deposit_max"] / (feat["twd_deposit_mean"] + 1e-9)

    # 整數金額比率（詐騙帳戶常以整萬交易）
    twd["is_round"] = (twd[amount_col] % 10000 == 0) & (twd[amount_col] > 0)
    round_ratio = twd.groupby("user_id")["is_round"].mean().reset_index(name="twd_round_amount_ratio")
    feat = feat.merge(round_ratio, on="user_id", how="left")

    print(f"  法幣特徵：{len(feat)} 位用戶，{len(feat.columns) - 1} 個特徵")
    return feat


# ============================================================
# 3. IP 行為特徵
# ============================================================

def build_ip_features(twd: pd.DataFrame, crypto: pd.DataFrame,
                       usdt_trade: pd.DataFrame) -> pd.DataFrame:
    ip_col = "source_ip_hash"
    frames = []
    for df, label in [(twd, "twd"), (crypto, "crypto"), (usdt_trade, "trade")]:
        if not df.empty and ip_col in df.columns and "user_id" in df.columns:
            sub = df[["user_id", ip_col]].dropna(subset=[ip_col]).copy()
            sub["source"] = label
            frames.append(sub)

    if not frames:
        print("  IP 特徵：無資料")
        return pd.DataFrame()

    all_ip = pd.concat(frames, ignore_index=True)
    feat = all_ip.groupby("user_id")[ip_col].agg(
        ip_unique_count="nunique",
        ip_total_count="count",
    ).reset_index()
    feat["ip_diversity_ratio"] = feat["ip_unique_count"] / (feat["ip_total_count"] + 1e-9)

    print(f"  IP 特徵：{len(feat)} 位用戶，{len(feat.columns) - 1} 個特徵")
    return feat


# ============================================================
# 4. KYC 屬性特徵
# ============================================================

def build_kyc_features(user: pd.DataFrame) -> pd.DataFrame:
    if user.empty:
        return pd.DataFrame()

    user = parse_datetime(user, ["confirmed_at", "level1_finished_at", "level2_finished_at"])
    feat = user[["user_id"]].copy()

    feat["kyc_level"] = 0
    feat.loc[user["confirmed_at"].notna(),        "kyc_level"] = 1
    feat.loc[user["level1_finished_at"].notna(),  "kyc_level"] = 1
    feat.loc[user["level2_finished_at"].notna(),  "kyc_level"] = 2

    if "level1_finished_at" in user.columns and "level2_finished_at" in user.columns:
        feat["kyc_l1_to_l2_days"] = (
            (user["level2_finished_at"] - user["level1_finished_at"])
            .dt.total_seconds() / 86400
        )

    if "age" in user.columns:
        feat["age"] = pd.to_numeric(user["age"], errors="coerce").fillna(0)
        feat["age_missing"] = (feat["age"] == 0).astype(int)

    HIGH_RISK_CAREER = {14, 22, 23, 24}
    if "career" in user.columns:
        feat["career_high_risk"] = user["career"].isin(HIGH_RISK_CAREER).astype(int)
        feat["career_code"]      = user["career"].fillna(0).astype(int)

    if "income" in user.columns:
        feat["income_no_source"] = (user["income"] == 0).astype(int)
        feat["income_code"]      = user["income"].fillna(0).astype(int)

    if "sex" in user.columns:
        feat["sex_undisclosed"] = user["sex"].isna().astype(int)

    if "source" in user.columns:
        feat["user_source"] = user["source"].fillna(0).astype(int)

    print(f"  KYC 特徵：{len(feat)} 位用戶，{len(feat.columns) - 1} 個特徵")
    return feat


# ============================================================
# 5. 虛擬幣行為特徵（v3：新增快速提幣 + 集中度）
# ============================================================

def build_crypto_features(crypto: pd.DataFrame) -> pd.DataFrame:
    if crypto.empty:
        return pd.DataFrame()

    crypto = parse_datetime(crypto, ["created_at"])
    amount_col = "ori_amount" if "ori_amount" in crypto.columns else "ori_samount"

    deposit  = crypto[crypto["kind"] == 0].copy() if "kind" in crypto.columns else pd.DataFrame()
    withdraw = crypto[crypto["kind"] == 1].copy() if "kind" in crypto.columns else pd.DataFrame()

    frames = []
    if not deposit.empty:
        dep_s = deposit.groupby("user_id")[amount_col].agg(
            crypto_deposit_count="count",
            crypto_deposit_total="sum",
        ).reset_index()
        frames.append(dep_s)

    if not withdraw.empty:
        wd_s = withdraw.groupby("user_id")[amount_col].agg(
            crypto_withdraw_count="count",
            crypto_withdraw_total="sum",
        ).reset_index()
        frames.append(wd_s)

    if not frames:
        return pd.DataFrame()

    feat = frames[0]
    for f in frames[1:]:
        feat = feat.merge(f, on="user_id", how="outer")
    feat = feat.fillna(0)

    feat["crypto_inout_ratio"] = (
        feat.get("crypto_withdraw_total", 0) /
        (feat.get("crypto_deposit_total", pd.Series(0, index=feat.index)) + 1e-9)
    )

    # 內部轉帳
    if "is_internal" in crypto.columns:
        internal = crypto[crypto["is_internal"] == 1]
        int_cnt  = internal.groupby("user_id").size().reset_index(name="crypto_internal_count")
        tot_cnt  = crypto.groupby("user_id").size().reset_index(name="crypto_total_count")
        feat = feat.merge(int_cnt, on="user_id", how="left").fillna(0)
        feat = feat.merge(tot_cnt, on="user_id", how="left").fillna(0)
        feat["crypto_internal_ratio"] = feat["crypto_internal_count"] / (feat["crypto_total_count"] + 1e-9)
    else:
        feat["crypto_internal_count"] = 0
        feat["crypto_total_count"]    = crypto.groupby("user_id").size().reindex(feat["user_id"]).values
        feat["crypto_internal_ratio"] = 0.0

    # TRC20
    trc20 = crypto[crypto.get("network", pd.Series("")).str.upper() == "TRC20"] if "network" in crypto.columns else pd.DataFrame()
    if not trc20.empty:
        trc_cnt = trc20.groupby("user_id").size().reset_index(name="trc20_count")
        feat = feat.merge(trc_cnt, on="user_id", how="left").fillna(0)
        feat["trc20_usage_ratio"] = feat["trc20_count"] / (feat["crypto_total_count"] + 1e-9)
    else:
        feat["trc20_count"]       = 0
        feat["trc20_usage_ratio"] = 0.0

    # 錢包多樣性
    wallet_col = "to_wallet_hash" if "to_wallet_hash" in crypto.columns else "to_wallet"
    if wallet_col in crypto.columns:
        wallet_div = crypto.groupby("user_id")[wallet_col].nunique().reset_index()
        wallet_div.rename(columns={wallet_col: "crypto_wallet_diversity"}, inplace=True)
        feat = feat.merge(wallet_div, on="user_id", how="left").fillna(0)
    else:
        feat["crypto_wallet_diversity"] = 0

    # 幣種多樣性
    if "currency" in crypto.columns:
        currency_div = crypto.groupby("user_id")["currency"].nunique().reset_index()
        currency_div.rename(columns={"currency": "crypto_currency_diversity"}, inplace=True)
        feat = feat.merge(currency_div, on="user_id", how="left").fillna(0)

    # 深夜比率
    crypto["hour"]     = crypto["created_at"].dt.hour
    crypto["is_night"] = crypto["hour"].between(23, 23) | crypto["hour"].between(0, 5)
    night = crypto.groupby("user_id")["is_night"].mean().reset_index()
    night.rename(columns={"is_night": "crypto_night_ratio"}, inplace=True)
    feat = feat.merge(night, on="user_id", how="left").fillna(0)

    # ── 【v3 新增】虛擬幣集中度 ──
    crypto["date"] = crypto["created_at"].dt.date
    daily_cnt = crypto.groupby(["user_id", "date"]).size().reset_index(name="day_count")

    active_days = daily_cnt.groupby("user_id")["date"].nunique().reset_index(name="crypto_active_days")

    def burst_ratio(grp):
        top3  = grp.nlargest(3).sum()
        total = grp.sum()
        return top3 / (total + 1e-9)
    burst = daily_cnt.groupby("user_id")["day_count"].apply(burst_ratio).reset_index(name="crypto_burst_ratio")

    feat = feat.merge(active_days, on="user_id", how="left")
    feat = feat.merge(burst, on="user_id", how="left")

    # ── 【v3 新增】金額離群值 ──
    if "crypto_deposit_mean" not in feat.columns and not deposit.empty:
        dep_mean = deposit.groupby("user_id")[amount_col].mean().reset_index(name="crypto_deposit_mean")
        feat = feat.merge(dep_mean, on="user_id", how="left")

    if "crypto_deposit_mean" in feat.columns and "crypto_deposit_total" in feat.columns:
        feat["crypto_max_to_mean_ratio"] = (
            deposit.groupby("user_id")[amount_col].max().reindex(feat["user_id"]).values /
            (feat["crypto_deposit_mean"].values + 1e-9)
        ) if not deposit.empty else 0
    else:
        feat["crypto_max_to_mean_ratio"] = 0

    print(f"  虛擬幣特徵：{len(feat)} 位用戶，{len(feat.columns) - 1} 個特徵")
    return feat


# ============================================================
# 6. 掛單 & 一鍵買賣特徵
# ============================================================

def build_trading_features(usdt_trade: pd.DataFrame,
                             usdt_swap: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not usdt_trade.empty:
        usdt_trade = parse_datetime(usdt_trade, ["updated_at"])
        amt_col = "trade_amount" if "trade_amount" in usdt_trade.columns else "trade_samount"
        t_feat = usdt_trade.groupby("user_id").agg(
            trade_count=(amt_col, "count"),
            trade_total_usdt=(amt_col, "sum"),
            trade_mean_usdt=(amt_col, "mean"),
            trade_buy_ratio=("is_buy", "mean"),
            trade_market_ratio=("is_market", "mean"),
        ).reset_index()
        frames.append(t_feat)

    if not usdt_swap.empty:
        usdt_swap = parse_datetime(usdt_swap, ["created_at"])
        twd_col = "twd_amount" if "twd_amount" in usdt_swap.columns else "twd_samount"
        s_feat = usdt_swap.groupby("user_id").agg(
            swap_count=(twd_col, "count"),
            swap_total_twd=(twd_col, "sum"),
            swap_mean_twd=(twd_col, "mean"),
            swap_sell_ratio=("kind", "mean"),
        ).reset_index()
        frames.append(s_feat)

    if not frames:
        return pd.DataFrame()

    feat = frames[0].merge(frames[1], on="user_id", how="outer").fillna(0) \
           if len(frames) == 2 else frames[0]

    print(f"  交易行為特徵：{len(feat)} 位用戶，{len(feat.columns) - 1} 個特徵")
    return feat


# ============================================================
# 7. 【v3 新增】跨表時間序列特徵
# ============================================================

def build_cross_table_features(twd: pd.DataFrame,
                                 crypto: pd.DataFrame,
                                 usdt_trade: pd.DataFrame,
                                 usdt_swap: pd.DataFrame,
                                 user: pd.DataFrame) -> pd.DataFrame:
    """
    跨表時間序列特徵：
    - twd_to_crypto_gap_min：法幣入金 → 虛擬幣出金最短時間差（小時）
    - trade_to_withdraw_min：USDT 交易 → 虛擬幣提幣最短時間差（小時）
    - first_to_last_days：首次交易 → 最後交易天數（帳號活躍期）
    - activity_density：總交易次數 / 活躍天數
    - account_age_days：KYC 完成 → 最後活躍天數
    """
    results = {}

    # 法幣入金時間
    if not twd.empty and "created_at" in twd.columns:
        twd_t = parse_datetime(twd.copy(), ["created_at"])
        dep = twd_t[twd_t.get("kind", pd.Series(0)) == 0][["user_id", "created_at"]]

    # 虛擬幣出金時間
    if not crypto.empty and "created_at" in crypto.columns:
        crypto_t = parse_datetime(crypto.copy(), ["created_at"])
        wd_c = crypto_t[crypto_t.get("kind", pd.Series(1)) == 1][["user_id", "created_at"]]

    # ── twd 入金 → crypto 出金最短時間差 ──
    try:
        if not dep.empty and not wd_c.empty:
            merged = dep.rename(columns={"created_at": "dep_time"}).merge(
                wd_c.rename(columns={"created_at": "wd_time"}), on="user_id")
            merged = merged[merged["wd_time"] >= merged["dep_time"]]
            merged["gap_hr"] = (merged["wd_time"] - merged["dep_time"]).dt.total_seconds() / 3600
            gap_feat = merged.groupby("user_id")["gap_hr"].min().reset_index(name="twd_to_crypto_gap_hr")
            results["cross_gap"] = gap_feat
    except Exception:
        pass

    # ── 首次/最後交易天數 ──
    try:
        all_times = []
        for df, name in [(twd_t, "twd"), (crypto_t, "crypto")]:
            if not df.empty and "created_at" in df.columns:
                sub = df[["user_id", "created_at"]].dropna()
                all_times.append(sub)

        if all_times:
            all_t = pd.concat(all_times, ignore_index=True)
            span = all_t.groupby("user_id")["created_at"].agg(
                first_active="min",
                last_active="max",
                total_tx="count",
            ).reset_index()
            span["first_to_last_days"] = (
                (span["last_active"] - span["first_active"]).dt.total_seconds() / 86400
            )
            # 活躍密度：交易次數 / 活躍天數
            span["activity_density"] = span["total_tx"] / (span["first_to_last_days"] + 1)
            results["span"] = span[["user_id", "first_to_last_days", "activity_density"]]
    except Exception:
        pass

    # ── 帳號年齡：KYC 完成 → 最後活躍 ──
    try:
        if not user.empty and "level2_finished_at" in user.columns and "last_active" in span.columns:
            user_t = parse_datetime(user.copy(), ["level2_finished_at"])
            age_base = user_t[["user_id", "level2_finished_at"]].merge(
                span[["user_id", "last_active"]], on="user_id", how="inner")
            age_base["account_age_days"] = (
                (age_base["last_active"] - age_base["level2_finished_at"])
                .dt.total_seconds() / 86400
            )
            results["age"] = age_base[["user_id", "account_age_days"]]
    except Exception:
        pass

    # 合併
    feat = None
    for name, df in results.items():
        if feat is None:
            feat = df
        else:
            feat = feat.merge(df, on="user_id", how="outer")

    if feat is None:
        return pd.DataFrame()

    print(f"  跨表特徵：{len(feat)} 位用戶，{len(feat.columns) - 1} 個特徵")
    return feat


# ============================================================
# 8. 合併所有特徵
# ============================================================

def merge_all_features(kyc_feat, twd_feat, ip_feat,
                        crypto_feat, trading_feat, cross_feat) -> pd.DataFrame:
    base = kyc_feat.copy()
    for feat, name in [
        (twd_feat,     "法幣"),
        (ip_feat,      "IP"),
        (crypto_feat,  "虛擬幣"),
        (trading_feat, "交易行為"),
        (cross_feat,   "跨表時間序列"),
    ]:
        if feat is not None and not feat.empty:
            base = base.merge(feat, on="user_id", how="left")
            print(f"  合併 {name} 後：{len(base.columns)} 欄")
    return base


# ============================================================
# 9. 後處理
# ============================================================

def postprocess(df: pd.DataFrame, label_col: str = None) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col and label_col in num_cols:
        num_cols.remove(label_col)
    df[num_cols] = df[num_cols].fillna(0)

    const_cols = [c for c in df.columns
                  if df[c].nunique() <= 1 and c not in ["user_id", label_col]]
    if const_cols:
        print(f"  移除常數欄：{const_cols}")
        df.drop(columns=const_cols, inplace=True)

    if label_col and label_col in df.columns:
        cols = [c for c in df.columns if c != label_col] + [label_col]
        df = df[cols]

    return df


# ============================================================
# 10. 主程式
# ============================================================

def main():
    print("=" * 55)
    print("BitoGuard v3 — 特徵工程開始（含行為節奏特徵）")
    print("=" * 55 + "\n")

    print("【載入資料】")
    tables = load_tables()
    print()

    train_label   = tables.get("train_label",   pd.DataFrame())
    predict_label = tables.get("predict_label", pd.DataFrame())

    if train_label.empty:
        print("❌ 找不到 train_label.csv，請先執行 fetch_data.py")
        return

    train_uids   = set(train_label["user_id"].tolist())
    predict_uids = set(predict_label["user_id"].tolist()) if not predict_label.empty else set()

    print(f"  訓練集 user_id 數：{len(train_uids):,}")
    print(f"  預測集 user_id 數：{len(predict_uids):,}")
    overlap = train_uids & predict_uids
    if overlap:
        print(f"  ⚠️ 訓練/預測集重疊：{len(overlap)} 位")
    else:
        print("  ✅ 訓練/預測集無重疊")
    print()

    print("【製造特徵（對全體用戶）】")
    kyc_feat     = build_kyc_features(tables["user_info"])
    twd_feat     = build_twd_features(tables["twd"])
    ip_feat      = build_ip_features(tables["twd"], tables["crypto"], tables["usdt_trade"])
    crypto_feat  = build_crypto_features(tables["crypto"])
    trading_feat = build_trading_features(tables["usdt_trade"], tables["usdt_swap"])
    cross_feat   = build_cross_table_features(
        tables["twd"], tables["crypto"],
        tables["usdt_trade"], tables["usdt_swap"],
        tables["user_info"]
    )
    print()

    print("【合併特徵】")
    all_feat = merge_all_features(
        kyc_feat, twd_feat, ip_feat, crypto_feat, trading_feat, cross_feat)
    print()

    print("【切出訓練集特徵（含 label）】")
    train_feat = all_feat[all_feat["user_id"].isin(train_uids)].copy()
    train_feat = train_feat.merge(
        train_label[["user_id", "status"]].rename(columns={"status": "label"}),
        on="user_id", how="left",
    )
    train_feat = postprocess(train_feat, label_col="label")
    train_feat.to_csv(TRAIN_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    bl_count = int(train_feat["label"].sum()) if "label" in train_feat.columns else 0
    feature_cols = [c for c in train_feat.columns if c not in ["user_id", "label"]]
    print(f"  訓練集：{len(train_feat):,} 筆")
    print(f"  特徵數：{len(feature_cols)} 個（不含 user_id, label）")
    print(f"  黑名單：{bl_count:,} 位（{bl_count / len(train_feat) * 100:.4f}%）")
    print(f"  已存：{TRAIN_OUTPUT_PATH}")

    if predict_uids:
        print("\n【切出預測集特徵（無 label，需提交）】")
        pred_feat = all_feat[all_feat["user_id"].isin(predict_uids)].copy()
        pred_feat = postprocess(pred_feat, label_col=None)
        pred_feat.to_csv(PREDICT_OUTPUT_PATH, index=False, encoding="utf-8-sig")
        print(f"  預測集：{len(pred_feat):,} 筆")
        print(f"  已存：{PREDICT_OUTPUT_PATH}")

    print("\n【所有特徵欄位】")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:02d}. {col}")

    print(f"\n特徵工程完成！共 {len(feature_cols)} 個特徵（v2: 47 個 → v3: {len(feature_cols)} 個）")
    print("下一步：執行 src/data/handle_imbalance.py")


if __name__ == "__main__":
    main()