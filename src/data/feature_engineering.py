"""
feature_engineering.py
位置：src/data/feature_engineering.py

【v3.5 終極提分版 — 強化行為交互與秒洗偵測】
目標：透過「複合型特徵」區分正常用戶與黑名單，預期提升 F1 至 0.35~0.42。
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
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# ============================================================
# 2. 法幣行為特徵 (強化深夜與離群偵測)
# ============================================================

def build_twd_features(twd: pd.DataFrame) -> pd.DataFrame:
    if twd.empty: return pd.DataFrame()
    twd = parse_datetime(twd, ["created_at"])
    amount_col = "ori_amount" if "ori_amount" in twd.columns else "ori_samount"
    
    deposit  = twd[twd["kind"] == 0].copy()
    withdraw = twd[twd["kind"] == 1].copy()

    dep_stats = deposit.groupby("user_id")[amount_col].agg(
        twd_deposit_count="count", twd_deposit_total="sum",
        twd_deposit_mean="mean", twd_deposit_max="max", twd_deposit_std="std",
    ).reset_index()

    wd_stats = withdraw.groupby("user_id")[amount_col].agg(
        twd_withdraw_count="count", twd_withdraw_total="sum",
        twd_withdraw_mean="mean", twd_withdraw_max="max", twd_withdraw_std="std",
    ).reset_index()

    feat = dep_stats.merge(wd_stats, on="user_id", how="outer").fillna(0)
    feat["twd_inout_ratio"] = feat["twd_withdraw_total"] / (feat["twd_deposit_total"] + 1e-9)

    # A. 快進快出分析
    if not deposit.empty and not withdraw.empty:
        dep_t = deposit[["user_id", "created_at", amount_col]].rename(columns={"created_at":"dep_t", amount_col:"dep_a"})
        wd_t  = withdraw[["user_id", "created_at", amount_col]].rename(columns={"created_at":"wd_t", amount_col:"wd_a"})
        paired = dep_t.merge(wd_t, on="user_id")
        paired = paired[paired["wd_t"] >= paired["dep_t"]]
        paired["stay_sec"] = (paired["wd_t"] - paired["dep_t"]).dt.total_seconds()
        
        stay = paired.groupby("user_id")["stay_sec"].agg(twd_stay_mean="mean", twd_stay_min="min").reset_index()
        paired["is_fast"] = (paired["stay_sec"] < 3600).astype(int)
        fast_ratio = paired.groupby("user_id")["is_fast"].mean().reset_index(name="twd_fast_exit_ratio")
        
        feat = feat.merge(stay, on="user_id", how="left").merge(fast_ratio, on="user_id", how="left")
    
    # B. 深夜交易與深夜大額交互
    twd["hour"] = twd["created_at"].dt.hour
    twd["is_night"] = twd["hour"].between(23, 23) | twd["hour"].between(0, 5)
    night_stats = twd.groupby("user_id").agg(
        twd_night_ratio=("is_night", "mean"),
        twd_night_max_amt=(amount_col, lambda x: x[twd.loc[x.index, "is_night"]].max() if any(twd.loc[x.index, "is_night"]) else 0)
    ).reset_index()
    # 交互：深夜交易比例 * 深夜最大金額 (抓出深夜出沒的大戶)
    night_stats["feat_night_whale_score"] = night_stats["twd_night_ratio"] * night_stats["twd_night_max_amt"]
    feat = feat.merge(night_stats, on="user_id", how="left")

    # C. 集中度與爆量
    twd["date"] = twd["created_at"].dt.date
    daily = twd.groupby(["user_id", "date"]).size().reset_index(name="cnt")
    feat = feat.merge(daily.groupby("user_id")["date"].nunique().reset_index(name="twd_active_days"), on="user_id", how="left")
    feat = feat.merge(daily.groupby("user_id")["cnt"].apply(lambda x: x.nlargest(3).sum()/(x.sum()+1e-9)).reset_index(name="twd_burst_ratio"), on="user_id", how="left")

    # D. 金額整數比率
    round_r = twd.groupby("user_id").apply(lambda x: ((x[amount_col]%10000==0) & (x[amount_col]>0)).mean()).reset_index(name="twd_round_amount_ratio")
    feat = feat.merge(round_r, on="user_id", how="left")

    print(f"  法幣特徵：{len(feat)} 位用戶")
    return feat

# ============================================================
# 3. 其他基礎特徵 (KYC, IP, Crypto, Trading)
# ============================================================

def build_kyc_features(user: pd.DataFrame) -> pd.DataFrame:
    if user.empty: return pd.DataFrame()
    user = parse_datetime(user, ["confirmed_at", "level1_finished_at", "level2_finished_at"])
    feat = user[["user_id"]].copy()
    feat["kyc_level"] = 0
    feat.loc[user["confirmed_at"].notna(), "kyc_level"] = 1
    feat.loc[user["level2_finished_at"].notna(), "kyc_level"] = 2
    if "age" in user.columns:
        feat["age"] = pd.to_numeric(user["age"], errors="coerce").fillna(0)
    if "career" in user.columns:
        feat["career_code"] = user["career"].fillna(0).astype(int)
    return feat

def build_ip_features(twd, crypto, trade) -> pd.DataFrame:
    ip_col = "source_ip_hash"
    dfs = [df[["user_id", ip_col]] for df in [twd, crypto, trade] if not df.empty and ip_col in df.columns]
    if not dfs: return pd.DataFrame()
    all_ip = pd.concat(dfs).dropna()
    feat = all_ip.groupby("user_id")[ip_col].agg(ip_unique_count="nunique", ip_total_count="count").reset_index()
    feat["ip_diversity_ratio"] = feat["ip_unique_count"] / (feat["ip_total_count"] + 1e-9)
    return feat

def build_crypto_features(crypto: pd.DataFrame) -> pd.DataFrame:
    if crypto.empty: return pd.DataFrame()
    crypto = parse_datetime(crypto, ["created_at"])
    amt_col = "ori_amount" if "ori_amount" in crypto.columns else "ori_samount"
    feat = crypto.groupby("user_id").agg(
        crypto_total_count=("user_id", "count"),
        crypto_withdraw_total=(amt_col, lambda x: x[crypto.loc[x.index, "kind"]==1].sum() if "kind" in crypto.columns else 0),
        crypto_deposit_total=(amt_col, lambda x: x[crypto.loc[x.index, "kind"]==0].sum() if "kind" in crypto.columns else 0),
        crypto_night_ratio=("created_at", lambda x: x.dt.hour.isin([23,0,1,2,3,4,5]).mean())
    ).reset_index()
    feat["crypto_inout_ratio"] = feat["crypto_withdraw_total"] / (feat["crypto_deposit_total"] + 1e-9)
    return feat

def build_trading_features(usdt_trade, usdt_swap) -> pd.DataFrame:
    frames = []
    if not usdt_trade.empty:
        frames.append(usdt_trade.groupby("user_id").agg(trade_count=("user_id", "count"), trade_buy_ratio=("is_buy", "mean")).reset_index())
    if not usdt_swap.empty:
        frames.append(usdt_swap.groupby("user_id").agg(swap_count=("user_id", "count")).reset_index())
    if not frames: return pd.DataFrame()
    feat = frames[0]
    for f in frames[1:]: feat = feat.merge(f, on="user_id", how="outer")
    return feat.fillna(0)

# ============================================================
# 4. 【v3.5 關鍵核心】跨表交互特徵 (秒洗、低KYC高流量、爆發強度)
# ============================================================

def build_cross_table_features(twd, crypto, user, twd_feat) -> pd.DataFrame:
    results = []
    
    # A. 跨表時間差 (TWD入 -> Crypto出)
    if not twd.empty and not crypto.empty:
        twd_t = parse_datetime(twd.copy(), ["created_at"])
        cry_t = parse_datetime(crypto.copy(), ["created_at"])
        dep = twd_t[twd_t["kind"]==0][["user_id", "created_at"]].rename(columns={"created_at":"t_in"})
        wdc = cry_t[cry_t["kind"]==1][["user_id", "created_at"]].rename(columns={"created_at":"t_out"})
        merged = dep.merge(wdc, on="user_id")
        merged = merged[merged["t_out"] >= merged["t_in"]]
        merged["gap"] = (merged["t_out"] - merged["t_in"]).dt.total_seconds() / 3600
        cross_gap = merged.groupby("user_id")["gap"].min().reset_index(name="twd_to_crypto_gap_hr")
        results.append(cross_gap)

    # B. 帳號生命週期與活躍密度
    all_t = pd.concat([twd[["user_id", "created_at"]], crypto[["user_id", "created_at"]]]).dropna()
    all_t["created_at"] = pd.to_datetime(all_t["created_at"])
    span = all_t.groupby("user_id")["created_at"].agg(["min", "max", "count"]).reset_index()
    span["first_to_last_days"] = (span["max"] - span["min"]).dt.total_seconds() / 86400
    span["activity_density"] = span["count"] / (span["first_to_last_days"] + 1)
    results.append(span[["user_id", "first_to_last_days", "activity_density"]])

    # C. 合併初步結果
    feat = results[0]
    for r in results[1:]: feat = feat.merge(r, on="user_id", how="outer")
    
    # D. 注入高級交互特徵
    # 交互1: 秒洗風險 (滯留時間短且快進快出比例高)
    if 'twd_stay_min' in twd_feat.columns and 'twd_fast_exit_ratio' in twd_feat.columns:
        feat = feat.merge(twd_feat[['user_id', 'twd_stay_min', 'twd_fast_exit_ratio']], on='user_id', how='left')
        feat['feat_instant_wash_risk'] = ((feat['twd_stay_min'] < 300) & (feat['twd_fast_exit_ratio'] > 0.8)).astype(int)
        feat.drop(columns=['twd_stay_min', 'twd_fast_exit_ratio'], inplace=True)

    # 交互2: 低 KYC 卻有高額入金 (極強的人頭戶特徵)
    user_kyc = build_kyc_features(user)
    if not user_kyc.empty and 'twd_deposit_total' in twd_feat.columns:
        tmp = twd_feat[['user_id', 'twd_deposit_total']].merge(user_kyc[['user_id', 'kyc_level']], on='user_id')
        tmp['feat_low_kyc_high_vol'] = ((tmp['kyc_level'] < 2) & (tmp['twd_deposit_total'] > 200000)).astype(int)
        feat = feat.merge(tmp[['user_id', 'feat_low_kyc_high_vol']], on='user_id', how='left')

    # 交互3: 爆發強度 (密集度 * 爆量比)
    if 'twd_burst_ratio' in twd_feat.columns:
        tmp = twd_feat[['user_id', 'twd_burst_ratio']].merge(feat[['user_id', 'activity_density']], on='user_id')
        tmp['feat_burst_intensity'] = tmp['activity_density'] * tmp['twd_burst_ratio']
        feat = feat.merge(tmp[['user_id', 'feat_burst_intensity']], on='user_id', how='left')

    return feat.fillna(0)

# ============================================================
# 5. 主流程
# ============================================================

def main():
    print("=" * 55)
    print("BitoGuard v3.5 — 提分大師：進階交互特徵版")
    print("=" * 55)

    tables = load_tables()
    train_label = tables.get("train_label", pd.DataFrame())
    predict_label = tables.get("predict_label", pd.DataFrame())

    # 建立各表特徵
    kyc_f = build_kyc_features(tables["user_info"])
    twd_f = build_twd_features(tables["twd"])
    ip_f  = build_ip_features(tables["twd"], tables["crypto"], tables["usdt_trade"])
    cry_f = build_crypto_features(tables["crypto"])
    td_f  = build_trading_features(tables["usdt_trade"], tables["usdt_swap"])
    
    # 跨表高級交互 (需依賴 twd_f)
    cross_f = build_cross_table_features(tables["twd"], tables["crypto"], tables["user_info"], twd_f)

    # 合併
    all_feat = kyc_f.merge(twd_f, on="user_id", how="left") \
                    .merge(ip_f,  on="user_id", how="left") \
                    .merge(cry_f, on="user_id", how="left") \
                    .merge(td_f,  on="user_id", how="left") \
                    .merge(cross_f, on="user_id", how="left")
    
    # 填充空值與過濾
    num_cols = all_feat.select_dtypes(include=[np.number]).columns
    all_feat[num_cols] = all_feat[num_cols].fillna(0)

    # 分離訓練與預測
    train_feat = all_feat[all_feat["user_id"].isin(train_label["user_id"])].copy()
    train_feat = train_feat.merge(train_label[["user_id", "status"]].rename(columns={"status":"label"}), on="user_id")
    train_feat.to_csv(TRAIN_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    pred_feat = all_feat[all_feat["user_id"].isin(predict_label["user_id"])].copy()
    pred_feat.to_csv(PREDICT_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n✅ 特徵工程完成！")
    print(f"   總特徵數：{len(train_feat.columns)-2}")
    print(f"   已存檔：{TRAIN_OUTPUT_PATH}")
    print("=" * 55)

if __name__ == "__main__":
    main()