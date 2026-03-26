"""
fetch_data.py
位置：src/data/fetch_data.py

【v3 修正說明 — 根據 Swagger 文件確認正確 API 路徑】
  - API 是 PostgREST，路徑直接是資料表名稱（無 /api/ 前綴）
  - 分頁參數改為 PostgREST 格式：offset + limit（非 page + limit）
  - 需在 Header 加入 Prefer: count=exact 取得總筆數
  - 路徑對照：
      /api/users          → /user_info
      /api/twd-transfers  → /twd_transfer
      /api/crypto-transfers → /crypto_transfer
      /api/usdt-twd-trading → /usdt_twd_trading
      /api/usdt-swap      → /usdt_swap
      /api/train-label    → /train_label
      /api/predict-label  → /predict_label

使用方式：python src/data/fetch_data.py
"""

import time
import requests
import pandas as pd
from pathlib import Path

# ============================================================
# 設定區
# ============================================================
API_BASE_URL = "https://aws-event-api.bitopro.com"

# PostgREST 標準 Header
# Prefer: count=exact  → 回傳 Content-Range header，可得知總筆數
HEADERS = {
    "Content-Type": "application/json",
    "Prefer": "count=exact",
}

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 【v3 修正】API 路徑：直接用資料表名稱，無 /api/ 前綴
# ============================================================
ENDPOINTS = {
    "user_info":        "/user_info",
    "twd_transfer":     "/twd_transfer",
    "crypto_transfer":  "/crypto_transfer",
    "usdt_twd_trading": "/usdt_twd_trading",
    "usdt_swap":        "/usdt_swap",
    "train_label":      "/train_label",
    "predict_label":    "/predict_label",
}

# 預期筆數（驗證用）
EXPECTED_COUNTS = {
    "user_info":        63770,
    "twd_transfer":     195601,
    "crypto_transfer":  239958,
    "usdt_twd_trading": 217634,
    "usdt_swap":        53841,
    "train_label":      51017,
    "predict_label":    12753,
}

# PostgREST 每次拉取筆數上限
PAGE_SIZE = 1000


# ============================================================
# 1. 分頁拉取（PostgREST 格式）
# PostgREST 分頁用 offset + limit，不用 page
# ============================================================

def fetch_table(table_name: str, endpoint: str) -> pd.DataFrame:
    all_records = []
    offset      = 0
    total       = None   # 從 Content-Range header 取得

    print(f"  正在拉取 {table_name}（預期 {EXPECTED_COUNTS.get(table_name, '?'):,} 筆）...")

    while True:
        url = f"{API_BASE_URL}{endpoint}"

        # PostgREST 分頁參數
        params = {
            "offset": offset,
            "limit":  PAGE_SIZE,
            "order":  "id"   # 確保每次排序一致（部分資料表有 id 欄）
        }

        # predict_label 和 train_label 沒有 id 欄，改用 user_id 排序
        if table_name in ("train_label", "predict_label"):
            params["order"] = "user_id"

        try:
            response = requests.get(
                url, headers=HEADERS, params=params, timeout=60
            )
            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            # 嘗試移除 order 參數重試（某些資料表可能不支援排序）
            if "order" in params:
                params.pop("order")
                try:
                    response = requests.get(
                        url, headers=HEADERS, params=params, timeout=60
                    )
                    response.raise_for_status()
                except Exception:
                    print(f"    HTTP 錯誤 (offset {offset}): {e}")
                    break
            else:
                print(f"    HTTP 錯誤 (offset {offset}): {e}")
                break

        except requests.exceptions.ConnectionError:
            print(f"    連線失敗，請確認 API 可否存取")
            break
        except requests.exceptions.Timeout:
            print(f"    逾時，重試 (offset {offset})...")
            time.sleep(3)
            continue

        # 從 Content-Range header 取得總筆數
        # 格式：0-999/63770
        if total is None:
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                try:
                    total = int(content_range.split("/")[1])
                    print(f"    總筆數（Content-Range）：{total:,}")
                except ValueError:
                    pass

        records = response.json()

        if not isinstance(records, list) or len(records) == 0:
            break

        all_records.extend(records)
        current_batch = len(records)
        print(f"    offset={offset:>7,}：取得 {current_batch} 筆（累計 {len(all_records):,} 筆）")

        # 若這批筆數小於 PAGE_SIZE，代表已是最後一頁
        if current_batch < PAGE_SIZE:
            break

        # 若已達預期總筆數，結束
        if total and len(all_records) >= total:
            break

        offset += PAGE_SIZE
        time.sleep(0.15)   # 避免打爆 API

    if not all_records:
        print(f"    警告：{table_name} 沒有取得任何資料")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    print(f"    完成：{len(df):,} 筆，欄位：{list(df.columns)}")

    # 驗證筆數
    expected = EXPECTED_COUNTS.get(table_name)
    if expected:
        diff = abs(len(df) - expected)
        if diff > expected * 0.02:   # 允許 2% 誤差
            print(f"    ⚠️  筆數異常！預期 {expected:,}，實際 {len(df):,}（差 {diff:,}）")
        else:
            print(f"    ✅ 筆數驗證通過（{len(df):,} 筆）")

    return df


# ============================================================
# 2. 金額欄位換算（乘以 1e-8）
# ============================================================

def fix_amount_columns(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    amount_cols_map = {
        "twd_transfer":     ["ori_samount"],
        "crypto_transfer":  ["ori_samount", "twd_srate"],
        "usdt_twd_trading": ["trade_samount", "twd_srate"],
        "usdt_swap":        ["twd_samount", "currency_samount"],
    }
    cols_to_fix = amount_cols_map.get(table_name, [])
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * 1e-8
            df.rename(columns={col: col.replace("_s", "_")}, inplace=True)
    return df


# ============================================================
# 3. 存檔
# ============================================================

def save_csv(df: pd.DataFrame, table_name: str) -> None:
    if df.empty:
        print(f"  跳過存檔：{table_name} 無資料")
        return
    path = RAW_DATA_DIR / f"{table_name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  已存檔：{path}  （{len(df):,} 筆，{len(df.columns)} 欄）")


# ============================================================
# 4. 分析黑名單比例
# ============================================================

def analyze_train_label(df_train_label: pd.DataFrame) -> None:
    if df_train_label.empty or "status" not in df_train_label.columns:
        return

    total = len(df_train_label)
    bl    = (df_train_label["status"] == 1).sum()
    ratio = bl / total * 100

    print("\n" + "=" * 55)
    print("黑名單比例分析（train_label）")
    print("=" * 55)
    print(f"  訓練集用戶數：{total:,}")
    print(f"  黑名單數    ：{bl:,}")
    print(f"  黑名單比率  ：{ratio:.4f}%")
    print(f"  正常:黑名單 ≈ {total - bl:,} : {bl:,}")
    if ratio < 1:
        print("  ⚠️  極度不平衡（< 1%），SMOTE 非常重要")
    elif ratio < 5:
        print("  ⚠️  嚴重不平衡（< 5%），需要 SMOTE")
    else:
        print("  資料比例尚可")
    print("=" * 55)


def show_predict_label_info(df_predict: pd.DataFrame) -> None:
    if df_predict.empty:
        return
    print("\n" + "=" * 55)
    print("預測提交名單（predict_label）")
    print("=" * 55)
    print(f"  需預測用戶數：{len(df_predict):,}")
    print(f"  欄位：{list(df_predict.columns)}")
    print("  ⚠️  這是最終需要填入預測結果並提交的名單")
    print("  提交格式：user_id + status（0 或 1）")
    print("=" * 55)


# ============================================================
# 5. 主程式
# ============================================================

def main():
    print("=" * 55)
    print("BitoGuard v3 — 資料拉取開始（PostgREST）")
    print(f"API：{API_BASE_URL}")
    print(f"存放位置：{RAW_DATA_DIR.resolve()}")
    print("=" * 55 + "\n")

    fetched = {}

    for table_name, endpoint in ENDPOINTS.items():
        print(f"【{table_name}】")
        df = fetch_table(table_name, endpoint)

        if not df.empty:
            df = fix_amount_columns(df, table_name)

        save_csv(df, table_name)
        fetched[table_name] = df
        print()

    if "train_label" in fetched:
        analyze_train_label(fetched["train_label"])

    if "predict_label" in fetched:
        show_predict_label_info(fetched["predict_label"])

    # 確認所有資料表都拿到了
    print("\n" + "=" * 55)
    print("拉取結果總覽")
    print("=" * 55)
    all_ok = True
    for name, df in fetched.items():
        status = "✅" if not df.empty else "❌"
        count  = f"{len(df):,}" if not df.empty else "0"
        print(f"  {status}  {name:<20} {count} 筆")
        if df.empty:
            all_ok = False

    if all_ok:
        print("\n✅ 所有資料表拉取成功！")
        print("下一步：執行 src/data/feature_engineering.py")
    else:
        print("\n⚠️  部分資料表拉取失敗，請確認 API 連線或端點路徑")


if __name__ == "__main__":
    main()