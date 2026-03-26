"""
fix_submission.py
修正 submission.csv 格式，只保留 user_id 和 status 兩個欄位

使用方式：python fix_submission.py
"""

import pandas as pd
from pathlib import Path

SUBMISSION_PATH = Path("outputs/reports/submission.csv")
SUBMISSION_BACKUP = Path("outputs/reports/submission_backup.csv")
SUBMISSION_NEW = Path("outputs/reports/submission_final.csv")

def main():
    print("=" * 60)
    print("修正 submission.csv 格式")
    print("=" * 60 + "\n")
    
    if not SUBMISSION_PATH.exists():
        print(f"❌ 找不到檔案：{SUBMISSION_PATH}")
        return
    
    # 讀取原始檔案
    df = pd.read_csv(SUBMISSION_PATH)
    print(f"原始檔案欄位：{list(df.columns)}")
    print(f"總筆數：{len(df):,}\n")
    
    # 備份原始檔案
    try:
        df.to_csv(SUBMISSION_BACKUP, index=False, encoding="utf-8-sig")
        print(f"✅ 已備份至：{SUBMISSION_BACKUP}\n")
    except:
        print(f"⚠️  無法備份（檔案可能被開啟）\n")
    
    # 只保留 user_id 和 status
    if "user_id" not in df.columns or "status" not in df.columns:
        print("❌ 檔案缺少必要欄位（user_id 或 status）")
        return
    
    df_clean = df[["user_id", "status"]].copy()
    
    # 嘗試覆蓋原檔案，若失敗則存成新檔案
    try:
        df_clean.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
        print(f"✅ {SUBMISSION_PATH.name} 已更新")
    except PermissionError:
        df_clean.to_csv(SUBMISSION_NEW, index=False, encoding="utf-8-sig")
        print(f"⚠️  原檔案被鎖定，已生成新檔案：{SUBMISSION_NEW}")
        print(f"   請關閉 Excel 後，將 {SUBMISSION_NEW.name} 重新命名為 submission.csv")
    
    print(f"   欄位：{list(df_clean.columns)}")
    print(f"   總筆數：{len(df_clean):,}")
    print(f"\n前 10 筆預測結果：")
    print(df_clean.head(10).to_string(index=False))
    
    # 統計
    normal = (df_clean["status"] == 0).sum()
    blacklist = (df_clean["status"] == 1).sum()
    print(f"\n預測統計：")
    print(f"  正常用戶：{normal:,} 位 ({normal/len(df_clean)*100:.2f}%)")
    print(f"  黑名單  ：{blacklist:,} 位 ({blacklist/len(df_clean)*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
