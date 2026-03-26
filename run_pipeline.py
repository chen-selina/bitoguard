"""
run_pipeline.py
位置：bitoguard/（根目錄）

功能：一鍵依序執行完整 BitoGuard Pipeline，
      並在每個步驟前後印出進度與耗時。

使用方式：python run_pipeline.py
選項：
  --skip-fetch    跳過 API 拉取（已有 data/raw/ 時使用）
  --skip-graph    跳過關聯圖譜（節省時間）
  --only-dashboard 只啟動儀表板

範例：
  python run_pipeline.py                    # 完整執行
  python run_pipeline.py --skip-fetch       # 跳過拉資料
  python run_pipeline.py --only-dashboard   # 只開儀表板
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================
# Pipeline 步驟定義
# ============================================================
STEPS = [
    {
        "name":    "① 資料拉取",
        "script":  "src/data/fetch_data.py",
        "flag":    "fetch",
        "desc":    "從 BitoPro API 拉取五張資料表",
        "skippable": True,
    },
    {
        "name":    "② 特徵工程",
        "script":  "src/data/feature_engineering.py",
        "flag":    "feature",
        "desc":    "合併資料表，製造 ~35 個特徵",
        "skippable": False,
    },
    {
        "name":    "③ 不平衡處理",
        "script":  "src/data/handle_imbalance.py",
        "flag":    "imbalance",
        "desc":    "SMOTE + 切分訓練/測試集",
        "skippable": False,
    },
    {
        "name":    "④ 模型訓練",
        "script":  "src/models/train_model.py",
        "flag":    "train",
        "desc":    "XGBoost / LightGBM / Isolation Forest + Ensemble",
        "skippable": False,
    },
    {
        "name":    "⑤ SHAP 解釋",
        "script":  "src/models/shap_explainer.py",
        "flag":    "shap",
        "desc":    "生成個人風險診斷書與特徵重要性圖",
        "skippable": False,
    },
    {
        "name":    "⑥ 關聯圖譜",
        "script":  "src/models/graph_analysis.py",
        "flag":    "graph",
        "desc":    "NetworkX 關聯圖 + 圖特徵合併",
        "skippable": True,
    },
]


# ============================================================
# 工具函式
# ============================================================

def print_banner():
    print("\n" + "═" * 60)
    print("  🛡️  BitoGuard — AI 防詐守門員")
    print("      智慧合規風險雷達 Pipeline")
    print("═" * 60)
    print(f"  開始時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60 + "\n")


def print_step(step: dict, index: int, total: int):
    print(f"\n{'─' * 60}")
    print(f"  [{index}/{total}] {step['name']}")
    print(f"  {step['desc']}")
    print(f"  執行：python {step['script']}")
    print(f"{'─' * 60}")


def run_script(script_path: str) -> tuple[bool, float]:
    """執行指定 Python 腳本，回傳 (成功與否, 耗時秒數)"""
    path = Path(script_path)
    if not path.exists():
        print(f"  ❌ 找不到 {script_path}")
        return False, 0.0

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✅ 完成（耗時 {elapsed:.1f} 秒）")
        return True, elapsed
    else:
        print(f"\n  ❌ 執行失敗（returncode={result.returncode}）")
        return False, elapsed


def launch_dashboard():
    """啟動 Streamlit 儀表板"""
    dashboard_path = Path("app/dashboard/dashboard.py")
    if not dashboard_path.exists():
        print("❌ 找不到 dashboard.py")
        return

    print("\n" + "═" * 60)
    print("  🚀 啟動 BitoGuard 儀表板")
    print("  瀏覽器開啟：http://localhost:8501")
    print("  按 Ctrl+C 停止")
    print("═" * 60 + "\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false",
    ])


# ============================================================
# 主程式
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BitoGuard Pipeline")
    parser.add_argument("--skip-fetch",   action="store_true",
                        help="跳過資料拉取步驟（已有 data/raw/ 時使用）")
    parser.add_argument("--skip-graph",   action="store_true",
                        help="跳過關聯圖譜分析（節省時間）")
    parser.add_argument("--only-dashboard", action="store_true",
                        help="跳過所有步驟，直接啟動儀表板")
    args = parser.parse_args()

    print_banner()

    if args.only_dashboard:
        launch_dashboard()
        return

    # 決定哪些步驟要跑
    skip_flags = set()
    if args.skip_fetch:
        skip_flags.add("fetch")
        print("  ⏭  跳過資料拉取（--skip-fetch）")
    if args.skip_graph:
        skip_flags.add("graph")
        print("  ⏭  跳過關聯圖譜（--skip-graph）")

    steps_to_run = [s for s in STEPS if s["flag"] not in skip_flags]
    total        = len(steps_to_run)

    results = []
    total_start = time.time()

    for i, step in enumerate(steps_to_run, 1):
        print_step(step, i, total)
        success, elapsed = run_script(step["script"])
        results.append({
            "step":    step["name"],
            "success": success,
            "elapsed": elapsed,
        })

        if not success:
            print(f"\n  ⚠️  步驟 {step['name']} 失敗，Pipeline 中止")
            print("  請檢查上方錯誤訊息後重新執行\n")
            break

    total_elapsed = time.time() - total_start

    # 執行摘要
    print("\n" + "═" * 60)
    print("  Pipeline 執行摘要")
    print("═" * 60)
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status}  {r['step']:<20}  {r['elapsed']:.1f} 秒")
    print(f"\n  總耗時：{total_elapsed:.1f} 秒")
    print(f"  結束時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_success = all(r["success"] for r in results)
    if all_success:
        print("\n  🎉 Pipeline 全部完成！")
        print("═" * 60)

        # 詢問是否啟動儀表板
        try:
            ans = input("\n  是否立即啟動儀表板？(y/n) > ").strip().lower()
            if ans == "y":
                launch_dashboard()
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        print("\n  Pipeline 未完全成功，請修正錯誤後重新執行")
        print("═" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()