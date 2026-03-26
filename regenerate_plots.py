"""
regenerate_plots.py
重新生成所有圖表，確保中文字型正確顯示

使用方式：python regenerate_plots.py
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path: str, description: str) -> bool:
    """執行指定腳本"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  執行：{script_path}")
    print(f"{'='*60}")
    
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ {description} 完成")
        return True
    else:
        print(f"❌ {description} 失敗")
        return False

def main():
    print("\n" + "="*60)
    print("  🎨 重新生成所有圖表（含中文字型修正）")
    print("="*60)
    
    # 只重新執行會產生圖表的步驟
    steps = [
        ("src/models/graph_analysis.py", "關聯圖譜分析"),
        ("src/models/shap_explainer.py", "SHAP 解釋性分析"),
    ]
    
    for script, desc in steps:
        if not Path(script).exists():
            print(f"⚠️  找不到 {script}，跳過")
            continue
        
        success = run_script(script, desc)
        if not success:
            print(f"\n⚠️  {desc} 執行失敗，請檢查錯誤訊息")
            return
    
    print("\n" + "="*60)
    print("  ✅ 所有圖表已重新生成！")
    print("  圖表位置：outputs/plots/")
    print("  現在可以重新整理 dashboard 查看結果")
    print("="*60)

if __name__ == "__main__":
    main()
