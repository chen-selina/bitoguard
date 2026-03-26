import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# 忽略警告
warnings.filterwarnings("ignore")

# ============================================================
# 1. 頁面與全域樣式設定
# ============================================================
st.set_page_config(
    page_title="BitoGuard — 智慧合規風險雷達",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+TC:wght@400;500;700&display=swap');
html, body, [class*="css"] { 
    font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif; 
}

/* 頂部 Header */
.bito-header {
    background: linear-gradient(135deg, #0a1020 0%, #0d1f3c 100%);
    border-bottom: 1px solid #1e3a5f;
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 1.5rem;
    display: flex; justify-content: space-between; align-items: center;
}
.bito-header h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; color: #e8f4ff; margin: 0; }

/* KPI 卡片樣式 */
.kpi-card {
    background: #0d1826; border: 1px solid #1e3354; border-radius: 10px; padding: 1.2rem;
    position: relative; overflow: hidden;
}
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.kpi-card.red::before   { background: #e24b4a; }
.kpi-card.blue::before  { background: #378add; }
.kpi-card.green::before { background: #3ba87d; }
.kpi-label { font-size: 0.75rem; color: #4a7aaa; text-transform: uppercase; margin-bottom: 5px; }
.kpi-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 600; color: #e8f4ff; }

/* 診斷書元件 */
.diagnosis-card { background: #080f1e; border: 1px solid #1a2e4a; border-radius: 12px; padding: 1.5rem; }
.factor-row { display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid #111c2e; }
.factor-name { font-size: 0.85rem; color: #a8c8e8; flex: 1; }
.factor-bar-wrap { width: 120px; height: 8px; background: #111c2e; border-radius: 4px; }
.factor-bar { height: 100%; border-radius: 4px; }
.bar-up { background: #e24b4a; }
.bar-down { background: #3ba87d; }
.nl-summary { 
    margin-top: 1rem; 
    padding: 1rem; 
    background: #0a1420; 
    border-left: 3px solid #378add; 
    color: #c8d8e8; 
    line-height: 1.6;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. 自動路徑定位 (解決 Errno 2 問題)
# ============================================================
# 從 app/dashboard/dashboard.py 往上兩層到專案根目錄 bitoguard
BASE_DIR = Path(__file__).resolve().parents[2] 
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

@st.cache_data
def load_all_data():
    # 使用實際存在的檔案路徑
    # 預測結果可能在 submission.csv 或 user_risk_scores.csv
    pred_path = REPORTS_DIR / "user_risk_scores.csv"
    if not pred_path.exists():
        pred_path = REPORTS_DIR / "submission.csv"
    
    diag_path = REPORTS_DIR / "risk_diagnosis.json"
    feat_path = PROCESSED_DIR / "features.csv"
    
    try:
        # 讀取預測結果（包含 user_id 和 risk_score 或 status）
        pred = pd.read_csv(pred_path)
        
        # 如果是 submission.csv，欄位名稱可能是 status，需要轉換為 risk_score
        if "status" in pred.columns and "risk_score" not in pred.columns:
            pred["risk_score"] = pred["status"].astype(float)
        
        # 讀取特徵檔案
        feat = pd.read_csv(feat_path, low_memory=False)
        
        # 讀取診斷報告
        with open(diag_path, "r", encoding="utf-8") as f:
            diag = json.load(f)
        
        return pred, feat, diag, pred_path.name
    except Exception as e:
        st.error(f"資料載入失敗，請確認路徑：\n{pred_path}\n錯誤：{e}")
        st.stop()

pred_df, feat_df, reports, model_ver = load_all_data()
# 合併風險分數與原始特徵資料
data = pred_df.merge(feat_df, on="user_id", how="left")
report_map = {r["user_id"]: r for r in reports}

# ============================================================
# 3. Sidebar 與 Header
# ============================================================
st.markdown(f"""
<div class="bito-header">
  <div><h1>🛡️ BitoGuard</h1><div style="color:#5a8fcf; font-size:0.8rem;">智慧合規風險雷達 · 2026 去偽存真</div></div>
  <div style="text-align:right;"><span style="color:#5ab8ff; font-family:monospace;">{model_ver}</span></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🗂 導覽選單")
    page = st.radio("功能", ["📊 風險總覽", "🔍 用戶診斷書", "📈 模型效能", "🕸️ 關聯圖譜"], label_visibility="collapsed")
    st.divider()
    threshold = st.slider("風險閾值設定", 0.1, 0.9, 0.5, 0.05)

# ============================================================
# 4. 分頁實作
# ============================================================
if page == "📊 風險總覽":
    total = len(data)
    flagged = (data["risk_score"] >= threshold).sum()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card blue"><div class="kpi-label">分析總人數</div><div class="kpi-value">{total:,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card red"><div class="kpi-label">高風險標記</div><div class="kpi-value">{flagged:,}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card blue"><div class="kpi-label">平均風險值</div><div class="kpi-value">{data["risk_score"].mean():.3f}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card green"><div class="kpi-label">系統狀態</div><div class="kpi-value" style="font-size:1.2rem;">ONLINE</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader("風險分數分佈圖")
        fig = px.histogram(data, x="risk_score", nbins=50, template="plotly_dark", color_discrete_sequence=["#378add"])
        fig.add_vline(x=threshold, line_dash="dash", line_color="orange")
        # 設定中文字型
        fig.update_layout(
            font=dict(family="Microsoft JhengHei, Noto Sans TC, sans-serif"),
            xaxis_title="風險分數",
            yaxis_title="用戶數量"
        )
        st.plotly_chart(fig, width="stretch")
    with col_r:
        st.subheader("風險用戶清單")
        st.dataframe(data.sort_values("risk_score", ascending=False).head(10)[["user_id", "risk_score"]], width="stretch")

elif page == "🔍 用戶診斷書":
    st.subheader("🔍 AI 風險診斷報告")
    selected_uid = st.selectbox("選擇用戶 ID", options=[r["user_id"] for r in reports])
    report = report_map[selected_uid]
    
    st.markdown(f"""
    <div class="diagnosis-card">
        <div style="display:flex; justify-content:space-between; border-bottom:1px solid #1e3a5f; padding-bottom:10px;">
            <div style="font-size:1.2rem; color:#a8c8e8;">用戶 <b>#{selected_uid}</b></div>
            <div style="font-size:2rem; font-weight:600; color:{'#e24b4a' if report['risk_score'] > 0.5 else '#3ba87d'}">{report['risk_score']:.2%}</div>
        </div>
        <div class="nl-summary">{report.get('nl_summary', '尚無診斷描述')}</div>
    """, unsafe_allow_html=True)

    for f in report.get("top_factors", []):
        pct = (abs(f["shap"]) / 0.5) * 100 # 假設 0.5 為比例基準
        clr = "bar-up" if f["shap"] > 0 else "bar-down"
        st.markdown(f"""
        <div class="factor-row">
            <div class="factor-name">{f['label']}</div>
            <div class="factor-bar-wrap"><div class="factor-bar {clr}" style="width:{min(pct, 100)}%"></div></div>
            <div style="color:{'#ff8787' if f['shap']>0 else '#69db7c'}">{f['shap']:+.3f}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "🕸️ 關聯圖譜":
    st.subheader("🕸️ 資金流向分析圖")
    plots_dir = BASE_DIR / "outputs" / "plots"
    graph_img = plots_dir / "graph_full_network.png"
    if graph_img.exists():
        st.image(str(graph_img), width="stretch")
    else:
        st.info("尚未生成圖譜影像檔案。")

elif page == "📈 模型效能":
    st.subheader("📈 模型評估指標")
    plots_dir = BASE_DIR / "outputs" / "plots"
    if (plots_dir / "shap_importance.png").exists():
        st.image(str(plots_dir / "shap_importance.png"), caption="SHAP 特徵重要性")