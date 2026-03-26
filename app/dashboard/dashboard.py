"""
dashboard.py
位置：app/dashboard/dashboard.py

功能：BitoGuard 智慧合規風險雷達 — Streamlit 互動儀表板
      整合所有模型輸出，提供：
      - 風險總覽儀表板
      - 高風險用戶排行榜
      - 個人風險診斷書（含 SHAP 瀑布圖）
      - 關聯圖譜展示
      - 模型效能報告

使用方式：streamlit run app/dashboard/dashboard.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# 頁面設定（必須是第一個 Streamlit 呼叫）
# ============================================================
st.set_page_config(
    page_title="BitoGuard — 智慧合規風險雷達",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 全局樣式
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+TC:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans TC', sans-serif;
}

/* 頂部 Header */
.bito-header {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1f3c 60%, #112244 100%);
    border-bottom: 1px solid #1e3a5f;
    padding: 1.4rem 2rem 1.2rem;
    margin: -1rem -1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.bito-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #e8f4ff;
    margin: 0;
    letter-spacing: 0.03em;
}
.bito-header .subtitle {
    font-size: 0.78rem;
    color: #5a8fcf;
    margin-top: 0.2rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.bito-badge {
    background: #1a3a6b;
    border: 1px solid #2a5a9f;
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #5ab8ff;
    letter-spacing: 0.08em;
}

/* KPI 卡片 */
.kpi-card {
    background: #0d1826;
    border: 1px solid #1e3354;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.red::before   { background: #e24b4a; }
.kpi-card.amber::before { background: #ef9f27; }
.kpi-card.blue::before  { background: #378add; }
.kpi-card.green::before { background: #3ba87d; }

.kpi-label {
    font-size: 0.72rem;
    color: #4a7aaa;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #e8f4ff;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #4a7aaa;
    margin-top: 0.3rem;
}

/* 風險評級徽章 */
.risk-badge {
    display: inline-block;
    padding: 0.18rem 0.65rem;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.risk-critical { background:#3d0e0e; color:#ff6b6b; border:1px solid #7a1f1f; }
.risk-high     { background:#3d2200; color:#ffa94d; border:1px solid #7a4400; }
.risk-medium   { background:#2d2d00; color:#ffd43b; border:1px solid #5a5a00; }
.risk-low      { background:#0e2d1e; color:#69db7c; border:1px solid #1a5c3a; }

/* 診斷書 */
.diagnosis-card {
    background: #080f1e;
    border: 1px solid #1a2e4a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.diagnosis-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #1a2e4a;
}
.diagnosis-uid {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    color: #a8d4ff;
    font-weight: 600;
}
.diagnosis-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    line-height: 1;
}
.score-critical { color: #ff6b6b; }
.score-high     { color: #ffa94d; }
.score-medium   { color: #ffd43b; }
.score-low      { color: #69db7c; }

.factor-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid #111c2e;
}
.factor-rank {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #2a5a9f;
    min-width: 20px;
}
.factor-name { font-size: 0.85rem; color: #a8c8e8; flex: 1; }
.factor-bar-wrap { width: 120px; height: 6px; background: #111c2e; border-radius: 3px; }
.factor-bar { height: 6px; border-radius: 3px; }
.bar-up   { background: #e24b4a; }
.bar-down { background: #3ba87d; }
.factor-shap {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    min-width: 60px;
    text-align: right;
}
.shap-up   { color: #ff8787; }
.shap-down { color: #69db7c; }

.nl-summary {
    background: #0a1525;
    border-left: 3px solid #2a5a9f;
    padding: 0.8rem 1rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #8ab4d8;
    line-height: 1.6;
    margin-top: 0.8rem;
}

/* 表格 */
.stDataFrame { font-size: 0.82rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #050c18;
    border-right: 1px solid #111f35;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 路徑設定
# ============================================================
REPORTS_DIR = Path("outputs/reports")
PLOTS_DIR   = Path("outputs/plots")

RISK_SCORES_PATH  = REPORTS_DIR / "user_risk_scores.csv"
DIAGNOSIS_PATH    = REPORTS_DIR / "risk_diagnosis.json"
DIAGNOSIS_CSV     = REPORTS_DIR / "risk_diagnosis.csv"
METRICS_PATH      = REPORTS_DIR / "model_metrics.json"
GRAPH_FEAT_PATH   = REPORTS_DIR / "graph_analysis.csv"
GRAPH_GROUPS_PATH = REPORTS_DIR / "graph_analysis.json"
SHAP_SUMMARY_IMG  = PLOTS_DIR   / "shap_summary.png"
SHAP_IMPORT_IMG   = PLOTS_DIR   / "shap_importance.png"
GRAPH_FULL_IMG    = PLOTS_DIR   / "graph_full_network.png"
GRAPH_BL_IMG      = PLOTS_DIR   / "graph_blacklist_neighborhood.png"


# ============================================================
# 資料載入（快取避免每次互動都重新讀檔）
# ============================================================

@st.cache_data
def load_risk_scores() -> pd.DataFrame:
    if RISK_SCORES_PATH.exists():
        return pd.read_csv(RISK_SCORES_PATH)
    return pd.DataFrame()


@st.cache_data
def load_diagnosis() -> list:
    if DIAGNOSIS_PATH.exists():
        with open(DIAGNOSIS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    if DIAGNOSIS_CSV.exists():
        return pd.read_csv(DIAGNOSIS_CSV).to_dict("records")
    return []


@st.cache_data
def load_metrics() -> list:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def load_graph_features() -> pd.DataFrame:
    if GRAPH_FEAT_PATH.exists():
        return pd.read_csv(GRAPH_FEAT_PATH)
    return pd.DataFrame()


@st.cache_data
def load_graph_groups() -> list:
    if GRAPH_GROUPS_PATH.exists():
        with open(GRAPH_GROUPS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ============================================================
# 工具函式
# ============================================================

def risk_css_class(level: str) -> str:
    mapping = {"極高": "critical", "高": "high", "中": "medium", "低": "low"}
    return f"risk-{mapping.get(level, 'low')}"


def score_css_class(score: float) -> str:
    if score >= 0.85: return "score-critical"
    if score >= 0.65: return "score-high"
    if score >= 0.40: return "score-medium"
    return "score-low"


def risk_level_from_score(score: float) -> str:
    if score >= 0.85: return "極高"
    if score >= 0.65: return "高"
    if score >= 0.40: return "中"
    return "低"


def risk_emoji(level: str) -> str:
    return {"極高": "🔴", "高": "🟠", "中": "🟡", "低": "🟢"}.get(level, "⚪")


# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="bito-header">
  <div>
    <h1>🛡️ BitoGuard</h1>
    <div class="subtitle">智慧合規風險雷達 &nbsp;·&nbsp; AI-Powered AML Detection</div>
  </div>
  <div style="margin-left:auto; display:flex; gap:0.5rem; align-items:center;">
    <span class="bito-badge">LIVE</span>
    <span class="bito-badge">BitoPro × AWS</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar 導覽
# ============================================================
with st.sidebar:
    st.markdown("### 🗂 功能選單")
    page = st.radio(
        label="",
        options=[
            "📊 總覽儀表板",
            "🎯 高風險用戶排行",
            "🔍 個人風險診斷書",
            "🕸️ 關聯圖譜",
            "📈 模型效能報告",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ 閾值設定")
    threshold = st.slider(
        "風險判定閾值",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="高於此分數的用戶將被標記為高風險",
    )

    st.markdown("---")
    st.caption("BitoGuard v1.0  |  2026 去偽存真黑客松")


# ============================================================
# 載入資料
# ============================================================
risk_df    = load_risk_scores()
diagnoses  = load_diagnosis()
metrics    = load_metrics()
graph_feat = load_graph_features()
graph_grps = load_graph_groups()

# 資料防呆
has_risk     = not risk_df.empty
has_diag     = len(diagnoses) > 0
has_metrics  = len(metrics) > 0
has_graph    = not graph_feat.empty


# ============================================================
# ① 總覽儀表板
# ============================================================
if page == "📊 總覽儀表板":
    st.markdown("## 📊 總覽儀表板")

    if not has_risk:
        st.warning("⚠️ 尚未找到風險分數資料，請先執行完整 Pipeline。")
        st.stop()

    flagged_df = risk_df[risk_df["risk_score"] >= threshold]
    total      = len(risk_df)
    flagged    = len(flagged_df)
    confirmed_bl = int(risk_df["true_label"].sum()) if "true_label" in risk_df.columns else 0
    avg_score  = risk_df["risk_score"].mean()

    # KPI 卡片
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card red">
            <div class="kpi-label">高風險用戶數</div>
            <div class="kpi-value">{flagged:,}</div>
            <div class="kpi-sub">閾值 ≥ {threshold:.0%}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card amber">
            <div class="kpi-label">已知黑名單</div>
            <div class="kpi-value">{confirmed_bl:,}</div>
            <div class="kpi-sub">測試集標記數</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card blue">
            <div class="kpi-label">分析用戶總數</div>
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-sub">本次測試集</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card green">
            <div class="kpi-label">平均風險分數</div>
            <div class="kpi-value">{avg_score:.3f}</div>
            <div class="kpi-sub">集成模型輸出</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 圖表區
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 風險分數分布")
        risk_df["risk_level"] = risk_df["risk_score"].apply(risk_level_from_score)
        fig = px.histogram(
            risk_df, x="risk_score", nbins=50,
            color="risk_level",
            color_discrete_map={
                "極高": "#e24b4a", "高": "#ef9f27",
                "中":   "#ffd43b", "低": "#3ba87d",
            },
            labels={"risk_score": "風險分數", "count": "用戶數"},
            template="plotly_dark",
        )
        fig.add_vline(x=threshold, line_dash="dash",
                      line_color="#5ab8ff", annotation_text=f"閾值 {threshold:.2f}")
        fig.update_layout(
            paper_bgcolor="#080f1e", plot_bgcolor="#080f1e",
            font_color="#a8c8e8", legend_title_text="風險等級",
            margin=dict(l=0, r=0, t=10, b=0), height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 風險等級分布")
        level_counts = risk_df["risk_level"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=level_counts.index,
            values=level_counts.values,
            hole=0.55,
            marker_colors=["#e24b4a", "#ef9f27", "#ffd43b", "#3ba87d"],
        ))
        fig2.update_layout(
            paper_bgcolor="#080f1e", font_color="#a8c8e8",
            showlegend=True, height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="v", x=1.05),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 時間軸（若有 true_label 則顯示抓獲率）
    if has_metrics:
        best = max(metrics, key=lambda x: x.get("f1", 0))
        st.markdown("---")
        st.markdown("#### 最佳模型指標快覽")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("F1 Score",  f"{best.get('f1', 0):.4f}")
        m2.metric("Precision", f"{best.get('precision', 0):.4f}")
        m3.metric("Recall",    f"{best.get('recall', 0):.4f}")
        m4.metric("AUC-PR",    f"{best.get('auc_pr', 0):.4f}")
        st.caption(f"模型：{best.get('model', 'N/A')}　閾值：{best.get('threshold', 0.5):.2f}")


# ============================================================
# ② 高風險用戶排行榜
# ============================================================
elif page == "🎯 高風險用戶排行":
    st.markdown("## 🎯 高風險用戶排行榜")

    if not has_risk:
        st.warning("⚠️ 尚未找到風險分數資料。")
        st.stop()

    flagged_df = (
        risk_df[risk_df["risk_score"] >= threshold]
        .sort_values("risk_score", ascending=False)
        .reset_index(drop=True)
    )
    flagged_df["風險等級"] = flagged_df["risk_score"].apply(risk_level_from_score)
    flagged_df["排名"] = range(1, len(flagged_df) + 1)

    st.markdown(f"共 **{len(flagged_df)}** 位用戶超過閾值 `{threshold:.2f}`")

    # 搜尋
    search = st.text_input("🔎 搜尋用戶 ID", placeholder="輸入 user_id...")

    display_df = flagged_df.copy()
    if search:
        display_df = display_df[
            display_df["user_id"].astype(str).str.contains(search)
        ]

    # 顯示欄位
    show_cols = ["排名", "user_id", "risk_score", "風險等級"]
    if "true_label" in display_df.columns:
        display_df["驗證標籤"] = display_df["true_label"].apply(
            lambda x: "✅ 黑名單" if x == 1 else "⬜ 正常"
        )
        show_cols.append("驗證標籤")

    # 加入其他模型分數（若有）
    prob_cols = [c for c in display_df.columns if c.startswith("prob_")]
    show_cols += prob_cols

    st.dataframe(
        display_df[show_cols].rename(columns={
            "user_id": "用戶 ID",
            "risk_score": "風險分數",
        }),
        use_container_width=True,
        height=500,
    )

    # 散點圖：各模型分數比較
    if len(prob_cols) >= 2:
        st.markdown("#### 各模型風險分數比較")
        fig = px.scatter(
            display_df,
            x=prob_cols[0], y=prob_cols[1],
            color="風險等級",
            hover_data=["user_id", "risk_score"],
            color_discrete_map={
                "極高": "#e24b4a", "高": "#ef9f27",
                "中":   "#ffd43b", "低": "#3ba87d",
            },
            template="plotly_dark",
            labels={prob_cols[0]: prob_cols[0].replace("prob_", "").upper(),
                    prob_cols[1]: prob_cols[1].replace("prob_", "").upper()},
        )
        fig.update_layout(
            paper_bgcolor="#080f1e", plot_bgcolor="#080f1e",
            font_color="#a8c8e8", height=350,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# ③ 個人風險診斷書
# ============================================================
elif page == "🔍 個人風險診斷書":
    st.markdown("## 🔍 個人風險診斷書")

    if not has_diag:
        st.warning("⚠️ 尚未找到診斷書資料，請先執行 shap_explainer.py。")
        st.stop()

    # 選擇用戶
    uid_list = [str(d.get("user_id", "")) for d in diagnoses]
    selected_uid = st.selectbox("選擇用戶 ID", uid_list)

    diag = next((d for d in diagnoses
                 if str(d.get("user_id")) == selected_uid), None)
    if not diag:
        st.error("找不到該用戶的診斷書")
        st.stop()

    score      = diag.get("risk_score", 0)
    level      = diag.get("risk_level", "低")
    nl_summary = diag.get("nl_summary", "無資料")
    factors    = diag.get("top_factors", [])
    true_label = diag.get("true_label")
    css_score  = score_css_class(score)
    css_badge  = risk_css_class(level)

    # 診斷書主體
    label_tag = ""
    if true_label == 1:
        label_tag = '<span style="color:#ff6b6b;font-size:0.8rem;margin-left:8px">✅ 已驗證黑名單</span>'
    elif true_label == 0:
        label_tag = '<span style="color:#69db7c;font-size:0.8rem;margin-left:8px">⬜ 驗證為正常用戶</span>'

    st.markdown(f"""
    <div class="diagnosis-card">
      <div class="diagnosis-header">
        <div>
          <div class="diagnosis-uid">用戶 #{selected_uid}{label_tag}</div>
          <div style="margin-top:0.5rem;">
            <span class="risk-badge {css_badge}">{risk_emoji(level)} {level}風險</span>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:0.7rem;color:#4a7aaa;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.2rem;">風險分數</div>
          <div class="diagnosis-score {css_score}">{score:.3f}</div>
        </div>
      </div>

      <div style="font-size:0.72rem;color:#4a7aaa;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem;">Top 5 風險因子</div>
    """, unsafe_allow_html=True)

    # 因子列表
    max_shap = max((abs(f["shap"]) for f in factors), default=1)
    for f in factors:
        bar_pct = int(abs(f["shap"]) / max_shap * 100)
        bar_cls = "bar-up" if f["shap"] > 0 else "bar-down"
        shap_cls = "shap-up" if f["shap"] > 0 else "shap-down"
        direction_icon = "↑" if f["shap"] > 0 else "↓"
        st.markdown(f"""
      <div class="factor-row">
        <span class="factor-rank">#{f['rank']}</span>
        <span class="factor-name">{f['label']}</span>
        <span style="font-size:0.75rem;color:#6a8aaa;min-width:70px;">{f['value']}</span>
        <div class="factor-bar-wrap">
          <div class="factor-bar {bar_cls}" style="width:{bar_pct}%;"></div>
        </div>
        <span class="factor-shap {shap_cls}">{direction_icon}{abs(f['shap']):.4f}</span>
      </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
      <div class="nl-summary">
        <span style="font-size:0.7rem;color:#2a5a9f;letter-spacing:0.08em;text-transform:uppercase;">AI 風險摘要</span><br>
        {nl_summary}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Waterfall 圖
    plot_path = diag.get("waterfall_plot")
    if plot_path and Path(plot_path).exists():
        st.markdown("#### SHAP Waterfall 圖")
        st.image(plot_path, use_column_width=True)
    else:
        st.caption("Waterfall 圖尚未生成（請執行 shap_explainer.py）")


# ============================================================
# ④ 關聯圖譜
# ============================================================
elif page == "🕸️ 關聯圖譜":
    st.markdown("## 🕸️ 關聯圖譜分析")

    tab1, tab2, tab3 = st.tabs(["🌐 全局圖譜", "🔴 黑名單鄰域", "📋 高風險群組"])

    with tab1:
        if GRAPH_FULL_IMG.exists():
            st.image(str(GRAPH_FULL_IMG), use_column_width=True)
            st.caption("紅點 = 黑名單  ·  橘點 = 直接關聯黑名單  ·  藍點 = 一般用戶")
        else:
            st.info("請先執行 graph_analysis.py 生成圖譜")

        if has_graph:
            st.markdown("#### 圖特徵統計")
            gc1, gc2, gc3, gc4 = st.columns(4)
            gc1.metric("總節點數", f"{len(graph_feat):,}")
            gc2.metric("黑名單節點",
                       f"{int(graph_feat['is_blacklist'].sum()):,}"
                       if "is_blacklist" in graph_feat.columns else "N/A")
            gc3.metric("1跳關聯正常用戶",
                       f"{int((graph_feat['graph_bl_neighbors'] > 0).sum()):,}"
                       if "graph_bl_neighbors" in graph_feat.columns else "N/A")
            gc4.metric("2跳關聯正常用戶",
                       f"{int((graph_feat['graph_bl_2hop'] > 0).sum()):,}"
                       if "graph_bl_2hop" in graph_feat.columns else "N/A")

    with tab2:
        if GRAPH_BL_IMG.exists():
            st.image(str(GRAPH_BL_IMG), use_column_width=True)
            st.caption("以黑名單用戶為中心，展示 2 層內的關聯帳號共犯結構")
        else:
            st.info("請先執行 graph_analysis.py 生成圖譜")

    with tab3:
        if graph_grps:
            st.markdown(f"共找到 **{len(graph_grps)}** 個高風險關聯群組")
            for i, grp in enumerate(graph_grps[:10], 1):
                with st.expander(
                    f"#{i}  黑名單中心：{grp['center_uid']}  |  "
                    f"1跳 {grp['hop1_count']} 人  2跳 {grp['hop2_count']} 人  "
                    f"群內其他黑名單 {grp['other_bl_count']} 位"
                ):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**直接關聯（1 跳）用戶**")
                        st.write(grp.get("hop1_users", []))
                    with col_b:
                        st.markdown("**2 跳關聯用戶（前 20）**")
                        st.write(grp.get("hop2_users", [])[:20])
                    if grp.get("other_bl_in_group"):
                        st.error(f"⚠️ 群內其他黑名單：{grp['other_bl_in_group']}")
        else:
            st.info("請先執行 graph_analysis.py")

        if has_graph and "graph_degree" in graph_feat.columns:
            st.markdown("#### 節點連接度分布")
            fig = px.histogram(
                graph_feat, x="graph_degree", nbins=40,
                template="plotly_dark",
                labels={"graph_degree": "連接度（Degree）", "count": "用戶數"},
                color_discrete_sequence=["#378add"],
            )
            fig.update_layout(
                paper_bgcolor="#080f1e", plot_bgcolor="#080f1e",
                font_color="#a8c8e8", height=250,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# ⑤ 模型效能報告
# ============================================================
elif page == "📈 模型效能報告":
    st.markdown("## 📈 模型效能報告")

    if not has_metrics:
        st.warning("⚠️ 尚未找到模型指標，請先執行 train_model.py。")
        st.stop()

    metrics_df = pd.DataFrame(metrics)

    # 模型比較表
    st.markdown("#### 各模型指標比較")
    show_m_cols = ["model", "precision", "recall", "f1", "auc_pr",
                   "fpr", "tp", "fp", "fn", "tn"]
    show_m_cols = [c for c in show_m_cols if c in metrics_df.columns]

    best_f1_idx = metrics_df["f1"].idxmax() if "f1" in metrics_df.columns else None

    st.dataframe(
        metrics_df[show_m_cols].rename(columns={
            "model": "模型", "precision": "精確率", "recall": "召回率",
            "f1": "F1 Score", "auc_pr": "AUC-PR", "fpr": "誤報率",
            "tp": "TP", "fp": "FP", "fn": "FN", "tn": "TN",
        }),
        use_container_width=True,
    )

    if best_f1_idx is not None:
        best = metrics_df.loc[best_f1_idx]
        st.success(f"🏆 最佳模型：**{best['model']}**　F1={best['f1']:.4f}　閾值={best.get('threshold', 0.5):.2f}")

    # 雷達圖
    st.markdown("#### 模型能力雷達圖")
    radar_metrics = ["precision", "recall", "f1", "auc_pr"]
    radar_metrics = [c for c in radar_metrics if c in metrics_df.columns]

    fig = go.Figure()
    colors = ["#e24b4a", "#ef9f27", "#378add", "#3ba87d", "#9b59b6"]
    for i, row in metrics_df.iterrows():
        vals = [row.get(m, 0) for m in radar_metrics]
        vals_closed = vals + [vals[0]]
        labels_closed = radar_metrics + [radar_metrics[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            name=row["model"],
            line_color=colors[i % len(colors)],
            opacity=0.6,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="#080f1e",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="#1a2e4a", color="#4a7aaa"),
            angularaxis=dict(gridcolor="#1a2e4a", color="#4a7aaa"),
        ),
        paper_bgcolor="#080f1e", font_color="#a8c8e8",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
        height=400, margin=dict(l=40, r=40, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # SHAP 特徵重要性圖
    st.markdown("#### SHAP 特徵重要性")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if SHAP_IMPORT_IMG.exists():
            st.image(str(SHAP_IMPORT_IMG), caption="全局平均 |SHAP|", use_column_width=True)
        else:
            st.info("請執行 shap_explainer.py 生成圖表")
    with col_s2:
        if SHAP_SUMMARY_IMG.exists():
            st.image(str(SHAP_SUMMARY_IMG), caption="SHAP Beeswarm（顏色=特徵值高低）", use_column_width=True)
        else:
            st.info("請執行 shap_explainer.py 生成圖表")

    # 混淆矩陣（最佳模型）
    if best_f1_idx is not None:
        best = metrics_df.loc[best_f1_idx]
        tp = best.get("tp", 0)
        fp = best.get("fp", 0)
        fn = best.get("fn", 0)
        tn = best.get("tn", 0)

        st.markdown(f"#### 混淆矩陣（{best['model']}）")
        cm_fig = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["預測正常", "預測黑名單"],
            y=["實際正常", "實際黑名單"],
            colorscale=[[0, "#080f1e"], [0.5, "#1a3a6b"], [1, "#e24b4a"]],
            text=[[str(tn), str(fp)], [str(fn), str(tp)]],
            texttemplate="%{text}",
            textfont={"size": 18, "family": "IBM Plex Mono"},
            showscale=False,
        ))
        cm_fig.update_layout(
            paper_bgcolor="#080f1e", font_color="#a8c8e8",
            height=300, margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(cm_fig, use_container_width=True)
        st.caption(
            f"TP（正確抓到黑名單）={tp}　"
            f"FP（誤報正常用戶）={fp}　"
            f"FN（漏掉的黑名單）={fn}　"
            f"TN（正確放行）={tn}"
        )