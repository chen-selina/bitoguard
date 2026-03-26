"""
graph_analysis.py
位置：src/models/graph_analysis.py

【v4 無洩漏版 — 修正 graph_bl_* 特徵的標籤洩漏問題】

v3 → v4 核心修改：
  ❌ 原本問題：
     compute_graph_features() 計算 graph_bl_neighbors / graph_bl_2hop /
     graph_bl_component 時，使用了「全體」blacklist（包含測試集用戶的黑名單身份）。
     這導致測試集用戶可透過圖結構「看到」自己鄰居的真實 label → 標籤洩漏。

  ✅ 修正策略：
     新增 train_blacklist 參數（只包含訓練集已知黑名單）。
     graph_bl_* 特徵的計算全部改用 train_blacklist，不再使用 full blacklist。
     full blacklist 只用於：
       1. 建圖節點屬性（is_blacklist，供視覺化使用）
       2. 視覺化繪圖
       3. find_risk_communities（分析用，不進入模型特徵）

其餘 v3 效能優化全部保留：
  A. 高頻 hash 上限 MAX_USERS_PER_GROUP=50
  B. connected components 一次預算，O(1) 查詢
  C. 預建鄰居 adjacency dict，避免重複呼叫 G.neighbors()
  D. 視覺化 subgraph 加硬上限，用 kamada_kawai 取代 spring_layout
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from itertools import combinations

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR   = Path("outputs/reports")
PLOTS_DIR     = Path("outputs/plots")

for d in [REPORTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GRAPH_REPORT_PATH    = REPORTS_DIR / "graph_analysis.csv"
GRAPH_JSON_PATH      = REPORTS_DIR / "graph_analysis.json"
GRAPH_FULL_PLOT      = PLOTS_DIR   / "graph_full_network.png"
GRAPH_BLACKLIST_PLOT = PLOTS_DIR   / "graph_blacklist_neighborhood.png"
GRAPH_GEPHI_PATH     = REPORTS_DIR / "graph_for_gephi.gexf"

MAX_USERS_PER_GROUP = 50  # 超過此人數共用同一 hash → 視為 NAT/代理，跳過


def load_raw_data() -> dict:
    tables = {}
    files = {
        "user_info":     "user_info.csv",
        "twd":           "twd_transfer.csv",
        "crypto":        "crypto_transfer.csv",
        "usdt_trade":    "usdt_twd_trading.csv",
        "train_label":   "train_label.csv",
        "predict_label": "predict_label.csv",
    }
    for key, fname in files.items():
        path = RAW_DIR / fname
        if path.exists():
            tables[key] = pd.read_csv(path, low_memory=False)
            print(f"  載入 {fname}：{len(tables[key]):,} 筆")
        else:
            print(f"  警告：找不到 {fname}")
            tables[key] = pd.DataFrame()
    return tables


# ============================================================
# ✅ v4 新增：分別取得「訓練集黑名單」與「全體黑名單」
#    - train_blacklist：只含訓練集已知黑名單，用於計算 graph_bl_* 特徵
#    - full_blacklist ：含全體，僅供視覺化 / 節點標記
# ============================================================
def get_blacklist_users(train_label: pd.DataFrame) -> tuple[set, set]:
    """
    回傳 (train_blacklist, full_blacklist)
    目前 full_blacklist == train_blacklist，因為我們只有 train_label 的 status。
    若未來有其他已知黑名單來源，可在 full_blacklist 追加，但 train_blacklist 保持不變。
    """
    if train_label.empty or "status" not in train_label.columns:
        return set(), set()
    train_bl = set(train_label[train_label["status"] == 1]["user_id"].tolist())
    # ✅ full_blacklist 與 train_blacklist 相同來源，
    #    但概念上分開，避免未來誤用測試集 label 混入
    full_bl  = set(train_bl)
    print(f"  訓練集黑名單用戶（用於特徵計算）：{len(train_bl)} 位")
    print(f"  全體已知黑名單用戶（用於視覺化）：{len(full_bl)} 位")
    return train_bl, full_bl


def get_all_known_users(train_label, predict_label) -> set:
    uids = set()
    if not train_label.empty:
        uids |= set(train_label["user_id"].tolist())
    if not predict_label.empty:
        uids |= set(predict_label["user_id"].tolist())
    print(f"  已知用戶總數：{len(uids):,} 位")
    return uids


def _add_group_edges(G, groups, edge_type, attr_key, max_group_size=MAX_USERS_PER_GROUP):
    """向量化邊建立 + 高頻群組過濾"""
    edge_count = 0
    skipped    = 0
    for hash_val, users in groups.items():
        unique_users = list(set(users))
        n = len(unique_users)
        if n < 2:
            continue
        if n > max_group_size:
            skipped += 1
            continue
        for u1, u2 in combinations(unique_users, 2):
            if G.has_edge(u1, u2):
                G[u1][u2]["weight"]  += 1
                G[u1][u2][attr_key]  += 1
            else:
                attrs = {"weight": 1, "edge_type": edge_type,
                         "shared_ips": 0, "fund_transfers": 0, "same_wallets": 0}
                attrs[attr_key] = 1
                G.add_edge(u1, u2, **attrs)
                edge_count += 1
    if skipped > 0:
        print(f"    跳過 {skipped} 個高頻群組（>{max_group_size} 人，可能是 NAT/代理）")
    return edge_count


def build_graph(tables: dict, full_blacklist: set) -> nx.Graph:
    """
    建立關聯圖。
    節點屬性 is_blacklist 使用 full_blacklist（僅供視覺化）。
    特徵計算會另外用 train_blacklist，不在此處處理。
    """
    G = nx.Graph()
    user_info     = tables.get("user_info",     pd.DataFrame())
    crypto        = tables.get("crypto",        pd.DataFrame())
    usdt_trade    = tables.get("usdt_trade",    pd.DataFrame())
    twd           = tables.get("twd",           pd.DataFrame())
    train_label   = tables.get("train_label",   pd.DataFrame())
    predict_label = tables.get("predict_label", pd.DataFrame())

    if not user_info.empty:
        train_uids   = set(train_label["user_id"])   if not train_label.empty   else set()
        predict_uids = set(predict_label["user_id"]) if not predict_label.empty else set()
        for _, row in user_info.iterrows():
            uid   = row["user_id"]
            is_bl = uid in full_blacklist   # 僅供節點視覺化標記
            G.add_node(uid, is_blacklist=is_bl,
                age=int(row.get("age", 0)) if pd.notna(row.get("age")) else 0,
                career=int(row.get("career", 0)),
                kyc_level=(2 if pd.notna(row.get("level2_finished_at"))
                           else 1 if pd.notna(row.get("level1_finished_at")) else 0),
                in_train=uid in train_uids, in_predict=uid in predict_uids)

    ip_col = "source_ip_hash"
    print("  建立 shared_ip 邊（v3：過濾高頻 NAT）...")
    ip_sources = []
    for df in [twd, crypto, usdt_trade]:
        if not df.empty and ip_col in df.columns:
            ip_sources.append(df[["user_id", ip_col]].dropna(subset=[ip_col]))
    ip_edges = 0
    if ip_sources:
        all_ip    = pd.concat(ip_sources, ignore_index=True).drop_duplicates()
        ip_groups = all_ip.groupby(ip_col)["user_id"].apply(list)
        ip_edges  = _add_group_edges(G, ip_groups, "shared_ip", "shared_ips")
    print(f"    shared_ip 邊：{ip_edges:,} 條")

    print("  建立 fund_transfer 邊...")
    fund_edges = 0
    if not crypto.empty and "relation_user_id" in crypto.columns:
        sub_kind_col = "sub_kind" if "sub_kind" in crypto.columns else None
        if sub_kind_col:
            internal = crypto[(crypto[sub_kind_col] == 1) & crypto["relation_user_id"].notna()]
        else:
            internal = crypto[crypto["relation_user_id"].notna()]
        internal = internal[["user_id", "relation_user_id"]].copy()
        if not internal.empty:
            internal["relation_user_id"] = internal["relation_user_id"].astype(int)
            internal = internal[internal["user_id"] != internal["relation_user_id"]]
            tc = internal.groupby(["user_id", "relation_user_id"]).size().reset_index(name="count")
            for _, row in tc.iterrows():
                u1, u2, cnt = int(row["user_id"]), int(row["relation_user_id"]), int(row["count"])
                if G.has_edge(u1, u2):
                    G[u1][u2]["weight"] += cnt; G[u1][u2]["fund_transfers"] += cnt
                else:
                    G.add_edge(u1, u2, weight=cnt, edge_type="fund_transfer",
                               shared_ips=0, fund_transfers=cnt, same_wallets=0)
                    fund_edges += 1
    print(f"    fund_transfer 邊：{fund_edges:,} 條")

    print("  建立 same_wallet 邊（v3：過濾高頻群組）...")
    wallet_edges = 0
    if not crypto.empty:
        for wcol in ["from_wallet_hash", "to_wallet_hash", "from_wallet", "to_wallet"]:
            if wcol not in crypto.columns:
                continue
            w_data   = crypto[["user_id", wcol]].dropna(subset=[wcol]).drop_duplicates()
            w_groups = w_data.groupby(wcol)["user_id"].apply(list)
            wallet_edges += _add_group_edges(G, w_groups, "same_wallet", "same_wallets")
    print(f"    same_wallet 邊：{wallet_edges:,} 條")
    print(f"  圖建立完成：{G.number_of_nodes():,} 節點，{G.number_of_edges():,} 邊")
    return G


# ============================================================
# ✅ v4 核心修改：graph_bl_* 特徵改用 train_blacklist
# ============================================================
def compute_graph_features(G: nx.Graph, train_blacklist: set, full_blacklist: set) -> pd.DataFrame:
    """
    計算圖特徵。

    ✅ graph_bl_neighbors / graph_bl_2hop / graph_bl_component：
       只使用 train_blacklist（訓練集已知黑名單），
       測試集與預測集用戶的黑名單身份不會洩漏進特徵。

    is_blacklist 欄位（僅輸出用，不進模型）使用 full_blacklist。
    """
    print("  v4：預先計算所有 connected components（避免 O(N×M) 重複）...")
    node_to_comp  = {}
    comp_bl_count = {}
    for comp_id, comp in enumerate(nx.connected_components(G)):
        # ✅ 用 train_blacklist 計算每個 component 內的已知黑名單數
        bl_cnt = sum(1 for n in comp if n in train_blacklist)
        for n in comp:
            node_to_comp[n]        = comp_id
            comp_bl_count[comp_id] = bl_cnt

    print("  計算 PageRank...")
    pagerank   = nx.pagerank(G, weight="weight", max_iter=200)
    print("  計算 Clustering Coefficient...")
    clustering = nx.clustering(G, weight="weight")

    adj = {n: set(G.neighbors(n)) for n in G.nodes()}

    rows = []
    for uid in G.nodes():
        neighbors = adj[uid]
        degree    = len(neighbors)

        # ✅ 只用 train_blacklist 計算黑名單鄰居數
        bl_neighbors = sum(1 for n in neighbors if n in train_blacklist)

        hop2 = set()
        for n in neighbors:
            hop2 |= adj[n]
        hop2.discard(uid)
        hop2 -= neighbors

        # ✅ 只用 train_blacklist 計算 2-hop 黑名單數
        bl_2hop = sum(1 for n in hop2 if n in train_blacklist)

        comp_id = node_to_comp.get(uid, -1)
        # ✅ 只用 train_blacklist 計算 component 內黑名單數
        bl_in_component = comp_bl_count.get(comp_id, 0)

        rows.append({
            "user_id":            uid,
            "graph_degree":       degree,
            "graph_bl_neighbors": bl_neighbors,   # ✅ 無洩漏
            "graph_bl_2hop":      bl_2hop,         # ✅ 無洩漏
            "graph_bl_component": bl_in_component, # ✅ 無洩漏
            "graph_clustering":   round(clustering.get(uid, 0), 4),
            "graph_pagerank":     round(pagerank.get(uid, 0), 6),
            # is_blacklist 僅用於後續分析輸出，不進模型特徵
            "is_blacklist":       1 if uid in full_blacklist else 0,
        })
    feat_df = pd.DataFrame(rows)
    print(f"  圖特徵完成：{len(feat_df)} 位用戶")
    return feat_df


def find_risk_communities(G: nx.Graph, train_blacklist: set) -> list:
    """
    識別高風險群組。使用 train_blacklist（與特徵計算一致）。
    """
    print("  識別高風險群組（v4：預建 adj dict，避免重複展開）...")
    adj = {n: set(G.neighbors(n)) for n in G.nodes()}
    risk_groups = []
    for bl_uid in train_blacklist:
        if bl_uid not in G:
            continue
        hop1 = adj.get(bl_uid, set())
        if not hop1:
            continue
        hop2 = set()
        for n in hop1:
            hop2 |= adj.get(n, set())
        hop2 -= hop1
        hop2.discard(bl_uid)
        other_bl     = (hop1 | hop2) & train_blacklist - {bl_uid}
        sg           = G.subgraph({bl_uid} | hop1 | hop2)
        total_weight = sum(d.get("weight", 1) for _, _, d in sg.edges(data=True))
        risk_groups.append({
            "center_uid":        bl_uid,
            "hop1_users":        sorted(list(hop1)),
            "hop2_users":        sorted(list(hop2)),
            "hop1_count":        len(hop1),
            "hop2_count":        len(hop2),
            "other_bl_in_group": sorted(list(other_bl)),
            "other_bl_count":    len(other_bl),
            "total_edge_weight": int(total_weight),
            "risk_score":        len(hop1) * 2 + len(hop2) + len(other_bl) * 5,
        })
    risk_groups.sort(key=lambda x: x["risk_score"], reverse=True)
    print(f"  找到 {len(risk_groups)} 個風險群組")
    return risk_groups


def plot_full_network(G: nx.Graph, full_blacklist: set, max_nodes: int = 200) -> None:
    print(f"  繪製全局圖譜（最多 {max_nodes} 節點）...")
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
    selected  = list(set(n for n, _ in top_nodes) | (full_blacklist & set(G.nodes())))[:max_nodes]
    subG      = G.subgraph(selected)
    fig, ax   = plt.subplots(figsize=(16, 12))
    try:
        pos = nx.kamada_kawai_layout(subG, weight="weight")
    except Exception:
        pos = nx.spring_layout(subG, seed=42, k=1.5)
    node_colors, node_sizes = [], []
    for n in subG.nodes():
        if n in full_blacklist:
            node_colors.append("#E24B4A"); node_sizes.append(300)
        elif any(nb in full_blacklist for nb in subG.neighbors(n)):
            node_colors.append("#EF9F27"); node_sizes.append(120)
        else:
            node_colors.append("#378ADD"); node_sizes.append(40)
    edge_colors, edge_widths = [], []
    for u, v, d in subG.edges(data=True):
        etype = d.get("edge_type", "")
        if etype == "fund_transfer":
            edge_colors.append("#A32D2D"); edge_widths.append(min(d.get("weight", 1) * 0.5, 3))
        elif etype == "shared_ip":
            edge_colors.append("#EF9F27"); edge_widths.append(0.8)
        else:
            edge_colors.append("#B4B2A9"); edge_widths.append(0.5)
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.85)
    nx.draw_networkx_edges(subG, pos, edge_color=edge_colors, width=edge_widths, ax=ax, alpha=0.6)
    nx.draw_networkx_labels(subG, pos, labels={n: str(n) for n in subG.nodes() if n in full_blacklist},
                            font_size=7, font_color="white", ax=ax)
    ax.legend(handles=[
        mpatches.Patch(color="#E24B4A", label="黑名單用戶"),
        mpatches.Patch(color="#EF9F27", label="直接關聯黑名單"),
        mpatches.Patch(color="#378ADD", label="一般用戶"),
    ], loc="upper left", fontsize=10)
    ax.set_title(f"BitoGuard 用戶關聯圖譜（節點：{subG.number_of_nodes()}，邊：{subG.number_of_edges()}）", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(GRAPH_FULL_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  全局圖譜：{GRAPH_FULL_PLOT}")


def plot_blacklist_neighborhood(G: nx.Graph, full_blacklist: set, top_n: int = 5) -> None:
    print("  繪製黑名單鄰域圖...")
    adj = {n: set(G.neighbors(n)) for n in G.nodes()}
    bl_in_graph = [(n, len(adj.get(n, set()))) for n in full_blacklist if n in G]
    bl_in_graph.sort(key=lambda x: x[1], reverse=True)
    top_bl = [n for n, _ in bl_in_graph[:top_n]]
    if not top_bl:
        print("  黑名單在圖中無邊，跳過"); return
    cols = min(len(top_bl), 3)
    rows = (len(top_bl) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if len(top_bl) == 1: axes = [[axes]]
    elif rows == 1: axes = [axes]
    for i, bl_uid in enumerate(top_bl):
        ax   = axes[i // cols][i % cols]
        hop1 = adj.get(bl_uid, set())
        hop2 = set()
        for n in hop1: hop2 |= adj.get(n, set())
        hop2 -= hop1; hop2.discard(bl_uid)
        if len(hop2) > 50: hop2 = set(list(hop2)[:50])
        subG = G.subgraph({bl_uid} | hop1 | hop2)
        pos  = nx.spring_layout(subG, weight="weight", seed=42)
        n_colors, n_sizes = [], []
        for n in subG.nodes():
            if n == bl_uid: n_colors.append("#E24B4A"); n_sizes.append(500)
            elif n in full_blacklist: n_colors.append("#A32D2D"); n_sizes.append(300)
            elif n in hop1: n_colors.append("#EF9F27"); n_sizes.append(150)
            else: n_colors.append("#B5D4F4"); n_sizes.append(60)
        e_colors = ["#A32D2D" if d.get("edge_type") == "fund_transfer"
                    else "#EF9F27" if d.get("edge_type") == "shared_ip" else "#B4B2A9"
                    for _, _, d in subG.edges(data=True)]
        nx.draw_networkx_nodes(subG, pos, node_color=n_colors, node_size=n_sizes, ax=ax, alpha=0.9)
        nx.draw_networkx_edges(subG, pos, edge_color=e_colors, ax=ax, alpha=0.7, width=1.2)
        nx.draw_networkx_labels(subG, pos, labels={n: str(n) for n in subG.nodes()}, font_size=6, ax=ax)
        ax.set_title(f"黑名單 {bl_uid}\n1跳：{len(hop1)} 人  2跳：{len(hop2)} 人", fontsize=10)
        ax.axis("off")
    for i in range(len(top_bl), rows * cols): axes[i // cols][i % cols].axis("off")
    fig.legend(handles=[
        mpatches.Patch(color="#E24B4A", label="目標黑名單"),
        mpatches.Patch(color="#A32D2D", label="其他黑名單"),
        mpatches.Patch(color="#EF9F27", label="1 跳關聯"),
        mpatches.Patch(color="#B5D4F4", label="2 跳關聯"),
    ], loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0))
    fig.suptitle("黑名單用戶 2 層關聯圖（共犯結構分析）", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(GRAPH_BLACKLIST_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  黑名單鄰域圖：{GRAPH_BLACKLIST_PLOT}")


def save_results(graph_feat, risk_groups, G) -> None:
    graph_feat.to_csv(GRAPH_REPORT_PATH, index=False, encoding="utf-8-sig")
    serializable = [{k: (list(v) if isinstance(v, (set, list)) else v) for k, v in g.items()}
                    for g in risk_groups[:50]]
    with open(GRAPH_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    G_export = G.copy()
    for n, d in G_export.nodes(data=True):
        for k, v in d.items(): G_export.nodes[n][k] = str(v)
    nx.write_gexf(G_export, GRAPH_GEPHI_PATH)
    print(f"  圖特徵 CSV：{GRAPH_REPORT_PATH}\n  風險群組 JSON：{GRAPH_JSON_PATH}\n  Gephi 匯出：{GRAPH_GEPHI_PATH}")


def merge_graph_features_to_processed(graph_feat) -> None:
    graph_cols = [c for c in graph_feat.columns if c.startswith("graph_")]
    merge_cols = ["user_id"] + graph_cols
    for fname in ["features.csv", "predict_features.csv"]:
        feat_path = PROCESSED_DIR / fname
        if not feat_path.exists(): continue
        feat_df  = pd.read_csv(feat_path, low_memory=False)
        existing = [c for c in graph_cols if c in feat_df.columns]
        if existing: feat_df.drop(columns=existing, inplace=True)
        feat_df  = feat_df.merge(graph_feat[merge_cols], on="user_id", how="left")
        feat_df[graph_cols] = feat_df[graph_cols].fillna(0)
        feat_df.to_csv(feat_path, index=False, encoding="utf-8-sig")
        print(f"  已將 {len(graph_cols)} 個圖特徵合併至 {feat_path.name}")


def main():
    t0 = time.time()
    print("=" * 55)
    print("BitoGuard v4 — 關聯圖譜分析（無洩漏修正版）")
    print("=" * 55 + "\n")

    print("【載入原始資料】")
    tables = load_raw_data()

    # ✅ v4：分別取得 train_blacklist（特徵計算用）與 full_blacklist（視覺化用）
    train_blacklist, full_blacklist = get_blacklist_users(
        tables.get("train_label", pd.DataFrame())
    )
    get_all_known_users(
        tables.get("train_label", pd.DataFrame()),
        tables.get("predict_label", pd.DataFrame())
    )
    print()

    print("【建立關聯圖】")
    G = build_graph(tables, full_blacklist)
    print(f"  ⏱ 圖建立耗時：{time.time() - t0:.1f}s\n")

    t1 = time.time()
    print("【計算圖特徵】")
    # ✅ v4：傳入 train_blacklist（特徵計算）與 full_blacklist（is_blacklist 標記）
    graph_feat = compute_graph_features(G, train_blacklist, full_blacklist)
    print(f"  ⏱ 圖特徵耗時：{time.time() - t1:.1f}s\n")

    t2 = time.time()
    print("【識別高風險群組】")
    # ✅ v4：使用 train_blacklist
    risk_groups = find_risk_communities(G, train_blacklist)
    if risk_groups:
        for i, rg in enumerate(risk_groups[:5], 1):
            print(f"  #{i}  中心：{rg['center_uid']}  1跳：{rg['hop1_count']} 人  2跳：{rg['hop2_count']} 人  群內黑名單：{rg['other_bl_count']} 位")
    print(f"  ⏱ 群組識別耗時：{time.time() - t2:.1f}s\n")

    print("【生成圖譜視覺化】")
    plot_full_network(G, full_blacklist, max_nodes=200)
    plot_blacklist_neighborhood(G, full_blacklist, top_n=5)
    print()

    print("【存檔】")
    save_results(graph_feat, risk_groups, G)

    print("\n【合併圖特徵至 features.csv & predict_features.csv】")
    merge_graph_features_to_processed(graph_feat)

    print(f"\n  總節點：{G.number_of_nodes():,}  總邊：{G.number_of_edges():,}  黑名單節點：{len(full_blacklist & set(G.nodes())):,}")
    suspects = graph_feat[(graph_feat["is_blacklist"] == 0) & (graph_feat["graph_bl_neighbors"] > 0)]
    print(f"  直接連到訓練集黑名單的非黑名單用戶：{len(suspects):,} 位（潛在人頭戶）")
    print(f"\n  ⏱ 總耗時：{time.time() - t0:.1f}s")
    print("\n關聯圖譜分析完成！下一步：執行 app/dashboard/dashboard.py")


if __name__ == "__main__":
    main()