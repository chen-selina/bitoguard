"""
graph_analysis.py
位置：src/models/graph_analysis.py

【v2 修正說明 — 根據新版欄位說明文件】
  1. IP 欄位：source_ip → source_ip_hash（MD5）
     - 相同 hash = 相同 IP，仍可建立「共用 IP」邊
     - 無法還原原始 IP，移除 int_to_ip 相關邏輯
  2. 錢包欄位：from_wallet/to_wallet → from_wallet_hash/to_wallet_hash
     - 相同 hash = 相同錢包，仍可建立「共用錢包」邊
  3. 黑名單來源：user_info.status → train_label.status
  4. 圖特徵合併目標：features.csv（訓練集）和 predict_features.csv（預測集）

使用方式：python src/models/graph_analysis.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ============================================================
# 路徑設定
# ============================================================
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


# ============================================================
# 1. 載入原始資料
# ============================================================

def load_raw_data() -> dict:
    tables = {}
    files = {
        "user_info":    "user_info.csv",
        "twd":          "twd_transfer.csv",
        "crypto":       "crypto_transfer.csv",
        "usdt_trade":   "usdt_twd_trading.csv",
        "train_label":  "train_label.csv",   # 【v2】黑名單標籤獨立資料表
        "predict_label":"predict_label.csv", # 【v2】預測集
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


def get_blacklist_users(train_label: pd.DataFrame) -> set:
    """
    【v2 修正】黑名單來自 train_label.status == 1
    不再從 user_info.status 讀取
    """
    if train_label.empty or "status" not in train_label.columns:
        return set()
    bl = set(train_label[train_label["status"] == 1]["user_id"].tolist())
    print(f"  已知黑名單用戶（train_label）：{len(bl)} 位")
    return bl


def get_all_known_users(train_label: pd.DataFrame,
                         predict_label: pd.DataFrame) -> set:
    """訓練集 + 預測集的所有 user_id"""
    uids = set()
    if not train_label.empty:
        uids |= set(train_label["user_id"].tolist())
    if not predict_label.empty:
        uids |= set(predict_label["user_id"].tolist())
    print(f"  已知用戶總數（訓練+預測）：{len(uids):,} 位")
    return uids


# ============================================================
# 2. 建立關聯圖
# 【v2 修正】IP 欄位改為 source_ip_hash，錢包欄位改為 *_hash
# ============================================================

def build_graph(tables: dict, blacklist: set) -> nx.Graph:
    """
    節點 = 用戶（user_id）
    邊的種類：
      - shared_ip     ：相同 source_ip_hash（= 相同 IP）
      - fund_transfer ：crypto_transfer 內轉（sub_kind=1）
      - same_wallet   ：相同 to_wallet_hash（= 相同接收錢包）

    【v2】IP 和錢包皆為 MD5 hash，
          雖無法還原真實值，但「hash 相同 = 同一個來源」的邏輯仍成立。
    """
    G = nx.Graph()

    user_info    = tables.get("user_info",    pd.DataFrame())
    crypto       = tables.get("crypto",       pd.DataFrame())
    usdt_trade   = tables.get("usdt_trade",   pd.DataFrame())
    twd          = tables.get("twd",          pd.DataFrame())
    train_label  = tables.get("train_label",  pd.DataFrame())
    predict_label= tables.get("predict_label",pd.DataFrame())

    # ── 加入所有用戶節點 ──
    # 節點來自 user_info（含基本屬性）
    if not user_info.empty:
        for _, row in user_info.iterrows():
            uid   = row["user_id"]
            is_bl = uid in blacklist
            G.add_node(
                uid,
                is_blacklist=is_bl,
                # 【v2】age 直接使用，無 birthday
                age=int(row.get("age", 0)) if pd.notna(row.get("age")) else 0,
                career=int(row.get("career", 0)),
                kyc_level=(2 if pd.notna(row.get("level2_finished_at"))
                           else 1 if pd.notna(row.get("level1_finished_at"))
                           else 0),
                in_train=uid in (set(train_label["user_id"]) if not train_label.empty else set()),
                in_predict=uid in (set(predict_label["user_id"]) if not predict_label.empty else set()),
            )

    # ── 邊 1：共用 IP（source_ip_hash）──
    # 【v2】欄位名稱：source_ip → source_ip_hash
    ip_col = "source_ip_hash"
    print("  建立 shared_ip 邊（based on source_ip_hash）...")
    ip_edges = 0
    ip_sources = []
    for df in [twd, crypto, usdt_trade]:
        if not df.empty and ip_col in df.columns and "user_id" in df.columns:
            ip_sources.append(
                df[["user_id", ip_col]].dropna(subset=[ip_col])
            )

    if ip_sources:
        all_ip    = pd.concat(ip_sources, ignore_index=True)
        ip_groups = all_ip.groupby(ip_col)["user_id"].apply(list)
        for ip_hash, users in ip_groups.items():
            unique_users = list(set(users))
            if len(unique_users) < 2:
                continue
            for i in range(len(unique_users)):
                for j in range(i + 1, len(unique_users)):
                    u1, u2 = unique_users[i], unique_users[j]
                    if G.has_edge(u1, u2):
                        G[u1][u2]["weight"]     += 1
                        G[u1][u2]["shared_ips"] += 1
                    else:
                        G.add_edge(u1, u2, weight=1, edge_type="shared_ip",
                                   shared_ips=1, fund_transfers=0, same_wallets=0)
                        ip_edges += 1
    print(f"    shared_ip 邊：{ip_edges:,} 條")

    # ── 邊 2：資金流向（內轉）──
    print("  建立 fund_transfer 邊...")
    fund_edges = 0
    if not crypto.empty and "relation_user_id" in crypto.columns:
        internal = crypto[
            (crypto["sub_kind"] == 1) & crypto["relation_user_id"].notna()
        ][["user_id", "relation_user_id"]].copy()
        internal["relation_user_id"] = internal["relation_user_id"].astype(int)

        transfer_counts = (
            internal.groupby(["user_id", "relation_user_id"])
            .size().reset_index(name="count")
        )
        for _, row in transfer_counts.iterrows():
            u1, u2, cnt = int(row["user_id"]), int(row["relation_user_id"]), int(row["count"])
            if u1 == u2:
                continue
            if G.has_edge(u1, u2):
                G[u1][u2]["weight"]         += cnt
                G[u1][u2]["fund_transfers"] += cnt
            else:
                G.add_edge(u1, u2, weight=cnt, edge_type="fund_transfer",
                           shared_ips=0, fund_transfers=cnt, same_wallets=0)
                fund_edges += 1
    print(f"    fund_transfer 邊：{fund_edges:,} 條")

    # ── 邊 3：共用錢包地址（to_wallet_hash）──
    # 【v2】欄位名稱：to_wallet → to_wallet_hash
    print("  建立 same_wallet 邊（based on to_wallet_hash）...")
    wallet_edges = 0
    if not crypto.empty:
        # 優先用新欄位名，若無則嘗試舊欄位名
        wallet_cols = []
        for candidate in ["from_wallet_hash", "to_wallet_hash", "from_wallet", "to_wallet"]:
            if candidate in crypto.columns:
                wallet_cols.append(candidate)

        for wcol in wallet_cols:
            w_groups = (
                crypto[["user_id", wcol]]
                .dropna(subset=[wcol])
                .groupby(wcol)["user_id"].apply(list)
            )
            for wallet_hash, users in w_groups.items():
                unique_users = list(set(users))
                if len(unique_users) < 2:
                    continue
                for i in range(len(unique_users)):
                    for j in range(i + 1, len(unique_users)):
                        u1, u2 = unique_users[i], unique_users[j]
                        if G.has_edge(u1, u2):
                            G[u1][u2]["weight"]       += 1
                            G[u1][u2]["same_wallets"] += 1
                        else:
                            G.add_edge(u1, u2, weight=1, edge_type="same_wallet",
                                       shared_ips=0, fund_transfers=0, same_wallets=1)
                            wallet_edges += 1
    print(f"    same_wallet 邊：{wallet_edges:,} 條")
    print(f"  圖建立完成：{G.number_of_nodes():,} 節點，{G.number_of_edges():,} 邊")
    return G


# ============================================================
# 3. 計算圖特徵
# ============================================================

def compute_graph_features(G: nx.Graph, blacklist: set) -> pd.DataFrame:
    print("  計算節點圖特徵...")
    pagerank   = nx.pagerank(G, weight="weight", max_iter=200)
    clustering = nx.clustering(G, weight="weight")

    rows = []
    for uid in G.nodes():
        neighbors = list(G.neighbors(uid))
        degree    = len(neighbors)

        bl_neighbors = sum(1 for n in neighbors if n in blacklist)

        hop2_nodes = set()
        for n in neighbors:
            hop2_nodes.update(G.neighbors(n))
        hop2_nodes.discard(uid)
        hop2_nodes -= set(neighbors)
        bl_2hop = sum(1 for n in hop2_nodes if n in blacklist)

        component = nx.node_connected_component(G, uid)
        bl_in_component = sum(1 for n in component if n in blacklist)

        rows.append({
            "user_id":            uid,
            "graph_degree":       degree,
            "graph_bl_neighbors": bl_neighbors,
            "graph_bl_2hop":      bl_2hop,
            "graph_bl_component": bl_in_component,
            "graph_clustering":   round(clustering.get(uid, 0), 4),
            "graph_pagerank":     round(pagerank.get(uid, 0), 6),
            "is_blacklist":       1 if uid in blacklist else 0,
        })

    feat_df = pd.DataFrame(rows)
    print(f"  圖特徵完成：{len(feat_df)} 位用戶")
    return feat_df


# ============================================================
# 4. 找出高風險關聯群組
# ============================================================

def find_risk_communities(G: nx.Graph, blacklist: set) -> list:
    print("  識別高風險群組...")
    risk_groups = []

    for bl_uid in blacklist:
        if bl_uid not in G:
            continue

        hop1 = set(G.neighbors(bl_uid))
        hop2 = set()
        for n in hop1:
            hop2.update(G.neighbors(n))
        hop2 -= hop1
        hop2.discard(bl_uid)

        if not hop1 and not hop2:
            continue

        other_bl     = (hop1 | hop2) & blacklist - {bl_uid}
        sg           = G.subgraph({bl_uid} | hop1 | hop2)
        total_weight = sum(d["weight"] for _, _, d in sg.edges(data=True))

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


# ============================================================
# 5. 視覺化：全局圖譜
# ============================================================

def plot_full_network(G: nx.Graph, blacklist: set, max_nodes: int = 300) -> None:
    print("  繪製全局圖譜...")

    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_nodes = list(set([n for n, _ in top_nodes]) | (blacklist & set(G.nodes())))
        subG = G.subgraph(top_nodes[:max_nodes + len(blacklist)])
    else:
        subG = G

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(subG, weight="weight", k=2, seed=42)

    node_colors, node_sizes = [], []
    for n in subG.nodes():
        if n in blacklist:
            node_colors.append("#E24B4A"); node_sizes.append(300)
        elif any(nb in blacklist for nb in subG.neighbors(n)):
            node_colors.append("#EF9F27"); node_sizes.append(120)
        else:
            node_colors.append("#378ADD"); node_sizes.append(40)

    edge_colors, edge_widths = [], []
    for u, v, d in subG.edges(data=True):
        etype = d.get("edge_type", "")
        if etype == "fund_transfer":
            edge_colors.append("#A32D2D")
            edge_widths.append(min(d.get("weight", 1) * 0.5, 3))
        elif etype == "shared_ip":
            edge_colors.append("#EF9F27"); edge_widths.append(0.8)
        else:
            edge_colors.append("#B4B2A9"); edge_widths.append(0.5)

    nx.draw_networkx_nodes(subG, pos, node_color=node_colors,
                           node_size=node_sizes, ax=ax, alpha=0.85)
    nx.draw_networkx_edges(subG, pos, edge_color=edge_colors,
                           width=edge_widths, ax=ax, alpha=0.6)
    bl_labels = {n: str(n) for n in subG.nodes() if n in blacklist}
    nx.draw_networkx_labels(subG, pos, labels=bl_labels,
                            font_size=7, font_color="white", ax=ax)

    legend_handles = [
        mpatches.Patch(color="#E24B4A", label="黑名單用戶"),
        mpatches.Patch(color="#EF9F27", label="直接關聯黑名單"),
        mpatches.Patch(color="#378ADD", label="一般用戶"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10)
    ax.set_title(
        f"BitoGuard 用戶關聯圖譜\n"
        f"（節點：{subG.number_of_nodes()}，邊：{subG.number_of_edges()}，"
        f"黑名單：{len(blacklist & set(subG.nodes()))}）",
        fontsize=13, pad=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(GRAPH_FULL_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  全局圖譜：{GRAPH_FULL_PLOT}")


# ============================================================
# 6. 視覺化：黑名單鄰域圖
# ============================================================

def plot_blacklist_neighborhood(G: nx.Graph, blacklist: set, top_n: int = 5) -> None:
    print("  繪製黑名單鄰域圖...")

    bl_in_graph = [(n, G.degree(n)) for n in blacklist if n in G]
    bl_in_graph.sort(key=lambda x: x[1], reverse=True)
    top_bl = [n for n, _ in bl_in_graph[:top_n]]

    if not top_bl:
        print("  黑名單在圖中無邊，跳過")
        return

    cols = min(len(top_bl), 3)
    rows = (len(top_bl) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if len(top_bl) == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for i, bl_uid in enumerate(top_bl):
        ax = axes[i // cols][i % cols]
        hop1 = set(G.neighbors(bl_uid))
        hop2 = set()
        for n in hop1:
            hop2.update(G.neighbors(n))
        hop2 -= hop1
        hop2.discard(bl_uid)

        subG = G.subgraph({bl_uid} | hop1 | hop2)
        pos  = nx.spring_layout(subG, weight="weight", seed=42)

        n_colors, n_sizes = [], []
        for n in subG.nodes():
            if n == bl_uid:
                n_colors.append("#E24B4A"); n_sizes.append(500)
            elif n in blacklist:
                n_colors.append("#A32D2D"); n_sizes.append(300)
            elif n in hop1:
                n_colors.append("#EF9F27"); n_sizes.append(150)
            else:
                n_colors.append("#B5D4F4"); n_sizes.append(60)

        e_colors = [
            "#A32D2D" if d.get("edge_type") == "fund_transfer"
            else "#EF9F27" if d.get("edge_type") == "shared_ip"
            else "#B4B2A9"
            for _, _, d in subG.edges(data=True)
        ]

        nx.draw_networkx_nodes(subG, pos, node_color=n_colors,
                               node_size=n_sizes, ax=ax, alpha=0.9)
        nx.draw_networkx_edges(subG, pos, edge_color=e_colors,
                               ax=ax, alpha=0.7, width=1.2)
        nx.draw_networkx_labels(subG, pos,
                                labels={n: str(n) for n in subG.nodes()},
                                font_size=6, ax=ax)
        ax.set_title(
            f"黑名單 {bl_uid}\n1跳：{len(hop1)} 人  2跳：{len(hop2)} 人",
            fontsize=10,
        )
        ax.axis("off")

    for i in range(len(top_bl), rows * cols):
        axes[i // cols][i % cols].axis("off")

    legend_handles = [
        mpatches.Patch(color="#E24B4A", label="目標黑名單"),
        mpatches.Patch(color="#A32D2D", label="其他黑名單"),
        mpatches.Patch(color="#EF9F27", label="1 跳關聯"),
        mpatches.Patch(color="#B5D4F4", label="2 跳關聯"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0))
    fig.suptitle("黑名單用戶 2 層關聯圖（共犯結構分析）", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(GRAPH_BLACKLIST_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  黑名單鄰域圖：{GRAPH_BLACKLIST_PLOT}")


# ============================================================
# 7. 存檔 & 合併圖特徵
# ============================================================

def save_results(graph_feat: pd.DataFrame, risk_groups: list, G: nx.Graph) -> None:
    graph_feat.to_csv(GRAPH_REPORT_PATH, index=False, encoding="utf-8-sig")
    print(f"  圖特徵 CSV：{GRAPH_REPORT_PATH}")

    serializable = []
    for g in risk_groups[:50]:
        row = {k: (list(v) if isinstance(v, (set, list)) else v)
               for k, v in g.items()}
        serializable.append(row)
    with open(GRAPH_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"  風險群組 JSON：{GRAPH_JSON_PATH}")

    G_export = G.copy()
    for n, d in G_export.nodes(data=True):
        for k, v in d.items():
            G_export.nodes[n][k] = str(v)
    nx.write_gexf(G_export, GRAPH_GEPHI_PATH)
    print(f"  Gephi 匯出：{GRAPH_GEPHI_PATH}")


def merge_graph_features_to_processed(graph_feat: pd.DataFrame) -> None:
    """
    【v2】同時合併至 features.csv（訓練集）和 predict_features.csv（預測集）
    """
    graph_cols  = [c for c in graph_feat.columns if c.startswith("graph_")]
    merge_cols  = ["user_id"] + graph_cols

    for fname in ["features.csv", "predict_features.csv"]:
        feat_path = PROCESSED_DIR / fname
        if not feat_path.exists():
            continue
        feat_df = pd.read_csv(feat_path, low_memory=False)
        feat_df = feat_df.merge(graph_feat[merge_cols], on="user_id", how="left")
        feat_df[graph_cols] = feat_df[graph_cols].fillna(0)
        feat_df.to_csv(feat_path, index=False, encoding="utf-8-sig")
        print(f"  已將 {len(graph_cols)} 個圖特徵合併至 {feat_path.name}")


# ============================================================
# 8. 主程式
# ============================================================

def main():
    print("=" * 55)
    print("BitoGuard v2 — 關聯圖譜分析開始")
    print("=" * 55 + "\n")

    print("【載入原始資料】")
    tables    = load_raw_data()
    # 【v2】黑名單從 train_label 取得
    blacklist = get_blacklist_users(tables.get("train_label", pd.DataFrame()))
    all_users = get_all_known_users(
        tables.get("train_label", pd.DataFrame()),
        tables.get("predict_label", pd.DataFrame()),
    )
    print()

    print("【建立關聯圖】")
    G = build_graph(tables, blacklist)
    print()

    print("【計算圖特徵】")
    graph_feat = compute_graph_features(G, blacklist)
    print()

    print("【識別高風險群組】")
    risk_groups = find_risk_communities(G, blacklist)
    if risk_groups:
        print(f"\n  Top 5 風險群組：")
        for i, rg in enumerate(risk_groups[:5], 1):
            print(f"  #{i}  黑名單中心：{rg['center_uid']}"
                  f"  1跳：{rg['hop1_count']} 人"
                  f"  2跳：{rg['hop2_count']} 人"
                  f"  群內其他黑名單：{rg['other_bl_count']} 位")
    print()

    print("【生成圖譜視覺化】")
    plot_full_network(G, blacklist, max_nodes=300)
    plot_blacklist_neighborhood(G, blacklist, top_n=5)
    print()

    print("【存檔】")
    save_results(graph_feat, risk_groups, G)

    print("\n【合併圖特徵至 features.csv & predict_features.csv】")
    merge_graph_features_to_processed(graph_feat)

    # 統計摘要
    print("\n" + "=" * 55)
    print("【圖譜統計摘要】")
    print("=" * 55)
    print(f"  總節點數  ：{G.number_of_nodes():,}")
    print(f"  總邊數    ：{G.number_of_edges():,}")
    print(f"  黑名單節點：{len(blacklist & set(G.nodes())):,}")

    bl_degrees = [G.degree(n) for n in blacklist if n in G]
    if bl_degrees:
        print(f"  黑名單平均連接度：{np.mean(bl_degrees):.1f}")
        print(f"  黑名單最高連接度：{max(bl_degrees)}")

    suspects = graph_feat[
        (graph_feat["is_blacklist"] == 0) &
        (graph_feat["graph_bl_neighbors"] > 0)
    ]
    print(f"  直接連到黑名單的正常用戶：{len(suspects):,} 位（潛在人頭戶）")

    suspects_2hop = graph_feat[
        (graph_feat["is_blacklist"] == 0) &
        (graph_feat["graph_bl_2hop"] > 0)
    ]
    print(f"  2 層內有黑名單的正常用戶：{len(suspects_2hop):,} 位")

    print("\n關聯圖譜分析完成！")
    print("下一步：執行 app/dashboard/dashboard.py")


if __name__ == "__main__":
    main()