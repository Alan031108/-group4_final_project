import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain


def filter_by_confidence(df, threshold=0.9):
    return df[df["Confidence"] > threshold]

# === 1. 繪製信心分數分布圖 ===
def draw_confidence_distribution(df, image_output):
    plt.figure(figsize=(8, 5))
    plt.hist(df["Confidence"], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Predicted Interaction Confidence Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_output)
    plt.show()

def draw_high_score_prediction(df, image_output):
    # === 2. 篩選高分預測互作（Confidence > 0.9） ===
    high_conf_df = filter_by_confidence(df)

    # === 3. 繪製高分預測互作網路圖 ===
    G = nx.Graph()
    for _, row in high_conf_df.iterrows():
        G.add_edge(row["Protein1"], row["Protein2"], weight=row["Confidence"])

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightgreen')
    nx.draw_networkx_edges(G, pos, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title("High Confidence Predicted Interactions (Confidence > 0.9)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(image_output)
    plt.show()

def louvain(df, file_path, image_output):
# === 4. Louvain 社群偵測（Edge Confidence Clustering）===
# 用高信心邊建立圖（延續上一步的 high_conf_df）
    G_louvain = nx.Graph()
    high_conf_df = filter_by_confidence(df)
    for _, row in high_conf_df.iterrows():
        G_louvain.add_edge(row["Protein1"], row["Protein2"], weight=row["Confidence"])

    # 執行 Louvain 社群分群（基於 confidence 權重）
    partition = community_louvain.best_partition(G_louvain, weight='weight')
    nx.set_node_attributes(G_louvain, partition, 'community')

    # 社群數統計
    num_communities = len(set(partition.values()))
    print(f"🧩 Detected {num_communities} communities among high-confidence predicted interactions.")

    # === 視覺化社群圖 ===
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_louvain, seed=42, k=0.15)
    node_colors = [partition[node] for node in G_louvain.nodes()]
    nx.draw_networkx_nodes(G_louvain, pos, node_size=300, node_color=node_colors, cmap=plt.cm.Set3)
    nx.draw_networkx_edges(G_louvain, pos, alpha=0.5)
    nx.draw_networkx_labels(G_louvain, pos, font_size=8)
    plt.title("Louvain Communities in High-Confidence Predicted PPI Network")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(image_output)
    plt.show()

    # === 匯出社群結果 CSV 檔 ===
    df_communities = pd.DataFrame([
        {"Protein": node, "Community": comm} for node, comm in partition.items()
    ])
    df_communities.to_csv(file_path, index=False)
    print("✅ Exported community assignments to 'yeast_predicted_communities.csv'")

def draw_community_protein(df, path_edges, path_communities, image_output):
    # === 5. 繪製每個社群的蛋白質子圖 ===
    df_communities = pd.read_csv(path_communities)

    # 建立每個社群對應的蛋白質清單
    subgraphs = {}
    for community in df_communities['Community'].unique():
        proteins = df_communities[df_communities['Community'] == community]['Protein'].tolist()
        subgraphs[community] = proteins

    # 用原本的高置信度資料建立互作圖
    high_conf_df = filter_by_confidence(df)
    G_full = nx.Graph()
    for _, row in high_conf_df.iterrows():
        G_full.add_edge(row["Protein1"], row["Protein2"], weight=row["Confidence"])

    # === 讀取 CSV 檔案）===
    df_edges = pd.read_csv(path_edges)

    # === 建立網絡圖 ===
    G = nx.Graph()
    for _, row in df_edges.iterrows():
        G.add_edge(row["Protein1"], row["Protein2"], weight=row["Confidence"])

    # === 加入社群屬性 ===
    community_map = dict(zip(df_communities["Protein"], df_communities["Community"]))
    nx.set_node_attributes(G, community_map, "community")

    # === 建立節點顏色 ===
    node_colors = [community_map.get(node, -1) for node in G.nodes()]

    # === Layout（避免重疊）===
    pos = nx.kamada_kawai_layout(G)

    # === 找出 degree 前 5 高的 hub nodes ===
    degree_dict = dict(G.degree())
    top5_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:5]

    # 先定義 hub 與非 hub 節點的 label 對應字典
    non_hubs = [node for node in G.nodes() if node not in top5_nodes]
    labels_non_hub = {node: node for node in non_hubs}
    labels_hub = {node: node for node in top5_nodes}

    # 開始畫圖
    plt.figure(figsize=(14, 12))

    # 畫節點，大小依據是否為 hub
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[600 if node in top5_nodes else 300 for node in G.nodes()],
        node_color=node_colors,
        cmap=plt.cm.Set3
    )

    # 畫邊
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=labels_non_hub, font_size=8)
    nx.draw_networkx_labels(G, pos, labels=labels_hub, font_size=10, font_weight='bold')
    plt.title("All Predicted PPI with Louvain Communities\n(Hub nodes highlighted)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(image_output)
    plt.show()
