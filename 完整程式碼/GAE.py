# === 最佳化 GAE 模型以進一步降低 loss 至 < 0.2 並提升 AUC（改為 Ranking Loss） ===
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import gzip
import seaborn as sns
import gzip
from collections import defaultdict
import numpy as np
import community as community_louvain
import os

# === 讀取資料 ===
file_path = r"C:\Users\ROG Zephyrus\Downloads\archive\BIOGRID-ALL-4.3.195.tab3\BIOGRID-ALL-4.3.195.tab3.txt"
df = pd.read_csv(file_path, sep='\t', low_memory=False)

# === 資料前處理 ===
df_yeast = df[(df['Organism ID Interactor A'] == 559292) & (df['Organism ID Interactor B'] == 559292)].copy()
df_yeast['Gene_Pair'] = df_yeast.apply(lambda row: tuple(sorted([row['Official Symbol Interactor A'], row['Official Symbol Interactor B']])), axis=1)
df_yeast = df_yeast.drop_duplicates(subset='Gene_Pair').drop(columns='Gene_Pair')
df_yeast['Score'] = pd.to_numeric(df_yeast['Score'], errors='coerce')
edges = df_yeast[['Official Symbol Interactor A', 'Official Symbol Interactor B', 'Score']].dropna()

# === 建立圖與節點特徵 ===
G = nx.Graph()
for _, row in edges.iterrows():
    G.add_edge(row['Official Symbol Interactor A'], row['Official Symbol Interactor B'], weight=float(row['Score']))

node_mapping = {node: i for i, node in enumerate(G.nodes())}
reverse_mapping = {i: node for node, i in node_mapping.items()}
G = nx.relabel_nodes(G, node_mapping, copy=True)

# 特徵：degree, clustering, betweenness, eigenvector（移除 pagerank）
deg = dict(G.degree())
clustering = nx.clustering(G)
betweenness = nx.betweenness_centrality(G, k=50, seed=42)
eigen = nx.eigenvector_centrality_numpy(G)
kcore = nx.core_number(G) 
closeness = nx.closeness_centrality(G)
features = np.array([
    [deg[i], clustering[i], betweenness[i], eigen[i], kcore[i],closeness[i]]  #  增加 kcore[i]
    for i in G.nodes()
], dtype=np.float32)
features = np.nan_to_num(features, nan=0.0)
features = StandardScaler().fit_transform(features)

# === PyG 格式轉換 ===
data = from_networkx(G)
data.x = torch.tensor(features, dtype=torch.float)
data.edge_index = to_undirected(data.edge_index)
data.num_nodes = G.number_of_nodes()

transform = RandomLinkSplit(is_undirected=True, split_labels=True, add_negative_train_samples=True)
train_data, val_data, test_data = transform(data)

# === 模型定義 ===
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.norm1 = nn.LayerNorm(2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x, edge_index):
        x = F.relu(self.norm1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.norm2(self.conv2(x, edge_index)))
        return F.normalize(x, p=2, dim=1)

class DotDecoder(nn.Module):
    def forward(self, z, edge_index):
        z_i = z[edge_index[0]]
        z_j = z[edge_index[1]]
        return (z_i * z_j).sum(dim=1)

in_dim = train_data.x.shape[1]
hidden_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GAE(GCNEncoder(in_dim, hidden_dim), decoder=DotDecoder()).to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-6)
loss_fn = nn.MarginRankingLoss(margin=0.2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)


# === 訓練 ===
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    pos_pred = model.decoder(z, train_data.pos_edge_label_index)
    neg_pred = model.decoder(z, train_data.neg_edge_label_index)

    neg_sample_idx = torch.randperm(neg_pred.size(0))[:pos_pred.size(0)]
    neg_pred = neg_pred[neg_sample_idx]

    y = torch.ones(pos_pred.size(0), device=pos_pred.device)
    loss = loss_fn(pos_pred, neg_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    pos_pred = model.decoder(z, data.pos_edge_label_index)
    neg_pred = model.decoder(z, data.neg_edge_label_index)
    preds = torch.cat([pos_pred, neg_pred])
    labels = torch.cat([
        torch.ones(pos_pred.size(0), device=preds.device),
        torch.zeros(neg_pred.size(0), device=preds.device)
    ])
    probs = torch.sigmoid(preds)
    auc = roc_auc_score(labels.cpu(), probs.cpu())
    return auc, z

# === Early stopping 訓練流程 ===
best_auc = 0
wait = 0
patience = 25
for epoch in range(1, 301):
    loss = train()
    if epoch % 10 == 0:
        auc, z = test(test_data)
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}")
        scheduler.step(auc)  #  自動調整學習率
        if auc > best_auc and loss < 0.2:
            best_auc = auc
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

# === 預測新互作 ===
@torch.no_grad()
def predict_new_edges(z, existing_edges, top_k=20):
    # 將張量轉為 numpy 並正規化每個向量（L2 norm = 1）
    z_np = z.cpu().numpy()
    norms = np.linalg.norm(z_np, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # 避免除以零
    z_norm = z_np / norms

    # 計算 cosine similarity（範圍應在 [-1, 1]）
    similarity = cosine_similarity(z_norm)

    # 清除對角線（避免自己跟自己配對）
    np.fill_diagonal(similarity, 0)

    # 已知邊
    known_edges = set((u.item(), v.item()) for u, v in zip(*existing_edges.cpu()))

    pred_edges = []
    num_nodes = similarity.shape[0]

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) not in known_edges and (j, i) not in known_edges:
                raw_score = similarity[i, j]
                score = (raw_score + 1) / 2  # ✅ 把 [-1, 1] 轉換為 [0, 1]
                score = np.clip(score, 0, 1)  # 安全保險
                pred_edges.append((i, j, score))

    # 依照信心分數排序，選前 top_k
    pred_edges = sorted(pred_edges, key=lambda x: x[2], reverse=True)[:top_k]
    return pred_edges


predicted_edges = predict_new_edges(z, train_data.edge_index, top_k=300)

print("\nTop 20 predicted new interactions:")
for u, v, score in predicted_edges:
    print(f"{reverse_mapping[u]} - {reverse_mapping[v]} (score: {score:.4f})")

output_path = r".\\yeast_predicted_interactions_optimized.csv"
predicted_named_edges = [(reverse_mapping[u], reverse_mapping[v], score) for u, v, score in predicted_edges]
df_predicted = pd.DataFrame(predicted_named_edges, columns=['Protein1', 'Protein2', 'Confidence'])
df_predicted.to_csv(output_path, index=False)
print(f"\n Exported optimized predictions to {output_path}")

# === 讀取模型預測結果 ===
df = pd.read_csv("yeast_predicted_interactions_optimized.csv") 

# === 1. 繪製信心分數分布圖 ===
plt.figure(figsize=(8, 5))
plt.hist(df["Confidence"], bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Confidence Score")
plt.ylabel("Frequency")
plt.title("Predicted Interaction Confidence Distribution")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 2. 篩選高分預測互作（Confidence > 0.9） ===
high_conf_df = df[df["Confidence"] > 0.9]

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
plt.show()

# === 4. Louvain 社群偵測（Edge Confidence Clustering）===

# 用高信心邊建立圖（延續上一步的 high_conf_df）
G_louvain = nx.Graph()
for _, row in high_conf_df.iterrows():
    G_louvain.add_edge(row["Protein1"], row["Protein2"], weight=row["Confidence"])

# 執行 Louvain 社群分群（基於 confidence 權重）
partition = community_louvain.best_partition(G_louvain, weight='weight')
nx.set_node_attributes(G_louvain, partition, 'community')

# 社群數統計
num_communities = len(set(partition.values()))
print(f" Detected {num_communities} communities among high-confidence predicted interactions.")

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
plt.show()

# === 匯出社群結果 CSV 檔 ===
df_communities = pd.DataFrame([
    {"Protein": node, "Community": comm} for node, comm in partition.items()
])
df_communities.to_csv("yeast_predicted_communities.csv", index=False)
print(" Exported community assignments to 'yeast_predicted_communities.csv'")


# === 5. 繪製每個社群的蛋白質子圖 ===
# 建立每個社群對應的蛋白質清單
subgraphs = {}
for community in df_communities['Community'].unique():
    proteins = df_communities[df_communities['Community'] == community]['Protein'].tolist()
    subgraphs[community] = proteins

# 用原本的高置信度資料建立互作圖
G_full = nx.Graph()
for _, row in high_conf_df.iterrows():
    G_full.add_edge(row["Protein1"], row["Protein2"], weight=row["Confidence"])




# === 讀取 CSV 檔案（桌面）===
path_edges = r"C:\Users\ROG Zephyrus\Desktop\yeast_predicted_interactions_optimized.csv"
path_communities = r"C:\Users\ROG Zephyrus\Desktop\yeast_predicted_communities.csv"
df_edges = pd.read_csv(path_edges)
df_communities = pd.read_csv(path_communities)

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
plt.show()

##############################################################################################
# ===== Gae_string_swarmplot 目的是要 將 GAE 預測的蛋白質互作（PPI）與 STRING 資料庫進行比對======
# === 路徑設定 ===
pred_path = r"C:\Users\ROG Zephyrus\Desktop\yeast_predicted_interactions_optimized.csv"
alias_path = r"C:\Users\ROG Zephyrus\Downloads\4932.protein.aliases.v12.0.txt.gz"
link_path = r"C:\Users\ROG Zephyrus\Downloads\4932.protein.links.full.v12.0.txt.gz"

# === Step 1: 讀取 STRING alias（所有來源）建立反查表 ===
alias_map = defaultdict(list)
with gzip.open(alias_path, 'rt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            protein_id, alias = parts[0], parts[1]
            alias_map[alias].append(protein_id)

# === Step 2: 讀取 STRING 互作分數 ===
string_links = {}
with gzip.open(link_path, 'rt') as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            p1, p2, score = parts[0], parts[1], int(parts[-1])
            string_links[(p1, p2)] = score
            string_links[(p2, p1)] = score

# === Step 3: 載入 GAE 預測結果 ===
df_pred = pd.read_csv(pred_path)

# === Step 4: 定義查詢函式 ===
def get_string_id(symbol):
    return alias_map[symbol][0] if symbol in alias_map else None

# === Step 5: 建立比對結果 ===
string_results = []
for _, row in df_pred.iterrows():
    g1, g2, conf = row["Protein1"], row["Protein2"], row["Confidence"]
    pid1 = get_string_id(g1)
    pid2 = get_string_id(g2)
    if pid1 and pid2:
        score = string_links.get((pid1, pid2), 0)
    else:
        score = None
    string_results.append((g1, g2, conf, score))

# === 建立 DataFrame 並分類 ===
df_verified = pd.DataFrame(string_results, columns=["Protein1", "Protein2", "Confidence", "STRING_Score"])

def classify(row):
    if pd.isna(row["STRING_Score"]):
        return "Unmapped"
    elif row["STRING_Score"] >= 700:
        return "High-confidence"
    elif row["STRING_Score"] == 0:
        return "Not in STRING"
    else:
        return "Low-confidence"

df_verified["Match_Status"] = df_verified.apply(classify, axis=1)
df_verified["Match_Status"] = pd.Categorical(df_verified["Match_Status"], categories=[
    "High-confidence", "Low-confidence", "Not in STRING", "Unmapped"
])

# === 匯出資料 ===
df_verified.to_csv("yeast_predicted_with_STRING.csv", index=False)
novel_df = df_verified[df_verified["STRING_Score"] == 0]
novel_df.to_csv("unmatched_predicted_interactions.csv", index=False)

# === 畫 swarmplot ===
plt.figure(figsize=(12, 6))
sns.stripplot(
    data=df_verified,
    x="Match_Status",
    y="Confidence",
    hue="Match_Status",
    palette={
        "High-confidence": "green",
        "Low-confidence": "gray",
        "Not in STRING": "red",
        "Unmapped": "orange"
    },
    dodge=False,
    alpha=0.6,
    jitter=True
)

plt.title("GAE Confidence by STRING Match Category (Swarmplot)")
plt.ylabel("GAE Predicted Confidence")
plt.xlabel("STRING Match Category")
plt.legend(title="Match Status")
plt.tight_layout()
plt.grid(True, axis='y')
plt.show()

# ===========================================================

# 讀取社群與預測互作資料
df_comm = pd.read_csv("yeast_predicted_communities.csv")
df_pred = pd.read_csv("yeast_predicted_interactions_optimized.csv")

# 將蛋白質映射到社群
community_map = dict(zip(df_comm["Protein"], df_comm["Community"]))
df_pred["Community1"] = df_pred["Protein1"].map(community_map)
df_pred["Community2"] = df_pred["Protein2"].map(community_map)

# 移除沒有社群資訊的資料
df_pred = df_pred.dropna(subset=["Community1", "Community2"])

# 對稱處理：社群對社群的平均 confidence score
avg_scores = df_pred.groupby(["Community1", "Community2"])["Confidence"].mean().unstack().fillna(0)

# 補齊對稱矩陣
avg_scores = (avg_scores + avg_scores.T) / 2

# 計算階層式聚類（使用 Ward 連結方法）
linkage_matrix = linkage(avg_scores, method="ward")

# 繪製 Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=avg_scores.index, leaf_rotation=90)
plt.title("Hierarchical Clustering of Communities (by Avg Confidence)")
plt.xlabel("Community")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# 同時以 heatmap 顯示社群互作關係
plt.figure(figsize=(12, 10))
sns.clustermap(avg_scores, method="ward", cmap="YlGnBu", linewidths=0.5, figsize=(12, 10))
plt.title("Community-to-Community Avg Confidence Heatmap")
plt.show()

# 計算每個社群的總連結信心值總和（衡量其在網絡中的「活躍度」）
total_confidence_per_comm = avg_scores.sum(axis=1).sort_values(ascending=False)

# 取出前 5 高的社群
top_communities = total_confidence_per_comm.head(5).index.tolist()
print(f"💡 Top 5 connected communities: {top_communities}")

# 繪製他們與其他社群的連結強度
plt.figure(figsize=(10, 6))
sns.heatmap(avg_scores.loc[top_communities], cmap="YlOrBr", annot=True, fmt=".2f", linewidths=0.5)
plt.title("Top 5 Hub Communities vs Others (Avg Confidence)")
plt.xlabel("Target Community")
plt.ylabel("Hub Community")
plt.tight_layout()
plt.show()

#=============================================================
#### 將需要做GO_enrichment的資料先處理 #######
# === 路徑設定 ===
input_path = r"C:\Users\ROG Zephyrus\Desktop\yeast_predicted_communities.csv"
output_dir = r"C:\Users\ROG Zephyrus\Desktop\split_communities"

# === 建立輸出資料夾 ===
os.makedirs(output_dir, exist_ok=True)

# === 讀取資料 ===
df = pd.read_csv(input_path)

# === 拆分並輸出 ===
for community in sorted(df["Community"].unique()):
    proteins = df[df["Community"] == community]["Protein"]
    filename = f"community_{community}.txt"
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w") as f:
        for protein in proteins:
            f.write(f"{protein}\n")

print(f" 已將社群蛋白質名單輸出至：{output_dir}")

###########################################################################

# === 參數設定 ===
input_folder = r"C:\Users\ROG Zephyrus\Desktop\split_communities"
output_folder = r"C:\Users\ROG Zephyrus\Desktop\go_results"
os.makedirs(output_folder, exist_ok=True)

# === 初始化 g:Profiler ===
gp = GProfiler(return_dataframe=True)

# === 記錄每個社群的 enrichment 狀態 ===
summary = []

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        path = os.path.join(input_folder, filename)
        with open(path, 'r') as f:
            genes = [line.strip() for line in f.readlines() if line.strip()]
        
        community_id = filename.replace(".txt", "")
        
        if len(genes) < 5:
            print(f" Skip {filename}: too few genes ({len(genes)})")
            summary.append({
                "Community": community_id,
                "Num Genes": len(genes),
                "Status": "Skipped (too few genes)",
                "Output File": ""
            })
            continue

        # 執行 GO和KEGG的enrichment 分析
        try:
            result = gp.profile(
                organism='scerevisiae',  # yeast
                query=genes,
                sources=["GO:BP", "GO:MF", "GO:CC","KEGG"]  # 生物程序、分子功能、細胞構造、KEGG
            )

            if result.empty:
                print(f" No enrichment results for {filename}")
                summary.append({
                    "Community": community_id,
                    "Num Genes": len(genes),
                    "Status": "No enrichment found",
                    "Output File": ""
                })
            else:
                out_path = os.path.join(output_folder, f"{community_id}_enrichment.csv")
                result.to_csv(out_path, index=False)
                print(f" Enrichment saved for {community_id} → {out_path}")
                summary.append({
                    "Community": community_id,
                    "Num Genes": len(genes),
                    "Status": "Enriched",
                    "Output File": out_path
                })

        except Exception as e:
            print(f" Error processing {filename}: {e}")
            summary.append({
                "Community": community_id,
                "Num Genes": len(genes),
                "Status": f"Error: {str(e)}",
                "Output File": ""
            })

# === 儲存總結 ===
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(output_folder, "enrichment_summary.csv"), index=False)
print("\n Summary saved to enrichment_summary.csv")

#######################################################################################
#    自動畫成bubble plot #
# === 設定目錄 ===
# === 自動畫成 bubble plot（GO 與 KEGG 分開繪製）===
input_folder = r"C:\Users\ROG Zephyrus\Desktop\go_results"
output_folder = os.path.join(input_folder, "plots_all")
os.makedirs(output_folder, exist_ok=True)

# === 遍歷每個 enrichment 檔案並繪圖 ===
for file in os.listdir(input_folder):
    if file.endswith("_enrichment.csv"):
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)

        required_cols = {"p_value", "name", "intersection_size", "source"}
        if not required_cols.issubset(df.columns):
            print(f" Skipping {file}: Missing required columns.")
            continue

        # === 分別處理 GO 與 KEGG ===
        for source_label, title_prefix in [("GO", "GO Enrichment"), ("KEGG", "KEGG Pathway Enrichment")]:
            df_sub = df[df["source"].str.startswith(source_label)]
            if df_sub.empty:
                continue

            # 計算 -log10(p_value)
            df_sub["-log10(p_value)"] = -np.log10(df_sub["p_value"].replace(0, 1e-300))  # 避免 log(0)

            plt.figure(figsize=(10, max(6, 0.4 * len(df_sub))))  # 圖高自適應
            scatter = sns.scatterplot(
                data=df_sub,
                x="-log10(p_value)",
                y="name",
                size="intersection_size",
                sizes=(50, 300),
                hue="-log10(p_value)",
                palette="coolwarm",
                legend="brief"
            )
            plt.title(f"{title_prefix} Bubble Plot: {file.replace('_enrichment.csv', '')}")
            plt.xlabel("-log10(p-value)")
            plt.ylabel("GO Term" if source_label == "GO" else "KEGG Pathway")
            plt.legend(title="Gene Count", loc="lower right", bbox_to_anchor=(1.25, 0))
            plt.tight_layout()
            plt.grid(True, axis='x')

            # 儲存圖片
            suffix = "_GO_all.png" if source_label == "GO" else "_KEGG_all.png"
            plot_filename = file.replace(".csv", suffix)
            plt.savefig(os.path.join(output_folder, plot_filename))
            plt.close()

            print(f" Saved {source_label} plot for {file} to {plot_filename}")
