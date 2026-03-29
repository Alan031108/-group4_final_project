import pandas as pd
import networkx as nx
import torch
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.transforms import RandomLinkSplit
from sklearn.preprocessing import StandardScaler
import numpy as np


# === 讀取資料 ===
def read_biogrid_data(file_path):
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    return df

# === 資料前處理 ===
def preprocess(df):
    df_yeast = df[(df['Organism ID Interactor A'] == 559292) & (df['Organism ID Interactor B'] == 559292)].copy()
    df_yeast['Gene_Pair'] = df_yeast.apply(lambda row: tuple(sorted([row['Official Symbol Interactor A'], row['Official Symbol Interactor B']])), axis=1)
    df_yeast = df_yeast.drop_duplicates(subset='Gene_Pair').drop(columns='Gene_Pair')
    df_yeast['Score'] = pd.to_numeric(df_yeast['Score'], errors='coerce')
    edges = df_yeast[['Official Symbol Interactor A', 'Official Symbol Interactor B', 'Score']].dropna()
    return edges

# === 建立圖與節點特徵 ===
def build_graph(edges):
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
        [deg[i], clustering[i], betweenness[i], eigen[i], kcore[i],closeness[i]]  # 增加 kcore[i]
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

    return reverse_mapping, train_data, val_data, test_data
