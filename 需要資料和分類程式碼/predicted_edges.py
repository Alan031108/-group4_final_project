import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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
                score = (raw_score + 1) / 2  #  把 [-1, 1] 轉換為 [0, 1]
                score = np.clip(score, 0, 1)  # 安全保險
                pred_edges.append((i, j, score))

    # 依照信心分數排序，選前 top_k
    pred_edges = sorted(pred_edges, key=lambda x: x[2], reverse=True)[:top_k]
    return pred_edges

def predict(output_path, z, train_data, reverse_mapping):
    predicted_edges = predict_new_edges(z, train_data.edge_index, top_k=300)

    print("\nTop 20 predicted new interactions:")
    for u, v, score in predicted_edges:
        print(f"{reverse_mapping[u]} - {reverse_mapping[v]} (score: {score:.4f})")

    predicted_named_edges = [(reverse_mapping[u], reverse_mapping[v], score) for u, v, score in predicted_edges]
    df_predicted = pd.DataFrame(predicted_named_edges, columns=['Protein1', 'Protein2', 'Confidence'])
    df_predicted.to_csv(output_path, index=False)
    print(f"\n Exported optimized predictions to {output_path}")
