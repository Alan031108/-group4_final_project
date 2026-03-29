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