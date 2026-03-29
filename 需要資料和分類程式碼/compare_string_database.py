import pandas as pd
import matplotlib.pyplot as plt
import gzip
import seaborn as sns
import gzip
from collections import defaultdict

def compare_string_database(pred_path, alias_path, link_path, predicted_string_output_path, unmatched_output_path):
    # ===== Gae_string_swarmplot 目的是要 將 GAE 預測的蛋白質互作（PPI）與 STRING 資料庫進行比對======
    # === 路徑設定 ===

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
    df_verified.to_csv(predicted_string_output_path, index=False)
    novel_df = df_verified[df_verified["STRING_Score"] == 0]
    novel_df.to_csv(unmatched_output_path, index=False)

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
