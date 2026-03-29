import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from gprofiler import GProfiler
import numpy as np


#### 將需要做GO_enrichment的資料先處理 #######
def split_communities(df, output_dir):
    # === 拆分並輸出 ===
    for community in sorted(df["Community"].unique()):
        proteins = df[df["Community"] == community]["Protein"]
        filename = f"community_{community}.txt"
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "w") as f:
            for protein in proteins:
                f.write(f"{protein}\n")

    print(f"✅ 已將社群蛋白質名單輸出至：{output_dir}")

def go_enrichment(input_folder, output_folder):
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
                print(f"⚠️ Skip {filename}: too few genes ({len(genes)})")
                summary.append({
                    "Community": community_id,
                    "Num Genes": len(genes),
                    "Status": "Skipped (too few genes)",
                    "Output File": ""
                })
                continue

            # 執行 GO enrichment 分析
            try:
                result = gp.profile(
                    organism='scerevisiae',  # yeast
                    query=genes,
                    sources=["GO:BP", "GO:MF", "GO:CC"]  # 生物程序、分子功能、細胞構造
                )

                if result.empty:
                    print(f"❌ No enrichment results for {filename}")
                    summary.append({
                        "Community": community_id,
                        "Num Genes": len(genes),
                        "Status": "No enrichment found",
                        "Output File": ""
                    })
                else:
                    out_path = os.path.join(output_folder, f"{community_id}_enrichment.csv")
                    result.to_csv(out_path, index=False)
                    print(f"✅ Enrichment saved for {community_id} → {out_path}")
                    summary.append({
                        "Community": community_id,
                        "Num Genes": len(genes),
                        "Status": "Enriched",
                        "Output File": out_path
                    })

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                summary.append({
                    "Community": community_id,
                    "Num Genes": len(genes),
                    "Status": f"Error: {str(e)}",
                    "Output File": ""
                })

    # === 儲存總結 ===
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_folder, r".\enrichment_summary.csv"), index=False)
    print("\n📄 Summary saved to enrichment_summary.csv")

def draw_bubble_plot(input_folder, output_folder):
    #    自動畫成bubble plot #

    # === 遍歷所有 enrichment 檔案 ===
    for file in os.listdir(input_folder):
        if file.endswith(r".\_enrichment.csv"):
            file_path = os.path.join(input_folder, file)
            df = pd.read_csv(file_path)

            # 檢查欄位是否齊全
            required_cols = {"p_value", "name", "intersection_size"}
            if not required_cols.issubset(df.columns):
                print(f"⚠️ Skipping {file}: Missing columns.")
                continue

            # 新增 log10 p-value 欄位
            df["-log10(p_value)"] = -np.log10(df["p_value"])
            df_sorted = df.sort_values("p_value").head(20)

            # 畫 bubble plot
            plt.figure(figsize=(10, 8))
            scatter = sns.scatterplot(
                data=df_sorted,
                x="-log10(p_value)",
                y="name",
                size="intersection_size",
                sizes=(50, 300),
                hue="-log10(p_value)",
                palette="coolwarm",
                legend="brief"
            )
            plt.title(f"GO Enrichment Bubble Plot: {file.replace('_enrichment.csv', '')}")
            plt.xlabel("-log10(p-value)")
            plt.ylabel("GO Term")
            plt.legend(title="Gene Count", loc="lower right", bbox_to_anchor=(1.25, 0))
            plt.tight_layout()
            plt.grid(True, axis='x')

            # 儲存圖片
            plot_filename = file.replace(".csv", ".png")
            plt.savefig(os.path.join(output_folder, plot_filename))
            plt.close()

            print(f"✅ Saved plot for {file} to {plot_filename}")