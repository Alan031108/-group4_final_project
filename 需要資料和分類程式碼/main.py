from data_preprocessing import read_biogrid_data, preprocess, build_graph
from gae_trainer import GAETrainer
from predicted_edges import predict
from drawing_and_communities import draw_confidence_distribution, draw_high_score_prediction, louvain, draw_community_protein
from compare_string_database import compare_string_database
from enrichment import split_communities, go_enrichment, draw_bubble_plot
import pandas as pd
import os

biogrid_file_path = r'.\archive\BIOGRID-ALL-4.3.195.tab3\BIOGRID-ALL-4.3.195.tab3.txt'
input_df = read_biogrid_data(biogrid_file_path)

file_dir = r'.\csv_output'
os.makedirs(file_dir, exist_ok=True)

plot_dir = r'.\images_output'
os.makedirs(plot_dir, exist_ok=True)

edges = preprocess(input_df)
reverse_mapping, train_data, val_data, test_data = build_graph(edges)

trainer = GAETrainer(train_data, test_data, hidden_dim=128, lr=0.0003)
model, best_auc, embeddings = trainer.fit(max_epochs=300, patience=25)

predicted_output_path = os.path.join(file_dir, 'yeast_predicted_interactions_optimized.csv')
predict(predicted_output_path, embeddings, train_data, reverse_mapping)

predicted_df = pd.read_csv(predicted_output_path)

image_output = os.path.join(plot_dir, 'confidence_distribution.png')
draw_confidence_distribution(predicted_df, image_output)

image_output = os.path.join(plot_dir, 'high_score_prediction.png')
draw_high_score_prediction(predicted_df, image_output)

communities_output_path = os.path.join(file_dir, 'yeast_predicted_communities.csv')
image_output = os.path.join(plot_dir, 'louvain.png')
louvain(predicted_df, communities_output_path, image_output)

image_output = os.path.join(plot_dir, 'community_protein.png')
draw_community_protein(predicted_df, predicted_output_path, communities_output_path, image_output)

alias_path = r'.\4932.protein.aliases.v12.0.txt.gz'
link_path = r'.\4932.protein.links.full.v12.0.txt.gz'

predicted_string_output_path = os.path.join(file_dir, 'yeast_predicted_with_STRING.csv')
unmatched_output_path = os.path.join(file_dir, 'unmatched_predicted_interactions.csv')
compare_string_database(predicted_output_path, alias_path, link_path, predicted_string_output_path, unmatched_output_path)

split_communities_dir = r'.\split_communities'
os.makedirs(split_communities_dir, exist_ok=True)
communities_df = pd.read_csv(communities_output_path)
split_communities(communities_df, split_communities_dir)

go_dir = r'.\go_results'
os.makedirs(go_dir, exist_ok=True)
go_enrichment(split_communities_dir, go_dir)

plot_folder = os.path.join(go_dir, 'plots')
os.makedirs(plot_folder, exist_ok=True)
draw_bubble_plot(go_dir, plot_folder)
