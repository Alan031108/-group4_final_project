import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from sklearn.metrics import roc_auc_score

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

class GAETrainer:
    def __init__(self, train_data, test_data, hidden_dim=128, lr=0.0003, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_data = train_data.to(self.device)
        self.test_data = test_data.to(self.device)
        in_dim = train_data.x.shape[1]

        self.model = GAE(GCNEncoder(in_dim, hidden_dim), decoder=DotDecoder()).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)
        self.loss_fn = nn.MarginRankingLoss(margin=0.2)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )

        self.best_auc = 0
        self.wait = 0

    def train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        pos_pred = self.model.decoder(z, self.train_data.pos_edge_label_index)
        neg_pred = self.model.decoder(z, self.train_data.neg_edge_label_index)

        neg_sample_idx = torch.randperm(neg_pred.size(0))[:pos_pred.size(0)]
        neg_pred = neg_pred[neg_sample_idx]

        y = torch.ones(pos_pred.size(0), device=pos_pred.device)
        loss = self.loss_fn(pos_pred, neg_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test_epoch(self, data):
        self.model.eval()
        z = self.model.encode(data.x, data.edge_index)
        pos_pred = self.model.decoder(z, data.pos_edge_label_index)
        neg_pred = self.model.decoder(z, data.neg_edge_label_index)

        preds = torch.cat([pos_pred, neg_pred])
        labels = torch.cat([
            torch.ones(pos_pred.size(0), device=preds.device),
            torch.zeros(neg_pred.size(0), device=preds.device)
        ])

        probs = torch.sigmoid(preds)
        auc = roc_auc_score(labels.cpu(), probs.cpu())
        return auc, z

    def fit(self, max_epochs=300, patience=25):
        embeddings = None
        for epoch in range(1, max_epochs + 1):
            loss = self.train_one_epoch()
            if epoch % 10 == 0:
                auc, embeddings = self.test_epoch(self.test_data)
                print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}")
                self.scheduler.step(auc)
                if auc > self.best_auc and loss < 0.2:
                    self.best_auc = auc
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= patience:
                        print("Early stopping.")
                        break
        return self.model, self.best_auc, embeddings.cpu() if embeddings is not None else None
