import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_class.method import method
from base_class.evaluate import evaluate


class Evaluate_Metrics_Adapted(evaluate):
    """Evaluation metrics class for node classification"""
    data = None
    metrics = [
        'Accuracy', 'F1 micro', 'F1 macro', 'F1 weighted',
        'Precision micro', 'Precision macro', 'Precision weighted',
        'Recall micro', 'Recall macro', 'Recall weighted'
    ]

    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)
        # Ensure instance metrics uses the full list
        self.metrics = Evaluate_Metrics_Adapted.metrics

    def evaluate(self):
        return {
            'Accuracy': accuracy_score(self.data['true_y'], self.data['pred_y']),
            'F1 micro': f1_score(self.data['true_y'], self.data['pred_y'], average='micro'),
            'F1 macro': f1_score(self.data['true_y'], self.data['pred_y'], average='macro', zero_division=0),
            'F1 weighted': f1_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0),
            'Precision micro': precision_score(self.data['true_y'], self.data['pred_y'], average='micro', zero_division=0),
            'Precision macro': precision_score(self.data['true_y'], self.data['pred_y'], average='macro', zero_division=0),
            'Precision weighted': precision_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0),
            'Recall micro': recall_score(self.data['true_y'], self.data['pred_y'], average='micro', zero_division=0),
            'Recall macro': recall_score(self.data['true_y'], self.data['pred_y'], average='macro', zero_division=0),
            'Recall weighted': recall_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0)
        }


class GCNLayer(nn.Module):
    """Single Graph Convolutional Layer"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, adj):
        x = torch.spmm(adj, x)
        return self.linear(x)


def normalize_adj(adj):
    """Normalize adjacency: D^{-1/2} A D^{-1/2}"""
    adj = adj.coalesce()
    indices = adj.indices(); values = adj.values(); n = adj.size(0)
    row_sum = torch.zeros(n).to(values.device)
    row_sum.index_add_(0, indices[0], values)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    new_values = d_inv_sqrt[indices[0]] * values * d_inv_sqrt[indices[1]]
    return torch.sparse_coo_tensor(indices, new_values, adj.size()).coalesce()


class Method_GCN_Adapted(nn.Module, method):
    """Graph Convolutional Network for Node Classification"""
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # Hyperparameters
        self.hidden_dim   = 32
        self.dropout_rate = 0.8
        self.learning_rate= 0.01
        self.weight_decay = 5e-4
        self.max_epoch    = 200
        self.patience     = 20
        self.loss_fn      = nn.CrossEntropyLoss()
        self.metric_evaluator = Evaluate_Metrics_Adapted('evaluator', '')
        # Storage for plotting curves
        self.curves = { 'epochs': [], 'train_loss': [], 'val_loss': [], 'test_loss': [] }
        self.gc1 = None; self.gc2 = None
        self.X_norm = None; self.adj_norm = None

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.gc2(x, adj)
        return x

    def train_model(self, X, y, adj, idx_train, idx_val, idx_test):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.X_norm  = F.normalize(X, p=2, dim=1)
        self.adj_norm= normalize_adj(adj)
        best_val_acc = -1; patience_counter = 0; best_test = None; final_test = None

        for epoch in range(self.max_epoch):
            # --- training step ---
            self.train(); optimizer.zero_grad()
            logits = self.forward(self.X_norm, self.adj_norm)
            loss_train = self.loss_fn(logits[idx_train], y[idx_train])
            loss_train.backward(); optimizer.step()

            # --- evaluation step ---
            self.eval()
            with torch.no_grad():
                logits = self.forward(self.X_norm, self.adj_norm)
                loss_val  = self.loss_fn(logits[idx_val], y[idx_val])
                loss_test = self.loss_fn(logits[idx_test],y[idx_test])
                pred_train= logits[idx_train].argmax(1)
                pred_val  = logits[idx_val].argmax(1)
                pred_test = logits[idx_test].argmax(1)

            # compute metrics
            self.metric_evaluator.data = {'true_y':y[idx_train].cpu(),'pred_y':pred_train.cpu()}
            train_m = self.metric_evaluator.evaluate()
            self.metric_evaluator.data = {'true_y':y[idx_val].cpu(),'pred_y':pred_val.cpu()}
            val_m   = self.metric_evaluator.evaluate()
            self.metric_evaluator.data = {'true_y':y[idx_test].cpu(),'pred_y':pred_test.cpu()}
            test_m  = self.metric_evaluator.evaluate()
            final_test = test_m.copy()

            # record curves
            self.curves['epochs'].append(epoch)
            self.curves['train_loss'].append(loss_train.item())
            self.curves['val_loss'].append(loss_val.item())
            self.curves['test_loss'].append(loss_test.item())

            # Print metrics every 10 epochs with values
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch:03d}")
                for m in Evaluate_Metrics_Adapted.metrics:
                    t = train_m[m]; v = val_m[m]; te = test_m[m]
                    print(f"{m}: train {t:.4f} | val {v:.4f} | test {te:.4f}")

            # early stopping
            if val_m['Accuracy'] > best_val_acc:
                best_val_acc = val_m['Accuracy']; patience_counter = 0; best_test = test_m.copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        print(f"\nTraining completed after {epoch+1} epochs")
        return best_test if best_test is not None else final_test

    def test_model(self, X, adj, idx_test):
        self.eval()
        with torch.no_grad():
            logits = self.forward(self.X_norm or F.normalize(X, p=2, dim=1),
                                  self.adj_norm or normalize_adj(adj))
            return logits[idx_test].argmax(1)

    def run(self):
        graph = self.data['graph']; split = self.data['train_test_val']
        X,y,adj = graph['X'],graph['y'],graph['utility']['A']
        idx_train,idx_val,idx_test = split['idx_train'],split['idx_val'],split['idx_test']
        # init layers
        self.gc1 = GCNLayer(X.shape[1], self.hidden_dim)
        self.gc2 = GCNLayer(self.hidden_dim, len(torch.unique(y)))
        # header
        print(f"-- {self.method_name} training start --\n")
        print(f"Model configuration:")
        print(f"- Input features: {X.shape[1]}")
        print(f"- Hidden dimension: {self.hidden_dim}")
        print(f"- Output classes: {len(torch.unique(y))}")
        print(f"- Training nodes: {len(idx_train)}")
        print(f"- Validation nodes: {len(idx_val)}")
        print(f"- Test nodes: {len(idx_test)}")
        print(f"- Learning rate: {self.learning_rate}")
        print(f"- Weight decay: {self.weight_decay}")
        print(f"- Dropout: {self.dropout_rate}")
        print(f"- Max epochs: {self.max_epoch}")
        print(f"- Patience: {self.patience}\n")

        best_test_metrics = self.train_model(X, y, adj, idx_train, idx_val, idx_test)
        pred_y = self.test_model(X, adj, idx_test)
        if best_test_metrics is None:
            self.metric_evaluator.data = {'true_y':y[idx_test].cpu(),'pred_y':pred_y.cpu()}
            best_test_metrics = self.metric_evaluator.evaluate()

        print("-- Final test evaluation --")
        for m, v in best_test_metrics.items():
            print(f"{m}: {v:.4f}")
        return {'pred_y':pred_y.cpu(), 'true_y':y[idx_test].cpu(), 'curves':self.curves, 'test_metrics':best_test_metrics}
