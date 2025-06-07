import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from base_class.method import method
from base_class.evaluate import evaluate


class Evaluate_Metrics(evaluate):
    data = None
    metrics = [
        'Accuracy', 'F1 micro', 'F1 macro', 'F1 weighted',
        'Precision micro', 'Precision macro', 'Precision weighted',
        'Recall micro', 'Recall macro', 'Recall weighted'
    ]

    def evaluate(self):
        return {
            'Accuracy': accuracy_score(self.data['true_y'], self.data['pred_y']),
            'F1 micro': f1_score(self.data['true_y'], self.data['pred_y'], average='micro'),
            'F1 macro': f1_score(self.data['true_y'], self.data['pred_y'], average='macro'),
            'F1 weighted': f1_score(self.data['true_y'], self.data['pred_y'], average='weighted'),
            'Precision micro': precision_score(self.data['true_y'], self.data['pred_y'], average='micro', zero_division=0),
            'Precision macro': precision_score(self.data['true_y'], self.data['pred_y'], average='macro', zero_division=0),
            'Precision weighted': precision_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0),
            'Recall micro': recall_score(self.data['true_y'], self.data['pred_y'], average='micro'),
            'Recall macro': recall_score(self.data['true_y'], self.data['pred_y'], average='macro'),
            'Recall weighted': recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        }


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, adj):
        x = torch.spmm(adj, x)
        return self.linear(x)


def normalize_adj(adj):
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    n = adj.size(0)

    row_sum = torch.zeros(n).to(values.device)
    row_sum.index_add_(0, indices[0], values)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    new_values = d_inv_sqrt[indices[0]] * values * d_inv_sqrt[indices[1]]
    return torch.sparse_coo_tensor(indices, new_values, adj.size()).coalesce()


class Method_GCN(method, nn.Module):
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.hidden_dim = 64
        self.dropout_rate = 0.3
        self.learning_rate = 0.005  # lower learning rate
        self.weight_decay = 5e-3
        self.max_epoch = 250
        self.loss_function = nn.CrossEntropyLoss()
        self.metric_evaluator = Evaluate_Metrics('evaluator', '')

        self.curves = {
            'epochs': [],
            'loss': [],
            'test_loss': []
        }
        for metric in self.metric_evaluator.metrics:
            self.curves[metric] = []
            self.curves[f'test_{metric}'] = []

        # Model architecture changed (3 GCN layers + BatchNorm)
        self.gc1 = GCNLayer(3703, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.gc2 = GCNLayer(self.hidden_dim, self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.gc3 = GCNLayer(self.hidden_dim, 6)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        x = self.gc2(x, adj)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        x = self.gc3(x, adj)
        return x

    def train_model(self, X, y, adj, idx_train, idx_test):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        X = F.normalize(X, p=2, dim=1)
        adj = normalize_adj(adj)

        for epoch in range(self.max_epoch):
            self.train()
            optimizer.zero_grad()

            logits = self.forward(X, adj)
            loss = self.loss_function(logits[idx_train], y[idx_train])
            loss.backward()
            optimizer.step()

            # TRAIN METRICS
            self.eval()
            with torch.no_grad():
                logits_train = self.forward(X, adj)
                pred_train = logits_train[idx_train].argmax(dim=1)

            self.metric_evaluator.data = {
                'true_y': y[idx_train],
                'pred_y': pred_train
            }
            train_results = self.metric_evaluator.evaluate()

            print(f"Epoch {epoch}")
            for metric_name in self.metric_evaluator.metrics:
                print(f"  Train {metric_name}: {train_results[metric_name]:.4f}")

            self.curves['epochs'].append(epoch)
            self.curves['loss'].append(loss.item())
            for metric_name in self.metric_evaluator.metrics:
                self.curves[metric_name].append(train_results[metric_name])

            # TEST METRICS
            with torch.no_grad():
                logits_test = self.forward(X, adj)
                pred_test = logits_test[idx_test].argmax(dim=1)
                loss_test = self.loss_function(logits_test[idx_test], y[idx_test])

            self.metric_evaluator.data = {
                'true_y': y[idx_test],
                'pred_y': pred_test
            }
            test_results = self.metric_evaluator.evaluate()

            self.curves['test_loss'].append(loss_test.item())
            for metric_name in self.metric_evaluator.metrics:
                self.curves[f'test_{metric_name}'].append(test_results[metric_name])

    def test(self, X, adj, idx_test):
        self.eval()
        with torch.no_grad():
            X = F.normalize(X, p=2, dim=1)
            adj = normalize_adj(adj)
            logits = self.forward(X, adj)
            return logits[idx_test].argmax(dim=1)

    def run(self):
        print('-- GCN training start --')
        graph = self.data['graph']
        split = self.data['train_test_val']

        X = graph['X']
        y = graph['y']
        adj = graph['utility']['A']
        idx_train = split['idx_train']
        idx_test = split['idx_test']

        self.train_model(X, y, adj, idx_train, idx_test)

        print('\n-- GCN final test evaluation --')
        pred_y = self.test(X, adj, idx_test)

        return {
            'pred_y': pred_y,
            'true_y': y[idx_test],
            'curves': self.curves
        }
