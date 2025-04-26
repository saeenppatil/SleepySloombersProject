
import os, sys
import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from base_class.method import method
from base_class.dataset import dataset
from base_class.evaluate import evaluate

class Evaluate_Metrics(evaluate):
    data = None
    metrics = ['Accuracy', 'F1 micro', 'F1 macro', 'F1 weighted', 'Precision micro', 'Precision macro', 'Precision weighted', 'Recall micro', 'Recall macro', 'Recall weighted']
    def evaluate(self):
        return {'Accuracy': accuracy_score(self.data['true_y'], self.data['pred_y']),
                'F1 micro': f1_score(self.data['true_y'], self.data['pred_y'], average='micro'),
                'F1 macro': f1_score(self.data['true_y'], self.data['pred_y'], average='macro'),
                'F1 weighted': f1_score(self.data['true_y'], self.data['pred_y'], average='weighted'),
                'Precision micro': precision_score(self.data['true_y'], self.data['pred_y'], average='micro', zero_division=0.0),
                'Precision macro': precision_score(self.data['true_y'], self.data['pred_y'], average='macro', zero_division=0.0),
                'Precision weighted': precision_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0.0),
                'Recall micro': recall_score(self.data['true_y'], self.data['pred_y'], average='micro'),
                'Recall macro': recall_score(self.data['true_y'], self.data['pred_y'], average='macro'),
                'Recall weighted': recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')}

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        X, y = [], []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        for line in f:
            elements = [int(i) for i in line.strip().split(',')]
            X.append(elements[1:])
            y.append(elements[0])
        f.close()
        return {'X': X, 'y': y}

class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 300
    learning_rate = 5e-4
    weight_decay = 5e-4
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    metric_evaluator = Evaluate_Metrics('evaluator', '')
    curves = {'epochs': [], 'loss': [], 'test loss': [], 'test accuracy': []}
    for metric in metric_evaluator.metrics:
        curves[metric] = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.LeakyReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.LeakyReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.LeakyReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.LeakyReLU(), nn.Dropout(0.4),
            nn.Linear(32, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

    def train(self, X, y):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for epoch in range(self.max_epoch):
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            y_true = torch.LongTensor(np.array(y))
            train_loss = self.loss_function(y_pred, y_true)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or (epoch + 1) == self.max_epoch:
                self.metric_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                evals = self.metric_evaluator.evaluate()
                for metric in evals.keys():
                    self.curves[metric].append(evals[metric])
                self.curves['epochs'].append(epoch)
                self.curves['loss'].append(train_loss.item())
                test1, test2 = self.test(self.data['test']['X'], raw=True)
                self.curves['test loss'].append(self.loss_function(torch.FloatTensor(test2), torch.LongTensor(self.data['test']['y'])).item())
                self.curves['test accuracy'].append(accuracy_score(test1, self.data['test']['y']))

    def test(self, X, raw=False):
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        if raw:
            return y_pred.max(1)[1], y_pred
        return y_pred.max(1)[1]

    def run(self):
        self.train(self.data['train']['X'], self.data['train']['y'])
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'curves': self.curves}
