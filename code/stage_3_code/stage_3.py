import os, sys
import pickle

# absolute path to your project root
HERE         = os.path.dirname(os.path.abspath(__file__))        
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))  

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code'))

# import from base_class directly:
from base_class.method    import method
from base_class.dataset   import dataset
from base_class.evaluate  import evaluate

import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    dname = None
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.dName = dName
    
    def load(self):
        print('loading data...')
        X = []
        y = []

        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f); f.close()

        # i = 0
        for instance in data[self.dName]:
            X.append(instance['image'])
            y.append(instance['label'])
            # i = i+1
            # if i==1000: #for testing  
            #     break
        X = torch.FloatTensor(np.array(X)).unsqueeze(1)
        y = torch.LongTensor(np.array(y))
        return {'X': X, 'y': y}
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Evaluate_Metrics(evaluate):
    data = None
    metrics = ['Accuracy', 'F1 micro', 'F1 macro', 'F1 weighted', 'Precision micro', 'Precision macro', 'Precision weighted', 'Recall micro', 'Recall macro', 'Recall weighted']
    def evaluate(self):
        print('evaluating performance...')
        return {'Accuracy': accuracy_score(self.data['true_y'], self.data['pred_y']),
                'F1 micro': f1_score(self.data['true_y'], self.data['pred_y'], average='micro'),
                'F1 macro': f1_score(self.data['true_y'], self.data['pred_y'], average='macro'),
                'F1 weighted': f1_score(self.data['true_y'], self.data['pred_y'], average='weighted'),
                'Precision micro': precision_score(self.data['true_y'], self.data['pred_y'], average='micro', zero_division=0.0),
                'Precision macro': precision_score(self.data['true_y'], self.data['pred_y'], average='macro', zero_division=0.0),
                'Precision weighted': precision_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0.0),
                'Recall micro': recall_score(self.data['true_y'], self.data['pred_y'], average='micro'),
                'Recall macro': recall_score(self.data['true_y'], self.data['pred_y'], average='macro'),
                'Recall weighted': recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        }
    
        

class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 75
    learning_rate = 1e-3
    loss_function = nn.CrossEntropyLoss()
    metric_evaluator = Evaluate_Metrics('evaluator', '')
    curves = {'epochs': [], 'loss': [], 'test loss': [], 'test accuracy': []}
    for metric in metric_evaluator.metrics:
        curves[metric] = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv = nn.Sequential(
            # Block 1 (no Dropout2d)
            nn.Conv2d(3, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2,2),  # 16×16

            # Block 2 (light Dropout2d after pooling)
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2,2),  
            nn.Dropout2d(0.2),  # only here

            # Block 3 (no Dropout2d)
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.dense = nn.Sequential(
            nn.Flatten(),                # flatten conv output
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),             # your existing FC dropout
            nn.Linear(256, 10)
        )

    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out, 1)
        out = self.dense(out)
        return out

    def train(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.LongTensor(y)

        assert X.ndim == 4 and X.shape[1:] == (3, 32, 32), f"Expected input shape [N, 3, 32, 32], got {X.shape}"
        assert y.ndim == 1, f"Expected labels to be 1D, got {y.shape}"

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        batch_size = 32
        n_samples = X.shape[0]

        for epoch in range(self.max_epoch):
            permutation = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                indices = permutation[i:i+batch_size]
                X_batch = X[indices]
                y_batch = y[indices]

                assert X_batch.shape[1:] == (3, 32, 32), f"Bad batch input shape: {X_batch.shape}"

                optimizer.zero_grad()
                y_pred = self.forward(X_batch)
                loss = self.loss_function(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            # Evaluate on training in mini-batches with no_grad to save RAM
            with torch.no_grad():
                preds = []
                for i in range(0, n_samples, batch_size):
                    X_eval_batch = X[i:i+batch_size]
                    preds.append(self.forward(X_eval_batch).argmax(1))
                pred_y = torch.cat(preds)

            self.metric_evaluator.data = {
                'true_y': y,
                'pred_y': pred_y
            }
            evals = self.metric_evaluator.evaluate()
            print(f'Epoch {epoch}:', ', '.join(f'{k}: {v:.4f}' for k, v in evals.items()))
            
            for metric, value in evals.items():
                self.curves[metric].append(value)
            
            self.curves['epochs'].append(epoch)
            self.curves['loss'].append(loss.item())

            # Test with torch.no_grad and batchwise
            test_X = self.data['test']['X']
            test_y = self.data['test']['y']
            if isinstance(test_X, np.ndarray):
                test_X = torch.FloatTensor(test_X)
            if isinstance(test_y, np.ndarray):
                test_y = torch.LongTensor(test_y)

            with torch.no_grad():
                test_preds = []
                test_logits = []
                for i in range(0, test_X.shape[0], batch_size):
                    x_batch = test_X[i:i+batch_size]
                    logits = self.forward(x_batch)
                    test_logits.append(logits)
                    test_preds.append(logits.argmax(1))
                test_logits = torch.cat(test_logits)
                test_preds = torch.cat(test_preds)

            test_loss = self.loss_function(test_logits, test_y).item()
            test_acc = accuracy_score(test_y, test_preds)

            self.curves['test loss'].append(test_loss)
            self.curves['test accuracy'].append(test_acc)


    def test(self, X, raw=False):
        y_pred = self.forward(X)
        if raw:
            return y_pred.argmax(1), y_pred
        else:
            return y_pred.argmax(1)

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'curves': self.curves}
