import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertTokenizer


# --- Evaluation ---
class Evaluate_Metrics:
    def __init__(self, name, desc): pass
    data = None
    metrics = ['Accuracy', 'F1 micro', 'F1 macro', 'F1 weighted',
               'Precision micro', 'Precision macro', 'Precision weighted',
               'Recall micro', 'Recall macro', 'Recall weighted']
    def evaluate(self):
        return {
            'Accuracy': accuracy_score(self.data['true_y'], self.data['pred_y']),
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


# --- LSTM Classifier + Generator ---
class Method_LSTM(nn.Module):
    def __init__(self, name, desc, output_dim=2, max_epoch=10):
        super().__init__()
        self.name = name
        self.max_epoch = max_epoch
        self.output_dim = output_dim
        self.learning_rate = 1e-3
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.loss_function = nn.CrossEntropyLoss()
        self.metric_evaluator = Evaluate_Metrics('eval', '')
        self.curves = {'epochs': [], 'loss': [], 'test loss': [], 'test accuracy': []}
        for metric in self.metric_evaluator.metrics:
            self.curves[metric] = []

        self.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        input_ids = x[:, 0, :]  # (B, seq_len)
        embeds = self.embedding(input_ids)
        out, _ = self.lstm(embeds)
        final_hidden = out[:, -1, :]
        return self.fc(final_hidden)

    def train_model(self, X, y, test_X, test_y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        batch_size = 32
        for epoch in range(self.max_epoch):
            self.train()
            perm = torch.randperm(X.size(0))
            for i in range(0, X.size(0), batch_size):
                idx = perm[i:i+batch_size]
                x_batch = X[idx]
                y_batch = y[idx]
                optimizer.zero_grad()
                out = self.forward(x_batch)
                loss = self.loss_function(out, y_batch)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                pred_y = self.forward(X).argmax(1)
                self.metric_evaluator.data = {'true_y': y, 'pred_y': pred_y}
                evals = self.metric_evaluator.evaluate()
                print(f"Epoch {epoch}: " + ', '.join([f'{k}: {v:.4f}' for k, v in evals.items()]))

            for k, v in evals.items():
                self.curves[k].append(v)
            self.curves['epochs'].append(epoch)
            self.curves['loss'].append(loss.item())

            with torch.no_grad():
                test_preds = self.forward(test_X).argmax(1)
                test_loss = self.loss_function(self.forward(test_X), test_y).item()
                test_acc = accuracy_score(test_y, test_preds)
            self.curves['test loss'].append(test_loss)
            self.curves['test accuracy'].append(test_acc)

    def test(self, X):
        return self.forward(X).argmax(1)

    def run(self, data):
        self.train_model(data['train']['X'], data['train']['y'], data['test']['X'], data['test']['y'])
        pred_y = self.test(data['test']['X'])
        return {'pred_y': pred_y, 'true_y': data['test']['y'], 'curves': self.curves}

    def generate(self, start_text, max_len=50):
        self.eval()
        tokens = self.tokenizer.encode(start_text, add_special_tokens=False)
        input_tensor = torch.LongTensor(tokens).unsqueeze(0)
        generated = tokens[:]
        hidden = None
        for _ in range(max_len):
            embeds = self.embedding(input_tensor)
            out, hidden = self.lstm(embeds, hidden)
            next_token = self.fc(out[:, -1, :]).argmax(dim=1).item()
            generated.append(next_token)
            input_tensor = torch.LongTensor([[next_token]])
        return self.tokenizer.decode(generated, skip_special_tokens=True)
