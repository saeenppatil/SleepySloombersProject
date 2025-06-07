#%%
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer

# Set seeds
np.random.seed(2)
torch.manual_seed(2)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_4_data', 'text_classification')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_4_result')
os.makedirs(RESULTS_DIR, exist_ok=True)

from stage_4_code.stage_4 import Method_LSTM, Evaluate_Metrics

# --- Load data from pos/neg folders ---
def load_classification_data(split_dir):
    data = []
    for label_name, label_val in [('pos', 1), ('neg', 0)]:
        label_dir = os.path.join(split_dir, label_name)
        for fname in os.listdir(label_dir):
            fpath = os.path.join(label_dir, fname)
            if os.path.isfile(fpath) and fpath.endswith('.txt'):
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    data.append({'text': text, 'label': label_val})
    return data

# --- Tokenize and convert to tensors ---
def prepare_tensor(dataset_list, tokenizer, max_len=128):
    texts = [d['text'] for d in dataset_list]
    labels = [d['label'] for d in dataset_list]
    encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    X = torch.stack([encodings['input_ids'], encodings['attention_mask']], dim=1)
    y = torch.LongTensor(labels)
    return X, y

# --- Load and prepare ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_data_raw = load_classification_data(os.path.join(DATA_DIR, 'train'))
test_data_raw = load_classification_data(os.path.join(DATA_DIR, 'test'))
X_train, y_train = prepare_tensor(train_data_raw, tokenizer)
X_test, y_test = prepare_tensor(test_data_raw, tokenizer)

data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}

# --- Run classifier ---
model = Method_LSTM('LSTM Sentiment Classifier', '', output_dim=2, max_epoch=10)
results = model.run(data)
curves = results['curves']

# --- Plots ---
metrics = ['Accuracy', 'F1 micro', 'F1 macro', 'F1 weighted']
plt.figure()
for metric in metrics:
    plt.plot(curves['epochs'], curves[metric], label=metric)
plt.title('Training Metrics'); plt.xlabel('Epochs'); plt.ylabel('Value'); plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'metrics.png')); plt.show()

plt.figure()
plt.plot(curves['epochs'], curves['loss'], label='Train Loss')
plt.plot(curves['epochs'], curves['test loss'], label='Test Loss')
plt.title('Loss Curve'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_curve.png')); plt.show()

plt.figure()
plt.plot(curves['epochs'], curves['Accuracy'], label='Train Accuracy')
plt.plot(curves['epochs'], curves['test accuracy'], label='Test Accuracy')
plt.title('Accuracy Curve'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'acc_curve.png')); plt.show()

# --- Final Evaluation ---
eval = Evaluate_Metrics('eval_final', '')
eval.data = {'true_y': results['true_y'], 'pred_y': results['pred_y']}
print("\nFinal Evaluation:")
for metric, val in eval.evaluate().items():
    print(f"{metric}: {val:.4f}")
#%%
