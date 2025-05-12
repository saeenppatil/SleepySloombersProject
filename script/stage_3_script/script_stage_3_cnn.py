#%%
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(2)
torch.manual_seed(2)

# Automatically find and insert the project root
HERE = os.path.dirname(os.path.abspath(__file__))  
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))  # go up to project root
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code')) 

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_3_data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_3_result')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Local imports
from stage_3_code.stage_3 import Method_CNN, Evaluate_Metrics, Dataset_Loader

# ---- CIFAR Loading Section ----
train_loader = Dataset_Loader('train', '')
train_loader.dataset_source_folder_path = os.path.join(DATA_DIR, '')
train_loader.dataset_source_file_name = 'CIFAR'
train_data = train_loader.load()

test_loader = Dataset_Loader('test', '')
test_loader.dataset_source_folder_path = os.path.join(DATA_DIR, '')
test_loader.dataset_source_file_name = 'CIFAR'
test_data = test_loader.load()

# Remove the extra dimension (dim=1)
# Fix data dimensions for CIFAR
train_data['X'] = train_data['X'].squeeze(1).permute(0, 3, 1, 2)
test_data['X'] = test_data['X'].squeeze(1).permute(0, 3, 1, 2)

# Optional: Normalize input data (you can skip if already normalized)
train_data['X'] /= 255.0
test_data['X'] /= 255.0

data = {'train': train_data, 'test': test_data}

# ---- Run CNN ----
method_obj = Method_CNN('CIFAR classifier', '')
method_obj.data = data
test_results = method_obj.run()
curves = test_results['curves']

# ---- Plot Training Metrics ----
metrics = ['Accuracy', 'F1 micro', 'F1 macro', 'F1 weighted', 'Precision micro',
           'Precision macro', 'Precision weighted', 'Recall micro', 'Recall macro', 'Recall weighted']

plt.figure()
for metric in metrics:
    plt.plot(curves['epochs'], curves[metric], label=metric)
plt.title('Training Evaluation Metrics')
plt.xlabel('Epochs'); plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'metrics.png'))
plt.show()

# ---- Loss Curve ----
plt.figure()
plt.title('Loss Curve')
plt.plot(curves['epochs'], curves['loss'], label='Train Loss')
plt.plot(curves['epochs'], curves['test loss'], label='Test Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_curve.png'))
plt.show()

# ---- Accuracy Curve ----
plt.figure()
plt.title('Accuracy Curve')
plt.plot(curves['epochs'], curves['Accuracy'], label='Train Accuracy')
plt.plot(curves['epochs'], curves['test accuracy'], label='Test Accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'acc_curve.png'))
plt.show()

# ---- Final Evaluation ----
eval = Evaluate_Metrics()
eval.data = {'true_y': test_results['true_y'], 'pred_y': test_results['pred_y']}
evals = eval.evaluate()

print('\nFinal Test Results:')
for metric, value in evals.items():
    print(f"{metric}: {value:.4f}")
  
# %%
