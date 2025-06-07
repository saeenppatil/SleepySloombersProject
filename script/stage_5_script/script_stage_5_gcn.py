#%%
# THIS SAME SCRIPT WAS USED TO RUN EVERYTHING.... JUST THE DATASET NAME WAS CHANGED
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Set seeds
np.random.seed(2)
torch.manual_seed(2)

# Project paths
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code'))

# Custom paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_5_data', 'pubmed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_5_result')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import method and loader
from stage_5_code.stage_5_pubmed import Method_GCN, Evaluate_Metrics
from stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader 

loader = Dataset_Loader(seed=2, dName='pubmed')
loader.dataset_source_folder_path = DATA_DIR
loader.dataset_name = 'pubmed'
data = loader.load()

gcn = Method_GCN('GCN', 'Graph Convolutional Network on Pubmed')
gcn.data = data
results = gcn.run()

curves = results['curves']


metrics = [
    'Accuracy', 'F1 micro', 'F1 macro', 'F1 weighted',
    'Precision micro', 'Precision macro', 'Precision weighted',
    'Recall micro', 'Recall macro', 'Recall weighted'
]

# 1) Loss Curve (Train vs. Test)
plt.figure()
plt.plot(curves['epochs'], curves['loss'],      label='Train Loss')
plt.plot(curves['epochs'], curves['test_loss'], label='Test Loss', linestyle='--')
plt.title('Loss Curve (Train vs. Test)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'pubmed_loss_train_vs_test.png'))
plt.show()

# 2) Accuracy Curve (Train vs. Test)
plt.figure()
plt.plot(curves['epochs'], curves['Accuracy'],       label='Train Accuracy')
plt.plot(curves['epochs'], curves['test_Accuracy'],  label='Test Accuracy', linestyle='--')
plt.title('Accuracy Curve (Train vs. Test)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'pubmed_accuracy_train_vs_test.png'))
plt.show()

# 3) All Other Metrics (Train vs. Test), one subplot per metric
plt.figure(figsize=(12, 8))

for metric in metrics:
    # Train curve
    plt.plot(
        curves['epochs'],
        curves[metric],
        label=f'Train {metric}',
        linewidth=1.5
    )
    # Test curve (dashed)
    plt.plot(
        curves['epochs'],
        curves[f'test_{metric}'],
        linestyle='--',
        label=f'Test {metric}',
        linewidth=1.5
    )

plt.title('All Metrics (Train vs. Test) Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend(loc='upper right', fontsize='small', ncol=2)  
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pubmed_all_metrics_train_vs_test.png'))
plt.show()

eval = Evaluate_Metrics()
eval.data = {
    'true_y': results['true_y'],
    'pred_y': results['pred_y']
}
final_metrics = eval.evaluate()

print('\nFinal Test Results:')
for k, v in final_metrics.items():
    print(f'{k}: {v:.4f}')

#%%