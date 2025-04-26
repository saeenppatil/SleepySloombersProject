
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_2_data_sleepy_sloombers')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'stage_2_results_sleepy_sloombers')

from stage_2_code.stage_2 import Dataset_Loader, Method_MLP, Evaluate_Metrics

if __name__ == "__main__":
    np.random.seed(2)
    torch.manual_seed(2)

    print("Loading training data...")
    train_loader = Dataset_Loader('train', '')
    train_loader.dataset_source_folder_path = DATA_DIR + os.sep
    train_loader.dataset_source_file_name = 'train.csv'
    train_data = train_loader.load()

    print("Loading testing data...")
    test_loader = Dataset_Loader('test', '')
    test_loader.dataset_source_folder_path = DATA_DIR + os.sep
    test_loader.dataset_source_file_name = 'test.csv'
    test_data = test_loader.load()

    data = {'train': train_data, 'test': test_data}

    method_obj = Method_MLP('ImprovedMLP_SleepySloombers', '')
    method_obj.data = data

    test_results = method_obj.run()
    curves = test_results['curves']

    plt.figure()
    metrics = ['Accuracy', 'F1 micro', 'F1 macro', 'F1 weighted', 'Precision micro', 'Precision macro', 'Precision weighted', 'Recall micro', 'Recall macro', 'Recall weighted']
    for metric in metrics:
        plt.plot(curves['epochs'], curves[metric], label=metric)
    plt.title('Evaluation Metrics on Training Data')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics_sleepy_sloombers.png'))
    plt.show()

    plt.figure()
    plt.title('Training Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.plot(curves['epochs'], curves['loss'], label='Training Loss')
    plt.plot(curves['epochs'], curves['test loss'], label='Testing Loss')
    plt.plot(curves['epochs'], curves['Accuracy'], label='Training Accuracy')
    plt.plot(curves['epochs'], curves['test accuracy'], label='Testing Accuracy')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'val_curve_sleepy_sloombers.png'))
    plt.show()

    eval = Evaluate_Metrics()
    eval.data = {'true_y': test_results['true_y'], 'pred_y': test_results['pred_y']}

    evals = eval.evaluate()
    print('Test Results:')
    for metric in evals.keys():
        print(f"{metric}: {evals[metric]:.4f}", end=", ")
