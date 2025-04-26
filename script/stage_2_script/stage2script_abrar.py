#%%
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/ataki/Documents/ECS189G_Winter_2025_Source_Code_Template/venv/Scripts')
sys.path.insert(1, 'C:/Users/ataki/Documents/ECS189G_Winter_2025_Source_Code_Template/data')

from local_code.stage_2_code.stage_2 import Dataset_Loader, Method_MLP, Evaluate_Metrics
from local_code.stage_1_code.Result_Saver import Result_Saver
from local_code.stage_1_code.Setting_KFold_CV import Setting_KFold_CV
#from local_code.stage_1_code.Setting_Train_Test_Split import Setting_Train_Test_Split
import numpy as np
import torch
import matplotlib.pyplot as plt
'''
Concrete IO class for a specific dataset
'''



#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    train_loader = Dataset_Loader('train', '')
    train_loader.dataset_source_folder_path = 'C:/Users/ataki/Documents/ECS189G_Winter_2025_Source_Code_Template/data/stage_2_data/'
    train_loader.dataset_source_file_name = 'train.csv'
    train_data = train_loader.load()

    test_loader = Dataset_Loader('test', '')
    test_loader.dataset_source_folder_path = 'C:/Users/ataki/Documents/ECS189G_Winter_2025_Source_Code_Template/data/stage_2_data/'
    test_loader.dataset_source_file_name = 'test.csv'
    test_data = test_loader.load()

    data = {'train': train_data, 'test': test_data}

    method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj.data = data

    test_results = method_obj.run()
    curves = test_results['curves']

    plt.figure()
    metrics = ['Accuracy', 'F1 micro', 'F1 macro', 'F1 weighted', 'Precision micro', 'Precision macro', 'Precision weighted', 'Recall micro', 'Recall macro', 'Recall weighted']
    for metric in metrics:
        plt.plot(curves['epochs'],curves[metric], label=metric)
    plt.title('Evaluation metrics on training data')
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('C:/Users/ataki/Documents/ECS189G_Winter_2025_Source_Code_Template/result/stage_2_result/metrics.png')
    plt.show()

    plt.figure()
    plt.title('Training curves')
    plt.xlabel('epochs')
    plt.ylabel('loss/accuracy')
    plt.plot(curves['epochs'],curves['loss'],label='training loss')
    plt.plot(curves['epochs'],curves['test loss'],label='testing loss')
    plt.plot(curves['epochs'],curves['Accuracy'],label='training accuracy')
    plt.plot(curves['epochs'],curves['test accuracy'],label='testing accuracy')
    plt.savefig('C:/Users/ataki/Documents/ECS189G_Winter_2025_Source_Code_Template/result/stage_2_result/val_curve.png')
    plt.legend()
    plt.show()
    
    eval = Evaluate_Metrics()
    eval.data = {'true_y': test_results['true_y'], 'pred_y': test_results['pred_y']}

    evals = eval.evaluate()
    print('Test results:')
    for metric in evals.keys():
        print(f"{metric}: {evals[metric]:.4f}", end=", ")

    plt.figure()
    plt.title('[Training curves')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(curves['epochs'],curves['loss'],label='training loss')
    plt.plot(curves['epochs'],curves['test loss'],label='testing loss')
    #plt.plot(curves['epochs'],curves['Accuracy'],label='training accuracy')
    #plt.plot(curves['epochs'],curves['test accuracy'],label='testing accuracy')
    plt.savefig('C:/Users/ataki/Documents/ECS189G_Winter_2025_Source_Code_Template/result/stage_2_result/loss_curve.png')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Training curves')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    # plt.plot(curves['epochs'],curves['loss'],label='training loss')
    # plt.plot(curves['epochs'],curves['test loss'],label='testing loss')
    plt.plot(curves['epochs'],curves['Accuracy'],label='training accuracy')
    plt.plot(curves['epochs'],curves['test accuracy'],label='testing accuracy')
    plt.savefig('C:/Users/ataki/Documents/ECS189G_Winter_2025_Source_Code_Template/result/stage_2_result/loss_curve.png')
    plt.legend()
    plt.show()

    for metric in evals.keys():
        print(f"{metric}: {curves[metric][-1]:.4f}", end=", ")

    
# %%
