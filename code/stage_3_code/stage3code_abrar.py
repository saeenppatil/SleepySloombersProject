'''
Concrete MethodModule class for a specific learning MethodModule
'''

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/ataki/Documents/ECS189G_Winter_2025_Source_Code_Template/venv/Scripts')
sys.path.insert(1, 'C:/Users/ataki/Documents/ECS189G_Winter_2025_Source_Code_Template/data')

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
from local_code.base_class.dataset import dataset
from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle 
import random

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
    # it defines the max rounds to train the model
    max_epoch = 75
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_function = nn.CrossEntropyLoss()
    # for training accuracy investigation purpose
    metric_evaluator = Evaluate_Metrics('evaluator', '')
    curves = {}
    curves['epochs'] = []
    curves['loss'] = []
    curves['test loss']= []
    curves['test accuracy']= []
    for metric in metric_evaluator.metrics:
        curves[metric] = []
    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # in 28x28
        self.conv = nn.Sequential(
            nn.Conv2d(1,3,(3,3),stride=1,padding=1), nn.MaxPool2d(2,2), nn.ReLU(), #14x14x2
            nn.Conv2d(3,5,(3,3),stride=1,padding=1), nn.MaxPool2d(2,2), nn.ReLU(), #7x7x4
            )
        self.dense = nn.Sequential(
            nn.Linear(7*7*5,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10), nn.Softmax(dim=1)
        )

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        out = self.conv(x)
        out = torch.flatten(out, 1)
        out = self.dense(out)
        return out

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        X_train = X; y_train = y
        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # # you can try to split X and y into smaller-sized batches by yourself
        # dataloader = torch.utils.data.DataLoader(torch.utils.data.Dataset(X,y), batch_size=100, shuffle=True)
        batch_size=1000
        n_samples = len(X)
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            permutation = np.arange(0,n_samples).tolist()
            random.shuffle(permutation)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            for i in range(0, n_samples-1, batch_size):
                try: 
                    indices = np.arange(i,i + batch_size).tolist()
                except:
                    indices = np.arange(i,n_samples).tolist()
                X_train = X_shuffled[indices]
                y_train = y_shuffled[indices]
                y_pred = self.forward(X_train)
                # convert y to torch.tensor as well
                y_true = y_train
                # calculate the training loss
                train_loss = self.loss_function(y_pred, y_true)

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()
                #print('Loss:', train_loss.item())

            if True: #epoch%10 == 0:
                self.metric_evaluator.data = {'true_y': y, 'pred_y': self.forward(X).max(1)[1]}
                evals = self.metric_evaluator.evaluate()
                print('Epoch:', epoch, end=" ")
                for metric in evals.keys():
                    print(f"{metric}: {evals[metric]:.4f}", end=", ")
                    self.curves[metric].append(evals[metric])
                self.curves['epochs'].append(epoch)
                self.curves['loss'].append(train_loss.item())
                test_X = self.data['test']['X']
                test1, test2 = self.test(test_X, raw=True)
                self.curves['test loss'].append(self.loss_function(torch.FloatTensor(test2), torch.LongTensor(self.data['test']['y'])).item())
                self.curves['test accuracy'].append(accuracy_score(test1, self.data['test']['y']))

            print('Loss:', train_loss.item())

                
    
    def test(self, X, raw=False):
        # do the testing, and result the result
        test_X = X
        y_pred = self.forward(test_X)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        if raw:
            return y_pred.max(1)[1], y_pred
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'curves': self.curves}
            