"""

Script for reconstruction models.

"""

import os
import sys 
import time
import random
import numpy as np
import torch
import torch.nn as nn
import csv
import joblib
from sklearn.linear_model import LinearRegression
from src.reconstruction.utils import EarlyStopping

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# MAE
def mae(x,y):
    return abs(x - y).mean()

# MSE
def mse(x,y):
    return ((x - y)**2).mean()

# Regression model
class LR(object):
    def __init__(self, pretrained_path=None):
        if pretrained_path:
            # load model
            self.lr_ = joblib.load(pretrained_path)
        else:
            self.lr_ = LinearRegression()

    def train(self, x, y, config=None):
        """
        Train function.
        x: input
        y: targets
        """
        self.lr_.fit(x, y)
        # save trained model
        joblib.dump(self.lr_, config['path'])

        # metrics 
        preds = self.lr_.predict(x)
        m = mae(preds, y)
        print(f"Train MAE: {m}")
        ms = mse(preds, y)
        print(f"Train MSE: {ms}")

        return ms, m, self.lr_.score(x, y)

    def __call__(self,x):
        """
        Main function to reconstruct.
        """
        return self.lr_.predict(x)


# Deep learning model (Multilayer perceptron)
class DGEX(nn.Module):
    def __init__(self, config):
        super().__init__()
        list_dims = config['list_dims']
        # self.proj1 = nn.Linear(config['input_dim'], config['hidden_dim1']) 
        # self.proj2 = nn.Linear(config['hidden_dim1'], config['hidden_dim2']) 
        # self.proj3 = nn.Linear(config['hidden_dim2'], config['output_dim']) # Reconstruction
        # self.bn1 = nn.BatchNorm1d(config['hidden_dim1'])
        # self.bn2 = nn.BatchNorm1d(config['hidden_dim2'])
        # self.block = nn.Sequential(self.proj1, 
        #                            self.bn1,
        #                             nn.ReLU(), 
        #                             nn.Dropout(config['dropout']), 
        #                             self.proj2, 
        #                             self.bn2,
        #                             nn.ReLU(), 
        #                             nn.Dropout(config['dropout']),
        #                             self.proj3)

        layers =[]
        for i in range(len(list_dims)-1):
            if i == len(list_dims)-2:
                layers.append(nn.Linear(list_dims[i], list_dims[i+1]))
            else:
                layers.append(nn.Linear(list_dims[i], list_dims[i+1]))
                layers.append(nn.BatchNorm1d(list_dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config['dropout']))

        self.block = nn.Sequential(*layers)
        
    
    def forward(self, x):
        """
        Forward pass
        """
        return self.block(x)
    

# Reconstruction model
class GeRec(object):
    def __init__(self, model_type:str='lr', pretrained:bool=True, config_mlp=None):
        self.model_type = model_type
        if self.model_type=='lr':
            self.model = LR(pretrained)

        elif self.model_type=='mlp':
            self.model = DGEX(config_mlp)

    def __call__(self, x):
        """
        Main function to reconstruct data from landmark genes.
        Parameters:
            x: input data as np.array or torch.tensor
        Returns:
            target genes
        """
        # Put to tensors and device if not done
        if self.model_type=='mlp':
            if not isinstance(x, torch.tensor):
                x = torch.from_numpy(x)
            # To device
            x = x.to(self.model.device)

        return self.model(x)
    

class Trainer(object):
    def __init__(self, config=None, model=None):
        """
        model (nn.Module): model to train
        config (dict): dictionary of hyperparameters and settings
        """
        self.config = config
        self.device = self.config['device']
        self.model = model.to(self.device)

    def __call__(self, TrainDataLoader, ValDataLoader, verbose=2):
            """
            training_data (tensor): tuple of train and validation data
            training_labels (tensor): tuple of train and validation labels
            verbose (int): verbose param
            -----
            Saves best model at self.path
            """

            # Init optimizer
            if self.config['optimizer'].lower() =='sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'])
            elif self.config['optimizer'].lower() =='adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
            else:
                print(f"ValueError: {self.config['optimizer'].lower()} not recognized as optimizer.")
            
            # Initialize error to save better model in training
            best_error = np.inf
            # Initialize train and validation history
            self.train_history_loss = list()
            self.train_history_mae = list()
            self.val_history_loss = list()
            self.val_history_mae = list()

            # Set loss
            self.loss_func = nn.MSELoss(reduction="mean")

            # Init early stop
            earlystop = EarlyStopping(patience=25, verbose=verbose, path=self.config['path'])

            # Init train metrics csv
            if os.path.exists(os.path.join(self.config['path_logs'], "train_metrics.csv")):
                os.remove(os.path.join(self.config['path_logs'], "train_metrics.csv"))
            with open(os.path.join(self.config['path_logs'], "train_metrics.csv"), "a") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_time", "train_mse", "train_mae", "val_mse", "val_mae"])
                f.close()

            # Start training
            train_start_time = time.time()
            for epoch in range(self.config['epochs']):
                # Keeping track of the loss at each epoch
                epoch_loss = 0
                epoch_mae = 0
                self.model.train() # train mode

                # Load batches of expression data, numerical covariates and encoded categorical covariates
                for i, (batch, labels) in enumerate(TrainDataLoader):
                    
                    # clearing the gradients of all optimized tensors
                    optimizer.zero_grad()

                    # To GPU, else CPU
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass through model
                    logits = self.model(batch)

                    # Compute mean loss over batch
                    loss = self.loss_func(logits, labels)
                    loss.backward() # Compute the gradient
                    optimizer.step() # Update parameters of the model
                    
                    # Add current batch loss to total training loss
                    epoch_loss += loss.item()
                    
                    # Metrics
                    epoch_mae += mae(labels, logits).item()
                    
                # End of epoch
                epoch_loss = epoch_loss/len(TrainDataLoader)
                epoch_mae = epoch_mae/len(TrainDataLoader)

                # Evaluation on validation data
                self.model.eval()
                val_loss = 0
                val_mae = 0
                
                with torch.no_grad(): # No need of gradient
                    for batch, labels in ValDataLoader:
                        batch = batch.to(self.device)
                        labels = labels.to(self.device)
                        logits = self.model(batch)

                        # Get loss
                        val_loss += self.loss_func(logits, labels).item()

                        # Metrics
                        val_mae += mae(labels, logits).item()

                val_loss = val_loss/len(ValDataLoader)
                val_mae = val_mae/len(ValDataLoader)

                csv_info = []
                csv_info.append(epoch)
                csv_info.append("%.2f" % (time.time() - train_start_time))
                csv_info.append(epoch_loss)
                csv_info.append(epoch_mae)
                csv_info.append(val_loss)
                csv_info.append(val_mae) 
                with open(os.path.join(self.config['path_logs'], "train_metrics.csv"), "a") as f:
                    w = csv.writer(f)
                    w.writerow(csv_info)
                    f.close()

                if verbose==2:
                    print("Epoch ", epoch,
                    " Loss:\t", round(epoch_loss, 3),
                    "Train MAE:\t", round(epoch_mae, 3),
                    " Val Loss:\t", round(val_loss, 3),
                    "\t / Val MAE:\t", round(val_mae, 3), flush=True)

                # Earlystopping
                earlystop(val_loss, self.model)
                if earlystop.early_stop:
                    print('Stopping training after early stop...')
                    break

            if verbose in [1,2]:
                print(f"End of training after {epoch} epochs. \t / Final training MAE: {round(epoch_mae, 3)}", flush=True)
                print(f"Final validation MAE:\t {round(val_mae, 3)}")
                print(f"Model saved at {self.config['path']}.")


    def last_prediction(self, ValDataLoader):
        """
        """
        # Evaluation on validation data
        self.model.eval()
        val_loss = 0
        val_mae = 0
        
        with torch.no_grad(): # No need of gradient
            for batch, labels in ValDataLoader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)

                # Get loss
                val_loss += self.loss_func(logits, labels).item()

                # Metrics
                val_mae += mae(labels, logits).item()

        val_loss = val_loss/len(ValDataLoader)
        val_mae = val_mae/len(ValDataLoader)

        return val_loss, val_mae
    
    def inference(self, ValDataLoader, return_input:bool=False):
        """
        """
        # Evaluation on validation data
        self.model.eval()
        preds, full_labels, full_inputs = [], [], []
        
        with torch.no_grad(): # No need of gradient
            for batch, labels in ValDataLoader:
                batch = batch.to(self.device)
                logits = self.model(batch)

                preds.append(logits.detach().cpu())
                full_labels.append(labels)
                full_inputs.append(batch.cpu())

        preds = torch.vstack((preds))
        full_labels = torch.vstack((full_labels))
        full_inputs = torch.vstack((full_inputs))

        if return_input:
            return full_labels, preds, full_inputs
        else:
            return full_labels, preds