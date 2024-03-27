#Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import random
from src.classification.utils import confusion_matrix, compute_auc, EarlyStopping
from sklearn.metrics import f1_score

# SEED
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class TissuePredictor():

    def __init__(self, config:dict=None, dataset:str=None):
        self.config = config
        self.path = self.config['path'].format(dataset) # Path where to store and load model
        self.path_history = self.config['path'].format(dataset+'_history')
        self.input_dim = self.config['input_dim']
        self.nb_classes = self.config['nb_classes']
        self.device = self.config['device']
        self.model = self.Classifier(self.input_dim, self.config['hidden_dim'], self.config['hidden_dim2'], self.nb_classes, self.config['dropout'])
        self.model.to(self.device)

    def train(self, TrainDataLoader, ValDataLoader, class_weights=None, verbose=2):
        """
        training_data (tensor): tuple of train and validation data
        training_labels (tensor): tuple of train and validation labels
        config (dict): dictionary of hyperparameters and settings
        verbose (int): verbose param
        -----
        Saves best model at self.path
        """

        # Init optimizer
        if self.config['optimizer'].lower() =='sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'], momentum=0.5)
        elif self.config['optimizer'].lower() =='adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        else:
            print(f"ValueError: {self.config['optimizer'].lower()} not recognized as optimizer.")
        
        # Initialize error to save better model in training
        best_error = np.inf
        # Initialize train and validation history
        self.train_history_loss = list()
        self.train_history_acc = list()
        self.train_history_f1 = list()
        self.val_history_loss = list()
        self.val_history_acc = list()
        self.val_history_f1 = list()
        self.val_history_auc = list()
        
        # Imbalance weights
        if class_weights is None:
            class_weights = torch.ones(size=(self.nb_classes,))/self.nb_classes
        class_weights = class_weights.to(self.device)

        # Set loss
        loss_func = nn.CrossEntropyLoss(reduction="mean", weight = class_weights, label_smoothing=0.1)

        # Init early stop
        earlystop = EarlyStopping(patience=50, verbose=verbose, path=self.path)

        # Start training
        for epoch in range(self.config['epochs']):
            # Keeping track of the loss at each epoch
            epoch_loss = 0
            pred_correct_train = 0
            train_samples_len = 0
            val_samples_len = 0
            y_true = torch.tensor([])
            y_pred = torch.tensor([])
            self.model.train() # train mode

            # Load batches of expression data, numerical covariates and encoded categorical covariates
            for i, (batch, labels) in enumerate(TrainDataLoader):
                
                #clearing the gradients of all optimized tensors
                optimizer.zero_grad()

                # To GPU, else CPU
                batch = batch.to(self.device)
                labels = labels.argmax(1).to(self.device)

                # Forward pass through model
                logits = self.model(batch)

                # Compute mean loss over batch
                loss = loss_func(logits, labels)
                loss.backward() # Compute the gradient
                optimizer.step() # Update parameters of the model
                
                # Add current batch loss to total training loss
                epoch_loss += loss.item()
                
                # Get predicted class
                pred = logits.argmax(1)
                

                # Compare prediction with ground truth
                pred_correct_train += (pred == labels).sum().item()
                train_samples_len += len(labels)
                # Keep predictions for f1 score
                y_true = torch.cat((labels.cpu().detach(), y_true), 0)
                y_pred = torch.cat((pred.cpu().detach(), y_pred), 0)
                
            # End of epoch

            # Save training accuracy
            self.train_history_acc.append(pred_correct_train / train_samples_len)

            # F1 score
            f1_score_train = f1_score(y_true, y_pred, average='weighted')
            self.train_history_f1.append(f1_score_train)

            # Evaluation on validation data
            self.model.eval()
            pred_correct = 0
            val_loss = 0
            val_samples_len = 0
            y_true = []
            y_pred = []
            y_logits=[]
            
            with torch.no_grad(): # No need of gradient
                for batch, labels in ValDataLoader:
                    # To GPU, else CPU
                    batch = batch.to(self.device)
                    labels = labels.argmax(1).to(self.device)
                    logits = self.model(batch)

                    # Get loss
                    val_loss += loss_func(logits, labels).item()
                    
                    # Get predicted class
                    pred = logits.argmax(1)

                    # Compare prediction with ground truth
                    pred_correct += (pred == labels).sum().item()
                    val_samples_len += len(labels)

                    # Keep predictions for f1 score
                    y_true.append(labels.cpu().detach())
                    y_pred.append(pred.cpu().detach())
                    
                    y_logits.append(logits.cpu().detach())

            # Save training/validation history
            self.train_history_loss.append(epoch_loss)
            self.val_history_loss.append(val_loss)
            self.val_history_acc.append(pred_correct / val_samples_len)

            # F1 score
            # Stack values
            y_true = torch.hstack(y_true)
            y_pred = torch.hstack(y_pred)
            f1_score_val = f1_score(y_true, y_pred, average='weighted')
            self.val_history_f1.append(f1_score_val)

            # AUC
            auc_val = compute_auc(y_true, y_logits)
            self.val_history_auc.append(auc_val)          

            if verbose==2:
                print("Epoch ", epoch,
                " Loss:\t", round(epoch_loss, 3),
                "Train accuracy:\t", round(pred_correct_train / train_samples_len, 3),
                "Train F1 score:\t", round(f1_score_train, 3),
                " Val Loss:\t", round(val_loss, 3),
                "\t / Val accuracy:\t", round(pred_correct / val_samples_len, 3),
                "\t / Val F1 score:\t", round(f1_score_val, 3),
                "\t / Val AUC:\t", round(auc_val, 3), flush=True)

            # Earlystopping
            self.last_epoch = epoch
            earlystop(self.val_history_acc[-1], self.model)
            if earlystop.early_stop:
                print('Stopping training after early stop...')
                break

        # Confusion matrix of last epoch
        print(f"Computing confusion matrix...")
        self.val_confusion_matrix = confusion_matrix(y_true, y_pred, self.nb_classes)

        # End of training
        # Save in dictionary
        torch.save({'train_history_loss': self.train_history_loss,
                    'train_history_acc':self.train_history_acc, 
                    'train_history_f1':self.train_history_f1,
                    'val_history_loss': self.val_history_loss, 
                    'val_history_acc': self.val_history_acc,
                    'val_history_f1': self.val_history_f1, 
                    'val_history_auc': self.val_history_auc,
                    'val_confusion_matrix': self.val_confusion_matrix},  
                    self.path_history)

        if verbose ==1 or verbose==2:
            print(f"End of training after {epoch} epochs. \t / Final training accuracy: {round(pred_correct_train / train_samples_len, 3)} // Final training f1 score: {f1_score_train}", flush=True)
            print(f"Final validation accuracy:\t {round(pred_correct / val_samples_len, 3)}|| Final val f1 score: {f1_score_val}|| Final val AUC: {auc_val}.")
            print(f"Training history saved at {self.path_history}.")
            

    def load_model(self, path=None):
        """ Loading previously trained model with history
        ----
        Parameters:
            path (str): path where model parameters should be retrieved (default None)
        """
        if path is None:
            path = self.path
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            #Load history
            self.train_history_loss = checkpoint['train_history_loss']
            self.train_history_acc = checkpoint['train_history_acc']
            self.train_history_f1 = checkpoint['train_history_f1']
            self.val_history_loss = checkpoint['val_history_loss']
            self.val_history_acc = checkpoint['val_history_acc']
            self.val_history_f1 = checkpoint['val_history_f1']
            self.val_history_auc = checkpoint['val_history_auc']
            self.val_confusion_matrix = checkpoint['val_confusion_matrix']
            print('Model and history loaded.')

        except FileNotFoundError: #if no model saved at given path
            print("No previously trained model at given path. Please train classifier using 'train' function first or correct path")
        
        
    def test(self, DataLoader, output:str='predictions'):
        """ Predict cancer with previously trained model at self.path.
        ----
        Parameters:
            DataLoader (pytorch DataLoader): test data with labels
            output (str): type of output (either 'metric' or 'predictions'). Default 'predictions'
        Returns:
            (torch.tensor): predicted labels of size (nb_test_samples, )
            or
            (tuple): if output is set on 'metric', the function returns accuracy and f1 score
        """                

        # Set model in evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        test_preds = []
        test_labels = []
        test_logits = []
        test_samples_len = 0
        pred_correct = 0
        test_samples_len = 0

        with torch.no_grad(): # no need of gradient
            for batch, labels in DataLoader:
                # To GPU, else CPU
                batch = batch.to(self.device)
                labels = labels.argmax(1).to(self.device)
                logits = self.model(batch)

                # Get predicted class
                pred = logits.argmax(1)

                # Compare prediction with ground truth
                pred_correct += (pred == labels).sum().item()
                test_samples_len += len(labels)

                # Keep predictions for f1 score
                test_labels.append(labels.cpu().detach())
                test_preds.append(pred.cpu().detach())
                test_logits.append(logits.cpu().detach())

        acc_test = pred_correct / test_samples_len

        # F1 score
        # Stack values
        test_labels = torch.hstack(test_labels)
        test_preds = torch.hstack(test_preds)
        f1_score_test = f1_score(test_labels, test_preds, average='weighted')

        # AUC
        auc_test = compute_auc(test_labels, test_logits)     
            
        # Confusion matrix 
        cm_test = confusion_matrix(test_labels, test_preds, self.nb_classes)
            

        if output.lower()=='predictions':
            return test_preds
        elif output.lower()=='metric':
            return acc_test, f1_score_test, auc_test, cm_test

    def predict(self, batch, labels=None, output:str='predictions'):
        """ Predict cancer with previously trained model at self.path.
        ----
        Parameters:
            data (tensor): test data with labels
            output (str): type of output (either 'metric' or 'predictions'). Default 'predictions'
        Returns:
            (torch.tensor): predicted labels of size (nb_test_samples, )
            or
            (tuple): if output is set on 'metric', the function returns accuracy and f1 score
        """                

        # Set device
        if self.model.device != self.device:
            self.model = self.model.to(self.device)
        # Set model in evaluation mode
        self.model.eval()

        with torch.no_grad(): # no need of gradient
            # To GPU, else CPU
            batch = batch.to(self.device)
            logits = self.model(batch)

            # Get predicted class
            pred = logits.argmax(1)

            if labels is not None:
                labels = labels.argmax(1).to(self.device)

                # Accuracy
                pred_correct = (pred == labels).sum()
                acc_test = pred_correct / len(labels)

                # F1 score
                f1_score_test = f1_score(labels.cpu(), pred.cpu(), average='weighted')

                # AUC
                auc_test = compute_auc(labels.cpu(), logits.cpu())     
                    
                # Confusion matrix 
                cm_test = confusion_matrix(labels.cpu(), pred.cpu(), self.nb_classes)
            

        if output.lower()=='predictions':
            return pred
        elif output.lower()=='probabilities':
            return torch.nn.functional.softmax(logits, dim=1)
        elif output.lower()=='metric':
            return acc_test, f1_score_test, auc_test, cm_test


    class Classifier(nn.Module):
        """ Classifier class for multiclass"""

        def __init__(self, input_dim, hidden_dim, hidden_dim2, output_dim, dropout_ratio=0.5):
            """
            input_dim:size of input sample
            hidden_dim: the hidden representation dim
            output_dim: number of output class (2, cancer or not)
            ---------------"""
            super().__init__()
            self.proj1 = nn.Linear(input_dim, hidden_dim) # Projection in hidden space
            self.proj2 = nn.Linear(hidden_dim, hidden_dim2) 
            self.proj3 = nn.Linear(hidden_dim2, output_dim) # Projection in output space
            self.block1 = nn.Sequential(self.proj1, nn.ReLU(), nn.Dropout(dropout_ratio), self.proj2, nn.ReLU(), nn.Dropout(dropout_ratio))

        def forward(self, x):
            """
            Forward pass
            """
            x = self.block1(x)
            return self.proj3(x)
