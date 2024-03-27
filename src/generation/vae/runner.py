import os
import sys 
import csv 
import time
import torch
from src.generation.vae.utils import loss_function, save_weights


class Trainer(object):
    """
    """
    def __init__(self, config=None, model=None):
        """
        model (nn.Module): model to train
        config (dict): dictionary of hyperparameters and settings
        """
        self.config = config
        self.device = self.config['device']
        self.model = model.to(self.device)
        

    def __call__(self, TrainDataLoader, verbose=2):
        """
        training_data: tuple of train data and labels
        validation_data: tuple of validation data and labels
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

        # Init train metrics csv
        if os.path.exists(os.path.join(self.config['log_dir'], f"train_metrics_{self.config['dataset']}.csv")):
            os.remove(os.path.join(self.config['log_dir'], f"train_metrics_{self.config['dataset']}.csv"))
        with open(os.path.join(self.config['log_dir'], f"train_metrics_{self.config['dataset']}.csv"), "a") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_time", "train_loss", "train_mse", "train_kl"])
            f.close()
        
        # Start training
        self.model.train()
        train_start_time = time.time()
        for epoch in range(self.config['epochs']):
            epoch_loss = 0
            epoch_mse = 0
            epoch_kl = 0
            for i, (x, labels) in enumerate(TrainDataLoader):
                x = x.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                x_hat, mean, log_var = self.model(x, labels)
                mse, kl = loss_function(x, x_hat, mean, log_var)
                loss = mse+kl
                
                epoch_loss += loss.item()
                epoch_mse += mse.item()
                epoch_kl += -kl.item()
                
                loss.backward()
                optimizer.step()

            # end of epochs
            epoch_loss = epoch_loss/len(TrainDataLoader)
            epoch_mse = epoch_mse/len(TrainDataLoader)
            epoch_kl = epoch_kl/len(TrainDataLoader)

            # Store metrics
            csv_info = []
            csv_info.append(epoch)
            csv_info.append("%.2f" % (time.time() - train_start_time))
            csv_info.append(epoch_loss)
            csv_info.append(epoch_mse)
            csv_info.append(epoch_kl)
            with open(os.path.join(self.config['log_dir'], f"train_metrics_{self.config['dataset']}.csv"), "a") as f:
                w = csv.writer(f)
                w.writerow(csv_info)
                f.close()

            if verbose ==2:
                print("\tEpoch", epoch + 1, "\tEpoch Loss: ", epoch_loss,  "\tEpoch MSE: ", epoch_mse,  "\tEpoch KL: ", epoch_kl)

        
        # Save last weigths for encoder and decoder       
        save_weights(self.model.encoder, 
                     self.model.decoder, 
                     e_path=self.config['checkpoint_dir']+f"/enc_{self.config['dataset']}.pt", 
                     d_path=self.config['checkpoint_dir']+f"/dec_{self.config['dataset']}.pt")
        
        if verbose in [1,2]:
            print(f"End of training after {epoch} epochs. \t / Final training loss: {round(epoch_loss, 3)}", flush=True)
            print(f"Model saved at {self.config['checkpoint_dir']}")
        
        return self.model
