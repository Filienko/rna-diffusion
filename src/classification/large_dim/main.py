# Imports
import sys
import os
import time as t
import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from src.classification.large_dim.model import TissuePredictor
from src.classification.large_dim.utils import get_class_weights, get_gtex_datasets, get_tcga_datasets

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", "--dataset",
                        dest = "dataset",
                        type = str,
                        required = True,
                        help="Specify the dataset to use for training (required).")
parser.add_argument("-config", "--config",
                        dest = "config",
                        type = int,
                        required = True,
                        help="Specify the model configuration number (required).")
parser.add_argument("-gpu_device", "--gpu_device",
                        dest = "gpu_device",
                        type = str,
                        required = True,
                        help="Specify the GPU device to use for training (required).")

args = parser.parse_args()
DATASET = args.dataset
CONFIG_NB = args.config
DEVICE = args.gpu_device

# Cuda device
if "cuda" in DEVICE:
    CUDA_DEVICE = int(DEVICE.strip("cuda:"))
else:
    CUDA_DEVICE = "cpu"

PATH_RESULTS_DATAFRAME = './sr./classification/results//results.csv'

# Import model config and hyperparameters
from config import CONFIGS
CONFIG = CONFIGS[CONFIG_NB]
CONFIG['device'] = CUDA_DEVICE

def main():
    print("----> Loading data")
    
    # Dataloader
    if DATASET.lower() =='tcga':
        train_loader, val_loader, test_loader = get_tcga_datasets(scaler_type='standard')
    elif DATASET.lower() =='gtex':
        train_loader, val_loader, test_loader = get_gtex_datasets(scaler_type='standard')

    train_loader = data.DataLoader(
                                    train_loader,
                                    batch_size=CONFIG['batch_size'],
                                    shuffle=True,
                                    num_workers=CONFIG['num_workers'],
                                    pin_memory=True,
                                    prefetch_factor=2,
                                    persistent_workers=True)
    
    val_loader = data.DataLoader(
                                    val_loader,
                                    batch_size=CONFIG['batch_size'],
                                    shuffle=False,
                                    num_workers=CONFIG['num_workers'],
                                    pin_memory=True,
                                    prefetch_factor=2,
                                    persistent_workers=True)
    
    test_loader = data.DataLoader(
                                    test_loader,
                                    batch_size=CONFIG['batch_size'],
                                    shuffle=False,
                                    num_workers=CONFIG['num_workers'],
                                    pin_memory=True,
                                    prefetch_factor=2,
                                    persistent_workers=True)
    
    # Weights
    tissue_weights = get_class_weights(train_loader, val_loader)
    
     # Starting time
    start_time = t.time()
    # Model training
    print(f"----> Training classifier on dataset {DATASET} with configuration {CONFIG_NB}.")
    model = TissuePredictor(CONFIG, DATASET)
    model.train(train_loader, 
                val_loader,  
                verbose=2, 
                class_weights=tissue_weights)
    
    # End time
    end_time = t.time()
    training_time =end_time - start_time
    
    # Train history
    train_loss = model.train_history_loss 
    train_acc = model.train_history_acc 
    train_f1 = model.train_history_f1
    val_loss = model.val_history_loss
    val_acc = model.val_history_acc
    val_f1 =  model.val_history_f1
    val_auc = model.val_history_auc
    val_cm = model.val_confusion_matrix
    
    # Test
    acc_test, f1_score_test, auc_test, test_cm = model.test(test_loader, output='metric')
    
    # Free memory
    model = []

    # Save to csv
    print("----> Saving results to csv")
    d = {'config': CONFIG_NB,
         'dataset': DATASET,
         'train_history_path': CONFIG['path'].format(DATASET),
         'classification': CONFIG['classification_task'],
         'x_dim': CONFIG['input_dim'],
        'hidden_dim1': CONFIG['hidden_dim'], 
         'batch_size': CONFIG['batch_size'], 
         'epochs': CONFIG['epochs'], 
         'activation_func':CONFIG['activation'],
         'optimizer':CONFIG['optimizer'],
         'lr':CONFIG['lr'],
         'batch_norm':CONFIG['BN'],
         'device': DEVICE,
         'training_time': training_time,
         'total_time': end_time-start_time,
         'train_loss': train_loss[-1],
         'val_loss': val_loss[-1],
        'train_acc': [train_acc],
        'train_f1':[train_f1],
        'val_acc':[val_acc],
        'val_f1':[val_f1],
        'val_auc':[val_auc],
        'val_cm':[val_cm],
        'test_acc': acc_test,
        'test_f1': f1_score_test,
        'test_auc': auc_test,
        'test_cm': [test_cm]}

    # Build dataframe
    df_temp = pd.DataFrame(data=d, index=[0])

    # Load results dataframe
    df = pd.read_csv(PATH_RESULTS_DATAFRAME, sep =',')
    # Merge dataframes
    df = pd.concat([df, df_temp])
    df.to_csv(PATH_RESULTS_DATAFRAME, sep =',', header=True, index=False)
    
    
# Run function    
if __name__ == '__main__':
    main()