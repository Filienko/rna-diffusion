# Imports
import os
import sys
import random
import torch
import logging
import time as t
import argparse
import numpy as np
import pandas as pd
from src.reconstruction.model import DGEX, Trainer
from src.reconstruction.utils import get_datasets_split_landmarks_for_search, split_and_scale_datasets_split_landmark, build_loaders
# Import model config and hyperparameters
from src.reconstruction.config import CONFIGS
# sys.path.append(os.path.abspath("../metrics"))
from src.metrics.precision_recall import compute_prdc
from src.metrics.aats import compute_AAts
from src.metrics.correlation_score import gamma_coeff_score

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

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
parser.add_argument("-nb_runs", "--nb_runs",
                        dest = "nb_runs",
                        type = int,
                        required = True,
                        help="Number of runs for a config (required).")
parser.add_argument("-gpu_device", "--gpu_device",
                        dest = "gpu_device",
                        type = str,
                        required = True,
                        help="Specify the GPU device to use for training (required).")

args = parser.parse_args()
DATASET = args.dataset
CONFIG_NB = args.config
DEVICE = args.gpu_device
NB_RUNS = args.nb_runs

# Cuda device
if "cuda" in DEVICE:
    CUDA_DEVICE = int(DEVICE.strip("cuda:"))
else:
    CUDA_DEVICE = "cpu"

CONFIG = CONFIGS[CONFIG_NB]
CONFIG['device'] = CUDA_DEVICE


def main():
    """
    Main function to run.
    """

    print("----> Loading data")
    X, y, X_test, y_test, tissues_train, _ = get_datasets_split_landmarks_for_search(DATASET, landmark=False, split_landmark=True, with_tissues=True)

    # load results search
    df_best_params = pd.read_csv(f'./src/reconstruction/results/best_params_search_{DATASET}.csv', sep=',')

    print(f"----> Run best model and predict on test data ")
    CONFIG['lr'] = float(df_best_params['lr'].item())
    CONFIG['optimizer'] = str(df_best_params['optimizer'].item())
    CONFIG['batch_size'] = int(df_best_params['batch_size'].item())
    CONFIG['dropout'] = float(df_best_params['dropout'].item()) 
    # Init list dims with input dim
    CONFIG['list_dims'] = [CONFIG['input_dim']]
    CONFIG['n_blocks'] = int(df_best_params['n_blocks'].item())
    for i in range(1, CONFIG['n_blocks']+1):
        CONFIG['list_dims'].append(int(df_best_params[f"hidden_dim{i}"].item()))

    # Add output dim as last dim
    CONFIG['list_dims'].append(CONFIG['output_dim'])

    # metrics
    TRAIN_MSE, VAL_MSE, TEST_MSE = [], [], []
    TRAIN_MAE, VAL_MAE, TEST_MAE = [], [], []
    PREC, REC, DENS, COV, AATS, CORR = [],[],[],[],[], []

    print(f"----> Start with {NB_RUNS} runs")
    for i in range(NB_RUNS):
        # Dataloader
        print("----> Loading data")
        train_loader, val_loader, test_loader = split_and_scale_datasets_split_landmark(X, y, X_test, y_test, tissues= tissues_train, scaler_type="standard")
        train_loader, val_loader, test_loader = build_loaders(train_loader, val_loader, test=test_loader, config=CONFIG)
        
        print("----> Train best model")
        model = DGEX(CONFIG)
        # Model training
        trainer = Trainer(CONFIG, model)
        trainer(train_loader, 
                    val_loader,  
                    verbose=2)
        # Test
        print("----> Metrics")
        mse_train, mae_train = trainer.last_prediction(train_loader)
        mse_val, mae_val = trainer.last_prediction(val_loader)
        mse_test, mae_test = trainer.last_prediction(test_loader)

        TRAIN_MSE.append(mse_train)
        VAL_MSE.append(mse_val)
        TEST_MSE.append(mse_test)

        TRAIN_MAE.append(mae_train)
        VAL_MAE.append(mae_val)
        TEST_MAE.append(mae_test)

        # Other metrics
        full_labels, preds, full_inputs = trainer.inference(train_loader, return_input=True)
        full_labels, preds, full_inputs = full_labels.numpy(), preds.numpy(), full_inputs.numpy()
        
        # Concatenate
        full_true = np.concatenate((full_inputs, full_labels), axis=1)
        full_reconstructed = np.concatenate((full_inputs, preds), axis=1)
        
        # Precision/Recall/Density/Coverage
        print("PRDC...")
        prec, recall, dens, cov = compute_prdc(full_true, full_reconstructed, CONFIG['nb_nn'])
        PREC.append(prec)
        REC.append(recall)
        DENS.append(dens)
        COV.append(cov)
        # Adversarial accuracy
        print("AATS...")
        idx = np.random.choice(len(full_true), 2048, replace=False) # Sample random data
        _, _, aa = compute_AAts(real_data=full_true[idx], fake_data=full_reconstructed[idx])
        # Correlations
        print("Correlations...")
        corr = gamma_coeff_score(full_true[idx], full_reconstructed[idx])
        AATS.append(aa)
        CORR.append(corr)

        # memory
        full_true = []
        full_reconstructed = []
        full_labels, preds, full_inputs = [], [], []

    # Store results
    print(f"----> Store best results.")
    data = []
    data_cols = []
    for c in df_best_params.columns:
        if c.lower() not in ['train_mse', 'train_mae', 'val_mse', 'val_mae', 'test_mse', 'test_mae']:
            data_cols.append(c)
            data.append(df_best_params[c].item())

    df_res = pd.DataFrame(columns=data_cols,
                            data=np.array([data]))
    df_train_metrics = pd.read_csv(os.path.join(CONFIG['path_logs'], "train_metrics.csv"), sep=',')
    df_res['train_time'] = df_train_metrics['train_time'].iloc[-1]

    df_res['train_mse'], df_res['train_mae'] = np.mean(TRAIN_MSE), np.mean(TRAIN_MAE)
    df_res['val_mse'], df_res['val_mae'] = np.mean(VAL_MSE), np.mean(VAL_MAE)
    df_res['test_mse'], df_res['test_mae'] = np.mean(TEST_MSE), np.mean(TEST_MAE)

    df_res['train_mse_std'], df_res['train_mae_std'] = np.std(TRAIN_MSE), np.std(TRAIN_MAE)
    df_res['val_mse_std'], df_res['val_mae_std'] = np.std(VAL_MSE), np.std(VAL_MAE)
    df_res['test_mse_std'], df_res['test_mae_std'] = np.std(TEST_MSE), np.std(TEST_MAE)

    df_res['precision'], df_res['precision_std'] = np.mean(PREC), np.std(PREC)
    df_res['recall'], df_res['recall_std'] = np.mean(REC), np.std(REC)
    df_res['density'], df_res['density_std'] = np.mean(DENS), np.std(DENS)
    df_res['coverage'], df_res['coverage_std'] = np.mean(COV), np.std(COV)
    df_res['aats'], df_res['aats_std'] = np.mean(AATS), np.std(AATS)
    df_res['correlation'], df_res['correlation_std'] = np.mean(CORR), np.std(CORR)

    df_res.to_csv(f'./src/reconstruction/results/results_mlp_{DATASET}.csv')
    print(f"----> End. ")
    
# Run function    
if __name__ == '__main__':
    main()