# Imports
import os
import random
import sys
import logging
import time as t
import argparse
import numpy as np
import pandas as pd
from src.reconstruction.model import LR, mae, mse
from src.reconstruction.utils import get_datasets_split_landmarks_for_search, split_and_scale_split_landmark
# Import model config and hyperparameters
from src.reconstruction.config import CONFIGS
# sys.path.append(os.path.abspath("../metrics"))
from src.metrics.precision_recall import compute_prdc
from src.metrics.aats import compute_AAts
from src.metrics.correlation_score import gamma_coeff_score

# SEED
np.random.seed(42)
random.seed(42)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", "--dataset",
                        dest = "dataset",
                        type = str,
                        required = True,
                        help="Specify the dataset to use for training (required).")
parser.add_argument("-nb_runs", "--nb_runs",
                        dest = "nb_runs",
                        type = int,
                        required = True,
                        help="Number of runs for a config (required).")
parser.add_argument("-nb_nn", "--nb_nn",
                        dest = "nb_nn",
                        type = int,
                        required = True,
                        help="Number of nearest neighbors (NNs) for metrics computations (required).")
parser.add_argument("-gpu_device", "--gpu_device",
                        dest = "gpu_device",
                        type = str,
                        required = True,
                        help="Specify the GPU device to use for training (required).")

args = parser.parse_args()
DATASET = args.dataset
DEVICE = args.gpu_device
NB_RUNS = args.nb_runs
NB_NN = args.nb_nn


# Cuda device
if "cuda" in DEVICE:
    CUDA_DEVICE = int(DEVICE.strip("cuda:"))
else:
    CUDA_DEVICE = "cpu"

def main():
    """
    Main function to run.
    """

    print("----> Loading data")
    X, y, X_test, Y_test, tissues_train, _ = get_datasets_split_landmarks_for_search(DATASET, landmark=False, split_landmark=True, with_tissues=True)

    CONFIG = {'path': f'./src/reconstruction/results/lr_{DATASET}.joblib',
              'nb_nn': NB_NN}

    # metrics
    TRAIN_SCORE = []
    TRAIN_MSE, VAL_MSE, TEST_MSE = [], [], []
    TRAIN_MAE, VAL_MAE, TEST_MAE = [], [], []
    PREC, REC, DENS, COV, AATS, CORR = [],[],[],[],[], []

    print(f"----> Start with {NB_RUNS} runs")
    for i in range(NB_RUNS):
        # Dataloader
        print("----> Loading data")
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_and_scale_split_landmark(X, y, X_test, Y_test, scaler_type="standard", tissues=tissues_train)
        
        print("----> Train model")
        model = LR()
        # Model training
        start = t.time()
        mse_train, mae_train, train_score = model.train(x_train, y_train, CONFIG)
        train_time = t.time() - start
        # Test
        print("----> Metrics")
        preds_train = model(x_train)
        preds_val = model(x_val)
        preds_test = model(x_test)
        mse_val, mae_val = mse(preds_val, y_val), mae(preds_val, y_val)
        mse_test, mae_test = mse(preds_test, y_test), mae(preds_test, y_test)

        TRAIN_SCORE.append(train_score)

        TRAIN_MSE.append(mse_train)
        VAL_MSE.append(mse_val)
        TEST_MSE.append(mse_test)

        TRAIN_MAE.append(mae_train)
        VAL_MAE.append(mae_val)
        TEST_MAE.append(mae_test)

        # Other metrics
        # Concatenate
        full_true = np.concatenate((x_train, y_train), axis=1)
        full_reconstructed = np.concatenate((x_train, preds_train), axis=1)
        # debug
        print(full_true.shape, full_reconstructed.shape)
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

    # Store results
    print(f"----> Store best results.")
    data = [train_time]
    data_cols = ['train_time']

    df_res = pd.DataFrame(columns=data_cols,
                            data=np.array([data]))
    
    df_res['train_R2'] = np.mean(TRAIN_SCORE)
    df_res['train_R2_std'] = np.std(TRAIN_SCORE)
    
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

    df_res.to_csv(f'./src/reconstruction/results/results_lr_{DATASET}.csv')
    print(f"----> End. ")
    
# Run function    
if __name__ == '__main__':
    main()