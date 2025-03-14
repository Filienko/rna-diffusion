# Imports
import sys
import os
import time as t
import argparse
import numpy as np
import pandas as pd
import torch
import random
from src.generation.gans.model import WGAN
from src.generation.gans.utils import get_tcga_datasets, get_gtex_datasets,get_cambda_datasets, build_loaders
# Import model config and hyperparameters
from src.generation.gans.config import CONFIG
# sys.path.append(os.path.abspath("../metrics"))
from src.metrics.precision_recall import compute_prdc
from src.metrics.aats import compute_AAts
from src.metrics.correlation_score import gamma_coeff_score
from src.metrics.frechet import compute_frechet_distance_score

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
                        help="Indicate the dataset to use. (required).")
parser.add_argument("-with_best_params", "--with_best_params",
                        dest = "with_best_params",
                        type = str,
                        required = True,
                        help="Indicate if WGAN-GP model should be trained with best params (required).")
parser.add_argument("-gpu_device", "--gpu_device",
                        dest = "gpu_device",
                        type = str,
                        required = True,
                        help="Specify the GPU device to use for training (required)")
parser.add_argument("-gpu_device_frechet", "--gpu_device_frechet",
                        dest = "gpu_device_frechet",
                        type = str,
                        required = True,
                        help="Specify the GPU device to use for Frechet computations (required)")
args = parser.parse_args()
DATASET = args.dataset
WBP = args.with_best_params
DEVICE = args.gpu_device
DEVICE2 = args.gpu_device_frechet

# Cuda device
if "cuda" in DEVICE:
    CUDA_DEVICE = int(DEVICE.strip("cuda:"))
else:
    CUDA_DEVICE = "cpu"

# Cuda device frechet
if "cuda" in DEVICE2:
    CUDA_DEVICE2 = int(DEVICE2.strip("cuda:"))
else:
    CUDA_DEVICE2 = "cpu"

CONFIG['device'] = CUDA_DEVICE
CONFIG['device_frechet'] = CUDA_DEVICE2

# Dataset
CONFIG['dataset'] = DATASET
# Results
PATH_RESULTS_DATAFRAME = './src/generation/gans/results/results_wgangp.csv'


# Best params config
if WBP=='y':
    best_params = pd.read_csv(f"./src/generation/gans/results/tissue_search_{CONFIG['dataset']}.csv", sep=',')
    cols_to_skip = ['Search_time', 'Best_value']
    for c in best_params.columns:
        if c not in cols_to_skip:
            CONFIG[c.lower()] = best_params[c].item()

def main():
    print("-> Loading data")
    # Data loaders
    print("----> Loading data")
    if CONFIG['dataset']=='tcga':
        train, test = get_tcga_datasets(scaler_type='standard')  
        # CONFIG['x_dim'] = 974 # 974 landmark genes in gtex
    elif CONFIG['dataset'] =='gtex':
        train, test = get_gtex_datasets(scaler_type='standard')
        CONFIG['vocab_size'] = 26 # 26 tissue types in gtex
        CONFIG['x_dim'] = 974 # 974 landmark genes in gtex
    elif CONFIG['dataset'] =='cambda':
        train, test = get_cambda_datasets(scaler_type='standard')
        CONFIG['x_dim'] = 978
    # Update config with correct dimensions
    print(f"Dimensions for CAMBDA: x_dim={CONFIG['x_dim']}, vocab_size={CONFIG['vocab_size']}")


    # Dataloader
    print("----> Building dataloaders")
    train_loader, test_loader = build_loaders(train, test, config=CONFIG)
    for batch, labels in iter(train_loader):
        CONFIG['x_dim'] = batch.shape[1]  # Update input dimension from data
        CONFIG['vocab_size'] = labels.shape[1]  # Update vocabulary size from labels
        print(f"Updated dimensions for CAMBDA: x_dim={CONFIG['x_dim']}, vocab_size={CONFIG['vocab_size']}")
        break

    # Model
    print(f"--> Loading WGAN-GP.")
    model = WGAN(CONFIG)
    print(f"--> Training WGAN-GP.")
    model.train(train_loader, test_loader,  
                z_dim=CONFIG['latent_dim'], 
                epochs=CONFIG['epochs'], 
                iters_critic=CONFIG['iters_critic'], 
                lambda_penalty=CONFIG['lambda_penalty'], 
                step = CONFIG['step'],
                verbose=True, 
                checkpoint_dir=CONFIG['checkpoint_dir'], 
                log_dir=CONFIG['log_dir'], 
                fig_dir = CONFIG['fig_dir'],
                prob_success=CONFIG['prob_success'], 
                norm_scale=CONFIG['norm_scale'],
                optimizer = CONFIG['optimizer'],
                lr_g = CONFIG['lr_g'],
                lr_d = CONFIG['lr_d'],
               config=CONFIG,
               hyperparameters_search=False)
    
    training_time = model.time_sec
    
    ############### Prec/Recall/Density/Coverage/AAts 5 runs on train data ###############
    PREC = []
    RECALL = []
    DENSITY = []
    COVERAGE = []
    AATS = []
    CORR = []
    FD =[]
    
    for i in range(5):
        # Get new generated data batch
        x_real, x_gen = model.generate(train_loader, return_labels=False)
        x_real, x_gen = x_real.numpy(), x_gen.numpy()

        # Precision/Recall/Density/Coverage
        prec, recall, dens, cov = compute_prdc(x_real, x_gen, CONFIG['nb_nn'])
        # Adversarial accuracy
        sample_size = min(len(x_real), 4096)
        idx = np.random.choice(len(x_real), sample_size, replace=False)
        _, _, adversarial = compute_AAts(real_data=x_real[idx], fake_data=x_gen[idx])
        # Correlations
        corr = gamma_coeff_score(x_real, x_gen)
        # Frechet
        frechet = 0 # compute_frechet_distance_score(x_real, x_gen, dataset=CONFIG['dataset'], device=CUDA_DEVICE2, to_standardize=False)

        PREC.append(prec)
        RECALL.append(recall)
        DENSITY.append(dens)
        COVERAGE.append(cov)
        AATS.append(adversarial)
        CORR.append(corr)
        FD.append(frechet)
        
    # Statistics
    final_prec = np.mean(PREC)    
    final_prec_std = np.std(PREC)    
    
    final_rec = np.mean(RECALL)    
    final_rec_std = np.std(RECALL)    
    
    final_dens = np.mean(DENSITY)    
    final_dens_std = np.std(DENSITY)    
    
    final_cov = np.mean(COVERAGE)    
    final_cov_std = np.std(COVERAGE)  
    
    final_aats = np.mean(AATS)    
    final_aats_std = np.std(AATS)  

    final_corr = np.mean(CORR)    
    final_corr_std = np.std(CORR) 

    final_fd = np.mean(FD)    
    final_fd_std = np.std(FD)  

    # save results
    dict_res_5runs = {'precision': [final_prec, final_prec_std],
                        'recall': [final_rec, final_rec_std],
                        'density': [final_dens, final_dens_std]  ,
                        'coverage': [final_cov, final_cov_std]  ,
                        'aats':  [final_aats, final_aats_std],
                        'correlation_score':[final_corr, final_corr_std],
                         'frechet': [final_fd, final_fd_std] }
    np.save(model.log_dir+'/dict_res_metrics_5runs.npy', dict_res_5runs)
     
    # Get new generated data batch
    x_real, x_gen, y = model.generate(train_loader, return_labels=True)
    x_real_test, x_gen_test, y_test = model.generate(test_loader, return_labels=True)

    x_real, x_gen, y = x_real.numpy(), x_gen.numpy(), y.numpy()
    x_real_test, x_gen_test, y_test = x_real_test.numpy(), x_gen_test.numpy(), y_test.numpy()

    np.save("x_real_gan.npy", x_real)
    np.save("x_gen_gan.npy", x_gen)
    np.save("y.npy", y)

    np.save("x_real_gan_test.npy", x_real_test)
    np.save("x_gen_gan_test.npy", x_gen_test)
    np.save("y_test.npy", y_test)

    print("train saved")
    print("test saved")
    
    ############### Training history #####################
    train_history_path = model.log_dir+'/train_history.npy'
    train_history = np.load(train_history_path, allow_pickle=True).item()
    
    LOG_DIR = model.log_dir

    # Save to csv
    print("----> Saving results to csv")
    
    d = {'dataset': DATASET,
         'model_folder_id' : LOG_DIR,
         'train_history_path': train_history_path,
        'latent_dim': CONFIG['latent_dim'], 
         'hidden_dim1_g': CONFIG['hidden_dim1_g'], 
         'hidden_dim2_g': CONFIG['hidden_dim2_g'],
         'hidden_dim3_g': CONFIG['hidden_dim3_g'], 
         'hidden_dim4_g': CONFIG['hidden_dim4_g'],
         'hidden_dim5_g': CONFIG['hidden_dim5_g'],
         'hidden_dim1_d': CONFIG['hidden_dim1_d'],
         'hidden_dim2_d': CONFIG['hidden_dim2_d'],
         'batch_size': CONFIG['batch_size'], 
         'epochs': CONFIG['epochs'], 
         'iters_critic': CONFIG['iters_critic'], 
         'lambda_penalty': CONFIG['lambda_penalty'], 
         'prob_success': CONFIG['prob_success'], 
         'norm_scale': CONFIG['norm_scale'],
         'activation_func':CONFIG['activation'],
         'negative_slope':CONFIG['negative_slope'],
         'optimizer':CONFIG['optimizer'],
         'lr_d':CONFIG['lr_d'],
         'lr_g':CONFIG['lr_g'],
         'spectral_norm':CONFIG['sn'],
        'batch_norm': CONFIG['bn'],
         'device': DEVICE,
         'training_time': training_time,
         'training_time_hour': training_time/3600,
         'disc_loss': train_history['disc_loss_epoch'][-1],
        'g_loss': train_history['g_loss_epoch'][-1],
        'gp': train_history['gp'][-1],
         'correlation_train:': [[final_corr, final_corr_std]],
         'precision_train': [[final_prec, final_prec_std]],
         'recall_train': [[final_rec, final_rec_std]],
         'density_train': [[final_dens, final_dens_std]],
         'coverage_train': [[final_cov, final_cov_std]],
         'aats_train': [[final_aats, final_aats_std]],
         'frechet_train':[[final_fd, final_fd_std]]
        }
    
    # Build dataframe
    df_temp = pd.DataFrame(data=d, index=[0])
    df = df_temp
    # Check if file exists and has content
    if os.path.exists(PATH_RESULTS_DATAFRAME) and os.path.getsize(PATH_RESULTS_DATAFRAME) > 0:
        df = pd.read_csv(PATH_RESULTS_DATAFRAME, sep=',')
        # Merge dataframes
        df = pd.concat([df, df_temp])

    # Save the updated dataframe
    df.to_csv(PATH_RESULTS_DATAFRAME, sep=',', header=True, index=False)
    
# Run function    
if __name__ == '__main__':
    main()
