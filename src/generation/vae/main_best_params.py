# Imports
import sys
import os
import time as t
import argparse
import numpy as np
import pandas as pd
import torch
import random
from src.generation.vae.model import VAE
from src.generation.vae.runner import Trainer
from src.generation.vae.utils import get_tcga_datasets, get_gtex_datasets, build_loaders
# Import model config and hyperparameters
from src.generation.vae.config import CONFIG
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
                        help="Specify the dataset to use for training (required).")

parser.add_argument("-gpu_device", "--gpu_device",
                        dest = "gpu_device",
                        type = str,
                        required = True,
                        help="Specify the GPU device to use for training (required)")
parser.add_argument("-gpu_device2", "--gpu_device2",
                        dest = "gpu_device2",
                        type = str,
                        required = True,
                        help="Specify the GPU device to use for metrics (FD) (required)")
args = parser.parse_args()
DEVICE = args.gpu_device
DEVICE2 = args.gpu_device2
DATASET = args.dataset
NB_TRIALS = args.nb_trials

# Cuda device
if "cuda" in DEVICE:
    CUDA_DEVICE = int(DEVICE.strip("cuda:"))
else:
    CUDA_DEVICE = "cpu"

CONFIG['device'] = CUDA_DEVICE

def main():
    """
    Main function to run search.
    """
    # Functions
    def build_model(cfg):
        """
        Instantiate model.
        ----
        Parameters:
            cfg (dict) = model configuration
        Returns:
            model as torch.nn.module
        """
        model = VAE(cfg)
        return model

    # Data loaders
    print("----> Loading data")
    CONFIG['dataset'] = DATASET
    if DATASET=='tcga':
        train, test = get_tcga_datasets(scaler_type='standard')  
    elif DATASET =='gtex':
        train, test = get_gtex_datasets(scaler_type='standard')
        CONFIG['vocab_size'] = 26 # 26 tissue types in gtex
        CONFIG['x_dim'] = 974 # 974 landmark genes in gtex

    # Best params config
    best_params = pd.read_csv(f"./src/generation/vae/results/tissue_search_{DATASET}.csv", sep=',')
    CONFIG['lr'] = float(best_params['lr'].item())
    CONFIG['optimizer'] =  str(best_params['optimizer'].item())
    CONFIG['batch_size'] =  int(best_params['batch_size'].item())
    # Init list dims with input dim
    CONFIG['list_dims_encoder'] = [CONFIG['x_dim']]
    CONFIG['latent_dim'] = int(best_params['latent_dim'].item())
    CONFIG['list_dims_decoder'] = [CONFIG['latent_dim']]
    CONFIG['n_blocks'] = int(best_params['n_blocks'].item())
    for i in range(1, CONFIG['n_blocks']+1):
        CONFIG['list_dims_encoder'].append(int(best_params[f"hidden_dim{i}"].item()))
    # Add latent dim as last dim
    CONFIG['list_dims_encoder'].append(CONFIG['latent_dim'])
    for i in range(2, CONFIG['n_blocks']+2):
        CONFIG['list_dims_decoder'].append(CONFIG['list_dims_encoder'][-i])
    # Add output dim as last dim
    CONFIG['list_dims_decoder'].append(CONFIG['x_dim'])

    # Dataloader
    print("----> Building dataloaders")
    train_loader, _ = build_loaders(train, test, config=CONFIG)
    model = build_model(CONFIG)
    print(model)
    trainer = Trainer(CONFIG, model)
    # Model training
    start_time = t.time()
    model = trainer(train_loader, 
                    verbose=2)
    # End time
    end_time = t.time()
    train_time =end_time - start_time
    print(f"train_time: {round(train_time, 2)} sec. (= {round(train_time/60, 2)} min.)")  

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
        idx = np.random.choice(len(x_real), 4096, replace=False) # Sample random data
        _, _, adversarial = compute_AAts(real_data=x_real[idx], fake_data=x_gen[idx])
        # Correlations
        corr = gamma_coeff_score(x_real, x_gen)
        # Frechet
        frechet = compute_frechet_distance_score(x_real, x_gen, dataset=CONFIG['dataset'], device=DEVICE2, to_standardize=False)

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
    np.save(f'./logs/dict_res_metrics_5runs_{DATASET}.npy', dict_res_5runs)

    ############### Training history #####################
    train_history = pd.read_csv(f'./src/generation/vae/logs/train_metrics_{DATASET}.csv', sep=',')
    
    # Save to csv
    print("----> Saving results to csv")

    df = pd.DataFrame(columns=CONFIG.keys(),
                                data=CONFIG.values())
    
    # Add other results
    df['model_path'] = CONFIG['checkpoint_dir']+f'/dec_{DATASET}.pt'
    df['train_time'] = train_time
    df['training_time_hour'] = train_time/3600
    df['loss'] = train_history['train_loss'].values[-1]
    df['recon_loss'] = train_history['train_mse'].values[-1]
    df['kl'] = train_history['train_kl'].values[-1]
    df['correlation_train'] =final_corr
    df['correlation_train_std'] =final_corr_std
    df['precision_train'] =final_prec
    df['precision_train_std'] =final_prec_std
    df['recall_train'] =final_rec
    df['recall_train_std'] =final_rec_std
    df['density_train'] =final_dens
    df['density_train_std'] =final_dens_std
    df['coverage_train'] =final_cov
    df['coverage_train_std'] =final_cov_std
    df['aats_train'] =final_aats
    df['aats_train_std'] =final_aats_std
    df['frechet_train'] =final_fd
    df['frechet_train_std'] =final_fd_std

    df.to_csv('./src/generation/vae/results/results_{DATASET}.csv', sep =',', header=True, index=False)
    
# Run function    
if __name__ == '__main__':
    main()