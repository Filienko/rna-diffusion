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
from src.generation.vae.utils import get_tcga_datasets, get_gtex_datasets, build_loaders
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Import model config and hyperparameters
from src.generation.vae.config import CONFIG
# Import classifier config and hyperparameters
# sys.path.append("../")
from src.classification.model import TissuePredictor
from src.classification.config import CONFIGS
from src.classification.utils import get_class_weights

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
                        help="Indicate if model should be trained with best params (required).")
parser.add_argument("-nb_runs", "--nb_runs",
                        dest = "nb_runs",
                        type = int,
                        required = True,
                        help="Number of runs for the generation (required).")
parser.add_argument("-gpu_vae", "--gpu_vae",
                        dest = "gpu_vae",
                        type = str,
                        required = True,
                        help="Specify the GPU device to use for training the VAE (required)")
parser.add_argument("-gpu_mlp", "--gpu_mlp",
                        dest = "gpu_mlp",
                        type = str,
                        required = True,
                        help="Specify the GPU device to use for training the MLP classifier (required)")
args = parser.parse_args()
DATASET = args.dataset
WBP = args.with_best_params
NB_RUNS = args.nb_runs
DEVICE_VAE = args.gpu_vae
DEVICE_MLP = args.gpu_mlp

# Cuda device
if "cuda" in DEVICE_VAE:
    CUDA_DEVICE = int(DEVICE_VAE.strip("cuda:"))
else:
    CUDA_DEVICE = "cpu"

CONFIG['device'] = CUDA_DEVICE

# Cuda device MLP
if "cuda" in DEVICE_MLP:
    CUDA_DEVICE_MLP = int(DEVICE_MLP.strip("cuda:"))
else:
    CUDA_DEVICE_MLP = "cpu"

# Config classifier
if DATASET=='tcga':
    CONFIG_CLS = CONFIGS[1]
elif DATASET=='gtex':
    CONFIG_CLS = CONFIGS[2]

CONFIG_CLS['device'] = CUDA_DEVICE_MLP

# New path to store mlp
CONFIG_CLS['path'] = f'./src/generation/vae/results/cls_fake_{DATASET.lower()}.pth'

######### 0. Load data #########
# Data loaders
print("----> Loading data")
CONFIG['dataset'] = DATASET
if CONFIG['dataset']=='tcga':
    train, test = get_tcga_datasets(scaler_type='standard')  
elif CONFIG['dataset'] =='gtex':
    train, test = get_gtex_datasets(scaler_type='standard')
    CONFIG['vocab_size'] = 26 # 26 tissue types in gtex
    CONFIG['x_dim'] = 974 # 974 landmark genes in gtex

# Best params config
if WBP=='y':
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

print("----> Building true dataloaders")
train_loader, test_loader = build_loaders(train, test, config=CONFIG)

########## 1. Load generative model #########
print(f"--> Loading VAE.")
model = model = VAE(CONFIG)
model.load_decoder(path=f'./src/generation/vae/checkpoints/dec_{DATASET.lower()}.pt', location=torch.device(CUDA_DEVICE))
model = model.to(torch.device(CUDA_DEVICE))

# Init
ACC_TF_TF, F1_TF_TF, AUC_TF_TF = [], [], []
ACC_TT_TF, F1_TT_TF, AUC_TT_TF = [], [], []
ACC_TF_TT, F1_TF_TT, AUC_TF_TT = [], [], []

for i in range(NB_RUNS):

    ########## 2. Generate data and processing #########
    x_real, x_gen, y = model.generate(train_loader, return_labels=True)

    # model = []
    # train_loader = []

    # Train-val split
    X_train, X_val, y_train, y_val = train_test_split(x_gen, y, stratify=y, test_size=0.2, random_state=42)

    # Turn data into tensors and tensor datasets
    X_train = torch.tensor(X_train).type(torch.float)
    X_val = torch.tensor(X_val).type(torch.float)

    # Build fake data loaders
    train_loader_fake, val_loader_fake = build_loaders(torch.utils.data.TensorDataset(X_train, y_train) , 
                                                       torch.utils.data.TensorDataset(X_val, y_val), 
                                                       config=CONFIG_CLS)
    # Weights
    TISSUE_WEIGHTS = get_class_weights(train_loader_fake, val_loader_fake)

    # How to standardize the data ?

    ########## 3. Load classifier #########
    cls = TissuePredictor(CONFIG_CLS, DATASET)

    ########## 4. Train classifier #########
    # Model training
    cls.train(train_loader_fake, 
                val_loader_fake,  
                verbose=2, 
                class_weights=TISSUE_WEIGHTS)

    # Test on fake val data
    acc_val, f1_score_val, auc_val, cm_val = cls.test(val_loader_fake, output='metric')

    ACC_TF_TF.append(acc_val)
    F1_TF_TF.append(f1_score_val)
    AUC_TF_TF.append(auc_val)

    # Test on true test data
    acc_test, f1_score_test, auc_test, cm_test = cls.test(test_loader, output='metric')

    print('acc_test:', acc_test)

    ACC_TF_TT.append(acc_test)
    F1_TF_TT.append(f1_score_test)
    AUC_TF_TT.append(auc_test)

    ########## 5. Test pretrained classifier on generated data #########

    # Path where pretrained weights are stored    
    # path_model = f'/home/alacan/scripts/gerec_pipeline/classification/results/model_{DATASET.lower()}.pth'
    path_model = f'/home/alacan/scripts/classification/landmarks/results/model_{DATASET.lower()}.pth'
    # Instantiate model
    cls = TissuePredictor(CONFIG_CLS, DATASET)
    # Load pretrained model
    cls.model.load_state_dict(torch.load(path_model, map_location=torch.device(CUDA_DEVICE_MLP)))

    # To standardize (centering and reduction before MLP)
    # if to_standardize:
    #     scaler = StandardScaler()
    #     real_data = scaler.fit_transform(real_data)

    acc_train_true_test_fake, f1_score_train_true_test_fake, auc_train_true_test_fake, cm_train_true_test_fake = cls.test(train_loader_fake, output='metric')

    ACC_TT_TF.append(acc_train_true_test_fake)
    F1_TT_TF.append(f1_score_train_true_test_fake)
    AUC_TT_TF.append(auc_train_true_test_fake)


d = {
    'acc_train_fake_test_fake': ACC_TF_TF,
    'f1_score_train_fake_test_fake': F1_TF_TF,
    'auc_train_fake_test_fake': AUC_TF_TF,

    'acc_train_fake_test_true':  ACC_TF_TT,
    'f1_score_train_fake_test_true': F1_TF_TT,
    'auc_train_fake_test_true': AUC_TF_TT,

    'acc_train_true_test_fake': ACC_TT_TF,
    'f1_score_train_true_test_fake': F1_TT_TF,
    'auc_train_true_test_fake': AUC_TT_TF,
}

df_res = pd.DataFrame(data=d,
                      columns= d.keys())
# Save
df_res.to_csv(f'./src/generation/vae/results/reverse_validation_{DATASET}.csv')
print(f"----> End. ")