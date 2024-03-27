# Imports
import sys
import os
import time as t
import argparse
import numpy as np
import pandas as pd
import torch
import random
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.generation.large_dim.vae.utils import build_loaders, get_datasets_for_search, split_and_scale_datasets, get_tcga_datasets, get_gtex_datasets
# Import classifier config and hyperparameters
# sys.path.append("../")
from src.classification.large_dim.model import TissuePredictor
from src.classification.large_dim.config import CONFIGS
from src.classification.large_dim.utils import get_class_weights

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
parser.add_argument("-gen_model", "--gen_model",
                        dest = "gen_model",
                        type = str,
                        required = True,
                        help="Indicate the generative model used. (required).")
parser.add_argument("-recon_model", "--recon_model",
                        dest = "recon_model",
                        type = str,
                        required = True,
                        help="Indicate the reconstruction model used. (required).")
parser.add_argument("-nb_runs", "--nb_runs",
                        dest = "nb_runs",
                        type = int,
                        required = True,
                        help="Number of runs for the generation (required).")
parser.add_argument("-gpu_device", "--gpu_device",
                        dest = "gpu_device",
                        type = str,
                        required = True,
                        help="Specify the GPU device to use for training the MLP classifier (required)")
args = parser.parse_args()
DATASET = args.dataset
GEN_MODEL = args.gen_model
RECON_MODEL = args.recon_model
NB_RUNS = args.nb_runs
DEVICE_MLP = args.gpu_device

# Cuda device
if "cuda" in DEVICE_MLP:
    CUDA_DEVICE = DEVICE_MLP
else:
    CUDA_DEVICE = "cpu"

# Config classifier
if DATASET=='tcga':
    CONFIG_CLS = CONFIGS[1]
elif DATASET=='gtex':
    CONFIG_CLS = CONFIGS[2]

CONFIG_CLS['device'] = CUDA_DEVICE

# New path to store mlp
CONFIG_CLS['path'] = f'./src/reconstruction/results/cls_fake_reconstruction.pth'

######### 1. Load data #########

# Data loaders
print("----> Loading reconstructed data")
path_to_data =f'./src/reconstruction/results/fake_from_{GEN_MODEL.lower()}_reconstructed_{RECON_MODEL.lower()}_{DATASET}.csv'
path_to_labels =f'./src/reconstruction/results/fake_from_{GEN_MODEL.lower()}_reconstructed_{RECON_MODEL.lower()}_{DATASET}_labels.csv'

x_gen = pd.read_csv(path_to_data, sep=',').to_numpy()[:,1:]
y_gen = pd.read_csv(path_to_labels, sep=',')['tissue_type'].values

y = np.zeros((x_gen.shape[0], y_gen.max()+1))
y[np.arange(len(y)), y_gen.flatten()] = 1

print('x_gen', x_gen.shape)
print('y', y.shape)

if DATASET=='tcga':
    train, test = get_tcga_datasets(scaler_type='standard')  
elif DATASET =='gtex':
    train, test = get_gtex_datasets(scaler_type='standard')

# True loader
_, test_loader = build_loaders(train, test, config=CONFIG_CLS)
train=[] # free memory

# Init
ACC_TF_TF, F1_TF_TF, AUC_TF_TF = [], [], []
ACC_TT_TF, F1_TT_TF, AUC_TT_TF = [], [], []
ACC_TF_TT, F1_TF_TT, AUC_TF_TT = [], [], []

for i in range(NB_RUNS):

    ########## 2. Processing #########

    # Train-val split
    X_train, X_val, y_train, y_val = train_test_split(x_gen, y, stratify=y, test_size=0.2, random_state=42)
    
    # Turn data into tensors and tensor datasets
    X_train = torch.tensor(X_train).type(torch.float)
    X_val = torch.tensor(X_val).type(torch.float)
    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)
    # Build fake data loaders
    train_loader_fake, val_loader_fake = build_loaders(torch.utils.data.TensorDataset(X_train, y_train) , 
                                                        torch.utils.data.TensorDataset(X_val, y_val), 
                                                        config=CONFIG_CLS)
    # free memory
    X_train, y_train = [], []
    X_val, y_val = [], []
    
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

    ACC_TF_TT.append(acc_test)
    F1_TF_TT.append(f1_score_test)
    AUC_TF_TT.append(auc_test)

    print('Test Acc:', acc_test)

    ########## 5. Test pretrained classifier on generated data #########

    # Path where pretrained weights are stored    
    # path_model = f'/home/alacan/scripts/gerec_pipeline/classification/results/model_{DATASET.lower()}.pth'
    path_model = f'./src/classification/large_dim/results/model_{DATASET.lower()}.pth'
    # Instantiate model
    cls = TissuePredictor(CONFIG_CLS, DATASET)
    # Load pretrained model
    cls.model.load_state_dict(torch.load(path_model, map_location=CUDA_DEVICE))

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
df_res.to_csv(f'./src/reconstruction/results/reverse_val/reverse_validation_{DATASET}_from_{GEN_MODEL}_reconstructed_{RECON_MODEL}.csv')
print(f"----> End. ")