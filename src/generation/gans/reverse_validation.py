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
from src.generation.gans.utils import get_tcga_datasets, get_gtex_datasets, build_loaders, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Import model config and hyperparameters
from src.generation.gans.config import CONFIG
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
                        help="Indicate if WGAN-GP model should be trained with best params (required).")
parser.add_argument("-nb_runs", "--nb_runs",
                        dest = "nb_runs",
                        type = int,
                        required = True,
                        help="Number of runs for the generation (required).")
parser.add_argument("-gpu_gan", "--gpu_gan",
                        dest = "gpu_gan",
                        type = str,
                        required = True,
                        help="Specify the GPU device to use for training the GAN (required)")
parser.add_argument("-gpu_mlp", "--gpu_mlp",
                        dest = "gpu_mlp",
                        type = str,
                        required = True,
                        help="Specify the GPU device to use for training the MLP classifier (required)")
args = parser.parse_args()
DATASET = args.dataset
WBP = args.with_best_params
NB_RUNS = args.nb_runs
DEVICE_GAN = args.gpu_gan
DEVICE_MLP = args.gpu_mlp

# Cuda device
if "cuda" in DEVICE_GAN:
    CUDA_DEVICE = int(DEVICE_GAN.strip("cuda:"))
else:
    CUDA_DEVICE = "cpu"

CONFIG['device'] = CUDA_DEVICE

# Cuda device MLP
if "cuda" in DEVICE_MLP:
    CUDA_DEVICE_MLP = int(DEVICE_MLP.strip("cuda:"))
else:
    CUDA_DEVICE_MLP = "cpu"

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

path_res_gans = './src/generation/gans/results/results_wgangp.csv'
df_res_gans = pd.read_csv(path_res_gans, sep=',')
df_res_gans = df_res_gans.drop(0)
if DATASET=='gtex':
    i = 0
elif DATASET=='tcga':
    i = 1
MODEL_ID = df_res_gans['model_folder_id'].iloc[i].strip('./logs/')

# Config classifier
if DATASET=='tcga':
    CONFIG_CLS = CONFIGS[1]
elif DATASET=='gtex':
    CONFIG_CLS = CONFIGS[2]

CONFIG_CLS['device'] = CUDA_DEVICE_MLP

# New path to store mlp
CONFIG_CLS['path'] = f'./src/generation/gans/results/cls_fake_{DATASET.lower()}.pth'

######### 0. Load data #########
print("-> Loading data")
# Data loaders
print("----> Loading data")
if CONFIG['dataset']=='tcga':
    train, test = get_tcga_datasets(scaler_type='standard')  
elif CONFIG['dataset'] =='gtex':
    train, test = get_gtex_datasets(scaler_type='standard')
    CONFIG['vocab_size'] = 26 # 26 tissue types in gtex
    CONFIG['x_dim'] = 974 # 974 landmark genes in gtex

print("----> Building true dataloaders")
train_loader, test_loader = build_loaders(train, test, config=CONFIG)

########## 1. Load generative model #########
print(f"--> Loading WGAN-GP.")
model = WGAN(CONFIG)
model.load_generator(path=f'./src/generation/gans/checkpoints/{MODEL_ID}/_gen.pt', location=torch.device(CUDA_DEVICE))

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

    ACC_TF_TT.append(acc_test)
    F1_TF_TT.append(f1_score_test)
    AUC_TF_TT.append(auc_test)

    ########## 5. Test pretrained classifier on generated data #########

    # Path where pretrained weights are stored    
    path_model = f'/home/alacan/scripts/gerec_pipeline/classification/results/model_{DATASET.lower()}.pth'
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
    'acc_train_fake_test_fake': np.mean(ACC_TF_TF),
    'f1_score_train_fake_test_fake': np.mean(F1_TF_TF),
    'auc_train_fake_test_fake': np.mean(AUC_TF_TF),

    'acc_train_fake_test_fake_std': np.std(ACC_TF_TF),
    'f1_score_train_fake_test_fake_std':np.std(F1_TF_TF),
    'auc_train_fake_test_fake_std': np.std(AUC_TF_TF),

    'acc_train_fake_test_true':  np.mean(ACC_TF_TT),
    'f1_score_train_fake_test_true': np.mean(F1_TF_TT),
    'auc_train_fake_test_true': np.mean(AUC_TF_TT),

    'acc_train_fake_test_true_std':  np.std(ACC_TF_TT),
    'f1_score_train_fake_test_true_std': np.std(F1_TF_TT),
    'auc_train_fake_test_true_std': np.std(AUC_TF_TT),

    'acc_train_true_test_fake': np.mean(ACC_TT_TF),
    'f1_score_train_true_test_fake': np.mean(F1_TT_TF),
    'auc_train_true_test_fake': np.mean(AUC_TT_TF),

    'acc_train_true_test_fake_std': np.std(ACC_TT_TF),
    'f1_score_train_true_test_fake_std': np.std(F1_TT_TF),
    'auc_train_true_test_fake_std': np.std(AUC_TT_TF),

}

df_res = pd.DataFrame(data=d,
                      columns= d.keys(),
                      index=[0])
# Save
df_res.to_csv(f'./src/generation/gans/results/reverse_validation_{DATASET}.csv')
print(f"----> End. ")