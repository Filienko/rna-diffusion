# Imports
import os
import random
import sys
import time as t
import torch
import argparse
import numpy as np
import pandas as pd
from src.generation.gans.model import WGAN
from src.reconstruction.model import DGEX, mae, mse
from src.reconstruction.utils import get_datasets_split_landmarks_for_search
from src.metrics.precision_recall import compute_prdc, get_precision_recall
from src.metrics.aats import compute_AAts
from src.metrics.correlation_score import gamma_coeff_score
from sklearn.preprocessing import StandardScaler
import torch.utils.data as data_utils
# Import model config and hyperparameters
from src.generation.gans.config import CONFIG
# Import MLP config
from src.reconstruction.config import CONFIGS

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
                        help="Specify the GPU device to use for the generative model (required)")
parser.add_argument("-device2", "--device2",
                        dest = "device2",
                        type = str,
                        required = True,
                        help="Specify the device to use for reconstruction model (required)")

args = parser.parse_args()
DATASET = args.dataset
DEVICE = args.gpu_device
DEVICE2 = args.device2
NB_RUNS = args.nb_runs
NB_NN = args.nb_nn

# Cuda device
if "cuda" in DEVICE:
    CUDA_DEVICE = int(DEVICE.strip("cuda:"))
else:
    CUDA_DEVICE = "cpu"

if "cuda" in DEVICE2:
    CUDA_DEVICE2 = DEVICE2
else:
    CUDA_DEVICE2 = "cpu"

# Generative model configuration
CONFIG['device'] = CUDA_DEVICE
# Dataset
CONFIG['dataset'] = DATASET

# Best params config
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
    CONFIG['vocab_size'] = 26 # 26 tissue types in gtex
    CONFIG['x_dim'] = 974 # 974 landmark genes in gtex
elif DATASET=='tcga':
    i = 1
MODEL_ID = df_res_gans['model_folder_id'].iloc[i].strip('./logs/')

# Config for DGEX
if DATASET=='tcga':
    CONFIG_NB=1
elif DATASET=='gtex':
    CONFIG_NB=2
CONFIG_MLP = CONFIGS[CONFIG_NB]
CONFIG_MLP['device'] = CUDA_DEVICE2

df_best_params = pd.read_csv(f'./src/reconstruction/results/best_params_search_{DATASET}.csv')
CONFIG_MLP['lr'] = float(df_best_params['lr'].item())
CONFIG_MLP['optimizer'] = str(df_best_params['optimizer'].item())
CONFIG_MLP['batch_size'] = int(df_best_params['batch_size'].item())
CONFIG_MLP['dropout'] = float(df_best_params['dropout'].item()) 
# Init list dims with input dim
CONFIG_MLP['list_dims'] = [CONFIG_MLP['input_dim']]
CONFIG_MLP['n_blocks'] = int(df_best_params['n_blocks'].item())
for i in range(1, CONFIG_MLP['n_blocks']+1):
    CONFIG_MLP['list_dims'].append(int(df_best_params[f"hidden_dim{i}"].item()))
# Add output dim as last dim
CONFIG_MLP['list_dims'].append(CONFIG_MLP['output_dim'])


# Load true data in original dimensions
print("----> Loading true data")
X, y, _, _, tissues_train, _ = get_datasets_split_landmarks_for_search(DATASET, landmark=False, split_landmark=True, with_tissues=True)
# Scale the data
scaler = StandardScaler()
scaler_nl = StandardScaler()
X = scaler.fit_transform(X)
y = scaler_nl.fit_transform(y)

# Build dataloader for WGAN-GP
# Turn data into tensors and tensor datasets
X = torch.tensor(X).type(torch.float)
tissues_train = torch.from_numpy(np.array(tissues_train))

# Full true data
full_true = np.concatenate((X.numpy(), y), axis=1)

# Build fake data loaders
config={'batch_size':2048, 'num_workers':2}
X = data_utils.DataLoader(torch.utils.data.TensorDataset(X, tissues_train),
                                        batch_size=config['batch_size'],
                                        shuffle=True,
                                        num_workers=config['num_workers'],
                                        pin_memory=True,
                                        prefetch_factor=2,
                                        persistent_workers=False)

# Init generative model
print(f"--> Loading WGAN-GP.")
model = WGAN(CONFIG)
model.load_generator(path=f'./src/generation/gans/checkpoints/{MODEL_ID}/_gen.pt', location=torch.device(CUDA_DEVICE))

# Init metrics
PREC, REC, DENS, COV, AATS, CORR = [],[],[],[],[],[]

# Loop over runs
print(f"----> Start with {NB_RUNS} runs")
for i in range(5):
    # Generate landmark genes
    _, x_fake, y_fake = model.generate(X, return_labels=True)

    # Reconstruct target genes
    print("----> Load reconstructing model")
    GeRec = DGEX(CONFIG_MLP)
    GeRec.load_state_dict(torch.load(f'./src/reconstruction/results/model_{DATASET}.pth', map_location=CONFIG_MLP['device']))
    GeRec = GeRec.to(CONFIG_MLP['device'])
    x_fake= x_fake.to(CONFIG_MLP['device']).float()
    
    # Reconstruct
    with torch.no_grad():
        GeRec.eval()
        start = t.time()
        preds_fake = GeRec(x_fake).detach().cpu().numpy()
    recon_time = t.time() - start
    # Concatenate
    x_fake = np.concatenate((x_fake.cpu().numpy(), preds_fake), axis=1)

    # Precision/Recall/Density/Coverage
    print("PRDC...")
    # prec, recall, dens, cov = compute_prdc(full_true, x_fake, NB_NN)
    prec, recall = get_precision_recall(torch.from_numpy(full_true), torch.from_numpy(x_fake), [NB_NN])
    PREC.append(prec)
    REC.append(recall)
    # DENS.append(dens)
    # COV.append(cov)
    # Adversarial accuracy
    print("AATS...")
    idx = np.random.choice(len(full_true), 2048, replace=False) # Sample random data
    _, _, aa = compute_AAts(real_data=full_true[idx], fake_data=x_fake[idx])
    # Correlations
    print("Correlations...")
    corr = gamma_coeff_score(full_true[idx], x_fake[idx])
    AATS.append(aa)
    CORR.append(corr)

    # Save reconstructed data
    if i==0:
        if DATASET=='tcga':
            landmark_genes_id = pd.read_csv(f"/home/alacan/data_RNAseq_RTCGA/landmark_genes_ids.csv").values.flatten()
            non_landmark_genes_id = pd.read_csv(f"/home/alacan/data_RNAseq_RTCGA/non_landmark_genes_ids.csv").values.flatten()

        elif DATASET=='gtex':
            df_descript = pd.read_csv('/home/alacan/scripts/gerec_pipeline/gtex_description.csv', sep=',')
            # df_descript['entrez_id'] = df_descript['entrez_id'].astype('str')
            landmark_genes_id = df_descript[df_descript.Type=='landmark']['Description'].values.flatten()
            non_landmark_genes_id = df_descript[df_descript.Type=='target']['Description'].values.flatten()
        
        pd.DataFrame(data=x_fake, columns=np.concatenate((landmark_genes_id, non_landmark_genes_id))).to_csv(f'./src/reconstruction/results/fake_from_gan_reconstructed_mlp_{DATASET}.csv')
        pd.DataFrame(data=y_fake.cpu().numpy().argmax(1).reshape(-1,1), columns=['tissue_type']).to_csv(f'./src/reconstruction/results/fake_from_gan_reconstructed_mlp_{DATASET}_labels.csv')

    # memory
    x_fake = []


# Save results
print(f"----> Store best results.")
data = [recon_time]
data_cols = ['reconstruction_time']

df_res = pd.DataFrame(columns=data_cols,
                        data=np.array([data]))

df_res['precision'], df_res['precision_std'] = np.mean(PREC), np.std(PREC)
df_res['recall'], df_res['recall_std'] = np.mean(REC), np.std(REC)
# df_res['density'], df_res['density_std'] = np.mean(DENS), np.std(DENS)
# df_res['coverage'], df_res['coverage_std'] = np.mean(COV), np.std(COV)
df_res['aats'], df_res['aats_std'] = np.mean(AATS), np.std(AATS)
df_res['correlation'], df_res['correlation_std'] = np.mean(CORR), np.std(CORR)

df_res.to_csv(f'./src/reconstruction/results/results_mlp_{DATASET}_from_gan.csv', index=False)
print(f"----> End. ")