""" Numpy implementation of the Frechet Distance Score.

Reference:
-----
Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017).
GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.
NIPS.


Original code:
-----
https://github.com/bioinf-jku/TTUR.git
"""

# Imports
import os
import sys
# from benchmark_configs import CONFIG_DICT
# from benchmark_model_cls import CancerPredictor
import numpy as np
import random
import torch
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.preprocessing import StandardScaler
# Import model config and hyperparameters for benchmark classifiers
# sys.path.append(os.path.abspath("../../../"))
# # from model import TissuePredictor
# # from config import CONFIGS
# from metrics.pretrained_mlp import CONFIGS, TissuePredictor
from src.classification.model import TissuePredictor
from src.classification.config import CONFIGS

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)


def init_pretrained_classifier(config: dict, dataset:str, path_model: str, device: str):
    """
    Init pretrained classifier for classification task on cancer, cancer type or tissue type.
    ----
    Parameters:
        config (dict): model configuration dictionary
        dataset (str): dataset name used for pretraining
        path_model (str): path of pretrained model weights
        device (str): GPU/CPU device where to load the model
    Returns:
        model (torch Module): pretrained model
    """
    # Instantiate device in config dict
    config['device'] = device
    # Instantiate model
    model = TissuePredictor(config, dataset)
    # Load model
    model.model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))

    return model

def get_activations_for_frechet_dist( real: np.array,
                                        fake: np.array,
                                        model: torch.nn.Module):
    """
    Returns activations for real and fake input data in order to compute Frechet Score.
    ----
    Parameters:
        real_data (torch.tensor): first data vector
        fake_data (torch.tensor): second data vector
        model (torch Module): model used for features vector retrieval
    Returns:
        activations real (np.array), activation fake (np.array): features vector for both data input vectors
    """
    real = torch.from_numpy(real).float().to(model.device)
    fake = torch.from_numpy(fake).float().to(model.device)
    # Eval mode:
    model.model.eval() # no dropout
    with torch.no_grad():
        # Retrieve feature vectors of the last hidden dense layer before RelU
        act_real = model.model.proj2(torch.nn.functional.relu(model.model.proj1(real))).detach().cpu().numpy()
        act_fake = model.model.proj2(torch.nn.functional.relu(model.model.proj1(fake))).detach().cpu().numpy()

    return act_real, act_fake


def calculate_fid(act1: np.array, act2: np.array):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    ----
    Parameters:
        act1 (np.array): first feature vector
        act2 (np.array): second feature vector
    Returns:
        Frechet Distance Score between input feature vectors
    """
    # calculate mean and covariance statistics
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    # calculate sum squared difference between means
    diff = mu1 - mu2
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid


def compute_frechet_distance_score(
        real_data: np.array,
        fake_data: np.array,
        dataset: str = 'tcga',
        device: str = None,
        to_standardize:bool=False):
    """
    ----
    Parameters:
        real_data (np.array): first data vector to use for comparison
        fake_data (np.array): second data vector to use for comparison
        task (str): classification task leading to a different pretrained model
        device (str): device (GPU/CPU) where to load the pretrained model
    Returns:
        Frechet distance score between activatiosn for real data and fake data
    """
    
    # Config
    if dataset=='tcga':
        config = CONFIGS[1]
    elif dataset=='gtex':
        config = CONFIGS[2]

    # Path where pretrained weights are stored    
    # path_model = f'/home/alacan/scripts/classification/landmarks/results/model_{dataset.lower()}.pth'
    # path_model = f'/home/alacan/scripts/gerec_pipeline/src/classification/results/model_{dataset.lower()}.pth'
    path_model = f'/home/alacan/scripts/classification/landmarks/results/model_{dataset.lower()}.pth'

    # Init pretrained model
    model = init_pretrained_classifier(config, dataset, path_model, device)
    
    # To standardize (centering and reduction before MLP)
    if to_standardize:
        scaler = StandardScaler()
        real_data = scaler.fit_transform(real_data)
        fake_data = scaler.transform(fake_data)
    
    # Retrieve feature vectors for both real and synthetic data
    act1, act2 = get_activations_for_frechet_dist(real_data, fake_data, model)

    return calculate_fid(act1, act2)
