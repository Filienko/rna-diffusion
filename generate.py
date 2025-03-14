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

# Load your trained GAN model
model = WGAN(CONFIG)
model.load_generator(path='path/to/your/saved/generator.pt', location=torch.device(CUDA_DEVICE))

# Create a DataLoader with your input data
input_loader = data_utils.DataLoader(
    your_input_data,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers']
)

# Generate data similar to your input
_, generated_data = model.generate(input_loader, return_labels=False)

# Save the generated data
generated_data_np = generated_data.cpu().numpy()
pd.DataFrame(generated_data_np).to_csv('generated_data.csv', index=False, header=False)
