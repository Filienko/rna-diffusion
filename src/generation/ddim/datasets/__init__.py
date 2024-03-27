import os
import torch
from src.generation.ddim.datasets.utils import get_tcga_datasets, get_gtex_datasets
import numpy as np

def get_dataset(config):
    #si notre dataset a TCGA dans son nom
    if "TCGA" in config.data.dataset or "tcga" in config.data.dataset:
        dataset, test_dataset = get_tcga_datasets(config.data.scaler_type)

    elif "GTEX" in config.data.dataset or "gtex" in config.data.dataset:
        dataset, test_dataset = get_gtex_datasets(config.data.scaler_type)

    return dataset, test_dataset