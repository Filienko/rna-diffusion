"""
The code in this repository was adapted from the original DDIM paper code: https://github.com/ermongroup/ddim
"""

import os
import torch
from src.generation.ddim.utils import get_tcga_datasets, get_gtex_datasets
import numpy as np

def get_dataset(config):
    if "TCGA" in config.data.dataset or "tcga" in config.data.dataset:
        dataset, test_dataset = get_tcga_datasets(config)

    elif "GTEX" in config.data.dataset or "gtex" in config.data.dataset:
        dataset, test_dataset = get_gtex_datasets(config)

    return dataset, test_dataset