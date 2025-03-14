"""

Tool box for training.

"""

# Imports
import os
import sys
import numpy as np
import torch
import time as t
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils

def get_dataset(config):
    """
    Get dataset from config.
    Parameters:
        config (dict): dict of the model configuration
    Returns:
        tuple of train dataframe and test dataframe (csv)
    """
    if "TCGA" in config.data.dataset or "tcga" in config.data.dataset:
        dataset, test_dataset = get_tcga_datasets(config)

    elif "GTEX" in config.data.dataset or "gtex" in config.data.dataset:
        dataset, test_dataset = get_gtex_datasets(config)

    return dataset, test_dataset


def load_tcga(test:bool=False):
    # HARDCODED
    #path = "/home/alacan/scripts/diffusion_models/diffusion/diffusion/ddim/sources/datasets/"+dataset+".csv"
    if test:
        path = f"/home/daniilf/rna-diffusion/data/processed_tcga_data/test_df_covariates.csv"
    else:
        path = f"/home/daniilf/rna-diffusion/data/processed_tcga_data/train_df_covariates.csv"
    df_tcga = pd.read_csv(path)
    return df_tcga

def load_gtex(test:bool=False):
    """
    Parameters:
        test (bool): whether to load the test dataset
    Returns:
        GTEx dataframe (csv)
    """
    # HARDCODED
    if test:
        path = f"/home/alacan/GTEx_data/df_test_gtex_L974.csv"
    else:
        path = f"/home/alacan/GTEx_data/df_train_gtex_L974.csv"
    df = pd.read_csv(path, ',')
    return df

def clean_data(df, keep_cols = ['cancer']):
    """
    Selection of columns and removing of NaNs.

    Parameters:
        df (csv): dataframe to clean
        keep_cols (list): list of columns (other than genes) to keep^n the output dataset
    Returns:
        df (csv): clean dataframe
    """
    col_names_not_dna = [col for col in df.columns if not col.isdigit()]

    for col in keep_cols:
        col_names_not_dna.remove(col)

    df = df.drop(columns = col_names_not_dna)
    # Remove rows with NaN
    df = df[~pd.isnull(df).any(axis=1)]

    return df

def process_tcga_data(test:bool=False, landmark:bool=False):
    """
    Parameters:
        test (bool): whether to process the test set
        landmark (bool): whether to keep only landmark genes
    Returns:
        tuple of processed genes (tensor), processed numerical covariates (tensor), categorical covariates (tensor)
    """
    df_tcga = load_tcga(test)
    df_tcga = clean_data(df_tcga, keep_cols = ['age','gender','cancer', 'tissue_type'])

    # age, gender, cancer
    numerical_covs = df_tcga[['age','gender','cancer']]

    # convert gender column to 0 and 1
    numerical_covs.loc[numerical_covs['gender'] == "male", 'gender'] = 0
    numerical_covs.loc[numerical_covs['gender'] == "female", 'gender'] = 1

    numerical_covs = numerical_covs.values
    numerical_covs = numerical_covs.astype(np.float32)

    # One-hot tissue types
    TISSUES = ['adrenal', 'bladder', 'breast', 'cervical', 'liver', 'colon', 'blood', 'esophagus', 'brain', 'head', 'kidney', 'kidney', 'kidney', 'blood', 'brain', 'liver', 'lung', 'lung', 'lung', 'ovary', 'pancreas', 'kidney', 'prostate','rectum', 'soft-tissues', 'skin', 'stomach', 'stomach', 'testes', 'thyroid', 'thymus', 'uterus', 'uterus', 'eye']

    Tissue_Encoder = OneHotEncoder(handle_unknown='ignore') # Init encoder
    Tissue_Encoder.fit(np.unique(TISSUES).reshape(-1,1))

    categorical_covs = df_tcga['tissue_type'].to_numpy()
    # reshape to features vector
    categorical_covs = categorical_covs.reshape((-1,1))
    categorical_covs = Tissue_Encoder.transform(X = categorical_covs)
    print(categorical_covs.shape)
    categorical_covs = categorical_covs.astype(int)

    true = df_tcga.drop(columns = ['age','gender','cancer','tissue_type'])
    
    if landmark: # Get only 978 landmark genes
        df_tcga_union_landmark = pd.read_csv('/home/alacan/scripts/diffusion_models/diffusion/ddim/datasets/tcga_union_978landmark_genes.csv') # Load landmark genes ids
        df_tcga_union_landmark['genes_ids'] = df_tcga_union_landmark['genes_ids'].astype('str')
        true = true[df_tcga_union_landmark['genes_ids']] # keep only the genes ids in landmark genes ids
    
    true = true.values
    true = true.astype(np.float32)

    # convert to torch tensor
    true = torch.from_numpy(true)
    numerical_covs = torch.from_numpy(numerical_covs)
    categorical_covs = torch.from_numpy(categorical_covs.toarray())

    return true, numerical_covs, categorical_covs

def process_gtex_data(test:bool=False, landmark:bool=False):
    """
    Parameters:
        test (bool): whether to process the test set
        landmark (bool): whether to keep only landmark genes
    Returns:
        tuple of processed genes (tensor), processed numerical covariates (tensor), categorical covariates (tensor)
    """
    df_gtex = load_gtex(test)
    df_gtex = clean_data(df_gtex, keep_cols = ['age','gender', 'tissue_type'])

    # age, gender
    numerical_covs = df_gtex[['age','gender']]

    #convert gender column to 0 and 1
    numerical_covs.loc[numerical_covs['gender'] == "male", 'gender'] = 0
    numerical_covs.loc[numerical_covs['gender'] == "female", 'gender'] = 1

    numerical_covs = numerical_covs.values
    numerical_covs = numerical_covs.astype(np.float32)

    # tissue types as one hot
    TISSUES = ['Adipose Tissue', 'Adrenal Gland', 'Blood', 'Blood Vessel',
       'Brain', 'Breast', 'Colon', 'Esophagus', 'Heart', 'Liver', 'Lung',
       'Muscle', 'Nerve', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate',
       'Salivary Gland', 'Skin', 'Small Intestine', 'Spleen', 'Stomach',
       'Testis', 'Thyroid', 'Uterus', 'Vagina']

    Tissue_Encoder = OneHotEncoder(handle_unknown='ignore') # Init encoder
    Tissue_Encoder.fit(np.unique(TISSUES).reshape(-1,1))

    categorical_covs = df_gtex['tissue_type'].to_numpy()
    # reshape to features vector
    categorical_covs = categorical_covs.reshape((-1,1))
    categorical_covs = Tissue_Encoder.transform(X = categorical_covs)
    categorical_covs = categorical_covs.astype(int)

    df_gtex = df_gtex.drop(columns = ['age','gender','tissue_type'])
    df_gtex = df_gtex.values
    df_gtex = df_gtex.astype(np.float32)

    # convert to torch tensor
    df_gtex = torch.from_numpy(df_gtex)
    numerical_covs = torch.from_numpy(numerical_covs)
    categorical_covs = torch.from_numpy(categorical_covs.toarray())

    return df_gtex, numerical_covs, categorical_covs

def get_tcga_datasets(scaler_type:str="standard"):
    """
    Global function to obtain preprocessed TCGA dataset.
    ----
    Parameters:
        scaler_type (str): type of data scaling (e.g., minmax, standard, maxabs)
    Returns:
        tuple of train and test sets as pytorch datasets
    """
    # Load train data
    X_train, numerical_covs, y_train = process_tcga_data(test=False, landmark=True)
    # Load test data
    X_test, numerical_covs_test, y_test = process_tcga_data(test=True, landmark=True)
    
    # Scale the data
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "maxabs":
        scaler = MaxAbsScaler()
    else:
        raise Exception("Unknown scaler type")

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Turn data into tensors
    X_train = torch.tensor(X_train).type(torch.float)
    X_test = torch.tensor(X_test).type(torch.float)

    train = data_utils.TensorDataset(X_train, y_train) 
    test = data_utils.TensorDataset(X_test, y_test) 

    return train, test

def get_datasets_for_search(dataset:str):
    """
    Global function to obtain the preprocessed dataset of landmark genes (unscaled) for hyperparameters search.
    ----
    Parameters:
        dataset (str): dataset name ('tcga' or 'gtex')
    Returns:
        tuple of (X_train, y_train, X_test, y_test) as tensors
    """
    # Load train data
    if dataset=='tcga':
        process_func = process_tcga_data
    elif dataset =='gtex':
        process_func = process_gtex_data
    X, numerical_covs, y = process_func(test=False, landmark=True)
    # Load test data
    X_test, numerical_covs_test, y_test = process_func(test=True, landmark=True)

    return X, y, X_test, y_test

def split_and_scale_datasets(X, y, X_test, y_test, scaler_type:str="standard"):
    """
    Function to split and scale the training set.
    ----
    Parameters:
        X (array): Train data
        y (array): Train labels
        X_test (array): Test data
        y_test (array): Test labels
        scaler_type (str): type of scaling (e.g., minmax, standard, maxabs)
    Returns:
        tuple of train, val, test pytorch datasets
    """
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "maxabs":
        scaler = MaxAbsScaler()
    else:
        raise Exception("Unknown scaler type")

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Turn data into tensors
    X_train = torch.tensor(X_train).type(torch.float)
    X_val = torch.tensor(X_val).type(torch.float)
    X_test = torch.tensor(X_test).type(torch.float)

    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)
    y_test = torch.tensor(y_test)

    train = data_utils.TensorDataset(X_train, y_train) 
    val = data_utils.TensorDataset(X_val, y_val)
    test = data_utils.TensorDataset(X_test, y_test) 

    return train, val, test

def build_loaders(train, val, test=None, config:dict=None):
    """
    Buil pytorch dataloaders.
    ----
    Parameters:
        train (tensor): train tensor data
        val (tensor): validation tensor data
        test (tensor): test tensor data (default None)
        config (dict): dictionary of the model configuration (default None)
    Returns:
        tuple of train, val (and test if specified) pytorch dataloaders
    """
    train_loader = data_utils.DataLoader(
                                        train,
                                        batch_size=config['batch_size'],
                                        shuffle=True,
                                        num_workers=config['num_workers'],
                                        pin_memory=True,
                                        prefetch_factor=2,
                                        persistent_workers=False)
        
    val_loader = data_utils.DataLoader(
                                    val,
                                    batch_size=config['batch_size'],
                                    shuffle=False,
                                    num_workers=config['num_workers'],
                                    pin_memory=True,
                                    prefetch_factor=2,
                                    persistent_workers=False)
    if test is not None:
        test_loader = data_utils.DataLoader(
                                        test,
                                        batch_size=config['batch_size'],
                                        shuffle=False,
                                        num_workers=config['num_workers'],
                                        pin_memory=True,
                                        prefetch_factor=2,
                                        persistent_workers=False)
        
        return train_loader, val_loader, test_loader 
    else:
        return train_loader, val_loader 

def get_gtex_datasets(scaler_type:str="standard"):
    """
    Global function to obtain preprocessed GTEx dataset.
    ----
    Parameters:
        scaler_type (str): type of data scaling (e.g., minmax, standard, maxabs)
    Returns:
        tuple of train and test sets as pytorch datasets
    """
    # Load train data
    X_train, numerical_covs, y_train = process_gtex_data(test=False, landmark=True)
    # Load test data
    X_test, numerical_covs_test, y_test = process_gtex_data(test=True, landmark=True)

    # Scale the data
    print("scaler_type", scaler_type)
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "maxabs":
        scaler = MaxAbsScaler()
    else:
        raise Exception("Unknown scaler type")

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Turn data into tensors
    X_train = torch.tensor(X_train).type(torch.float)
    X_test = torch.tensor(X_test).type(torch.float)

    train = data_utils.TensorDataset(X_train, y_train) 
    test = data_utils.TensorDataset(X_test, y_test) 

    return train, test

