"""

Tool box for training.

"""

# Imports
import os
import sys
import random
import numpy as np
import torch
import time as t
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

def load_tcga(test:bool=False):
    # HARDCODED
    #path = "/home/alacan/scripts/diffusion_models/diffusion/diffusion/ddim/sources/datasets/"+dataset+".csv"
    if test:
        path = f"/home/alacan/data_RNAseq_RTCGA/test_df_covariates.csv"
    else:
        path = f"/home/alacan/data_RNAseq_RTCGA/train_df_covariates.csv"
    df_tcga = pd.read_csv(path, ',')
    return df_tcga

def load_gtex(test:bool=False):
    # HARDCODED
    #path = "/home/alacan/scripts/diffusion_models/diffusion/diffusion/ddim/sources/datasets/"+dataset+".csv"
    if test:
        # path = f"/home/alacan/GTEx_data/df_test_gtex_L974.csv"
        path = f"/home/alacan/GTEx_data/df_test_gtex.csv"
    else:
        # path = f"/home/alacan/GTEx_data/df_train_gtex_L974.csv"
        path = f"/home/alacan/GTEx_data/df_train_gtex.csv"
    df = pd.read_csv(path, ',')
    return df

def clean_data(df, keep_cols = ['cancer']):
    col_names_not_dna = [col for col in df.columns if not col.isdigit()]

    for col in keep_cols:
        col_names_not_dna.remove(col)

    df = df.drop(columns = col_names_not_dna)
    # Remove rows with NaN
    df = df[~pd.isnull(df).any(axis=1)]

    return df

def process_tcga_data(test:bool=False, landmark:bool=False, split_landmark:bool=False, to_tensors:bool=True):
    """
    """
    df_tcga = load_tcga(test)
    df_tcga = clean_data(df_tcga, keep_cols = ['age','gender','cancer', 'tissue_type'])

    #age, gender, cancer
    numerical_covs = df_tcga[['age','gender','cancer']]

    #convert gender column to 0 and 1
    numerical_covs.loc[numerical_covs['gender'] == "male", 'gender'] = 0
    numerical_covs.loc[numerical_covs['gender'] == "female", 'gender'] = 1

    numerical_covs = numerical_covs.values
    numerical_covs = numerical_covs.astype(np.float32)

    TISSUES = ['adrenal', 'bladder', 'breast', 'cervical', 'liver', 'colon', 'blood', 'esophagus', 'brain', 'head', 'kidney', 'kidney', 'kidney', 'blood', 'brain', 'liver', 'lung', 'lung', 'lung', 'ovary', 'pancreas', 'kidney', 'prostate','rectum', 'soft-tissues', 'skin', 'stomach', 'stomach', 'testes', 'thyroid', 'thymus', 'uterus', 'uterus', 'eye']

    Tissue_Encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # Init encoder
    Tissue_Encoder.fit(np.unique(TISSUES).reshape(-1,1))

    categorical_covs = df_tcga['tissue_type'].to_numpy()
    #reshape to features vector
    categorical_covs = categorical_covs.reshape((-1,1))
    categorical_covs = Tissue_Encoder.transform(X = categorical_covs)
    print(categorical_covs.shape)

    #tissues types as one hot
    categorical_covs = categorical_covs.astype(int)
    true = df_tcga.drop(columns = ['age','gender','cancer','tissue_type'])
    
    if split_landmark:
        true_l, true_nl = split_landmark_non_landmark(true, dataset='tcga')
        true_l = true_l.values.astype(np.float32)
        true_nl = true_nl.values.astype(np.float32)

        if to_tensors:
            true_l = torch.from_numpy(true_l)
            true_nl = torch.from_numpy(true_nl)
            numerical_covs = torch.from_numpy(numerical_covs)
            categorical_covs = torch.from_numpy(categorical_covs.toarray())

        return true_l, true_nl, numerical_covs, categorical_covs

    else:
        if landmark: # Get only 978 landmark genes
            df_tcga_union_landmark = pd.read_csv('/home/alacan/scripts/gerec_pipeline/tcga_union_978landmark_genes.csv') # Load landmark genes ids
            df_tcga_union_landmark['genes_ids'] = df_tcga_union_landmark['genes_ids'].astype('str')
            true = true[df_tcga_union_landmark['genes_ids']] # keep only the genes ids in landmark genes ids
        
        true = true.values
        true = true.astype(np.float32)

        #convert to torch tensor
        if to_tensors:
            true = torch.from_numpy(true)
            numerical_covs = torch.from_numpy(numerical_covs)
            categorical_covs = torch.from_numpy(categorical_covs.toarray())

        return true, numerical_covs, categorical_covs
    

def process_tcga_data_with_details(test:bool=False, landmark:bool=False, split_landmark:bool=False, to_tensors:bool=True):
    """
    """
    df_tcga = load_tcga(test)
    df_tcga = clean_data(df_tcga, keep_cols = ['age','gender','cancer', 'tissue_type', 'cancer_type'])

    #age, gender, cancer
    numerical_covs = df_tcga[['age','gender','cancer']]

    #convert gender column to 0 and 1
    numerical_covs.loc[numerical_covs['gender'] == "male", 'gender'] = 0
    numerical_covs.loc[numerical_covs['gender'] == "female", 'gender'] = 1

    numerical_covs = numerical_covs.values
    numerical_covs = numerical_covs.astype(np.float32)

    TISSUES = ['adrenal', 'bladder', 'breast', 'cervical', 'liver', 'colon', 'blood', 'esophagus', 'brain', 'head', 'kidney', 'kidney', 'kidney', 'blood', 'brain', 'liver', 'lung', 'lung', 'lung', 'ovary', 'pancreas', 'kidney', 'prostate','rectum', 'soft-tissues', 'skin', 'stomach', 'stomach', 'testes', 'thyroid', 'thymus', 'uterus', 'uterus', 'eye']

    Tissue_Encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # Init encoder
    Tissue_Encoder.fit(np.unique(TISSUES).reshape(-1,1))

    categorical_covs = df_tcga['tissue_type'].to_numpy()
    #reshape to features vector
    categorical_covs = categorical_covs.reshape((-1,1))
    categorical_covs = Tissue_Encoder.transform(X = categorical_covs)
    print(categorical_covs.shape)

    # cancer types
    ct_details = df_tcga['cancer_type'].to_numpy()

    #tissues types as one hot
    categorical_covs = categorical_covs.astype(int)
    true = df_tcga.drop(columns = ['age','gender','cancer','tissue_type', 'cancer_type'])
    
    if split_landmark:
        true_l, true_nl = split_landmark_non_landmark(true, dataset='tcga')
        true_l = true_l.values.astype(np.float32)
        true_nl = true_nl.values.astype(np.float32)

        if to_tensors:
            true_l = torch.from_numpy(true_l)
            true_nl = torch.from_numpy(true_nl)
            numerical_covs = torch.from_numpy(numerical_covs)
            categorical_covs = torch.from_numpy(categorical_covs.toarray())

        return true_l, true_nl, numerical_covs, categorical_covs, ct_details

    else:
        if landmark: # Get only 978 landmark genes
            df_tcga_union_landmark = pd.read_csv('/home/alacan/scripts/gerec_pipeline/tcga_union_978landmark_genes.csv') # Load landmark genes ids
            df_tcga_union_landmark['genes_ids'] = df_tcga_union_landmark['genes_ids'].astype('str')
            true = true[df_tcga_union_landmark['genes_ids']] # keep only the genes ids in landmark genes ids
        
        true = true.values
        true = true.astype(np.float32)

        #convert to torch tensor
        if to_tensors:
            true = torch.from_numpy(true)
            numerical_covs = torch.from_numpy(numerical_covs)
            categorical_covs = torch.from_numpy(categorical_covs.toarray())

        return true, numerical_covs, categorical_covs, ct_details


def split_landmark_non_landmark(df=None, dataset='tcga'):
    """
    """
    if dataset=='tcga':
        df_tcga_union_landmark = pd.read_csv('/home/alacan/scripts/gerec_pipeline/tcga_union_978landmark_genes.csv') # Load landmark genes ids
        list_tcga_union_landmark = df_tcga_union_landmark['genes_ids'].values
        df_tcga_union_landmark['genes_ids'] = df_tcga_union_landmark['genes_ids'].astype('str')
        # Landmarks
        df_landmark = df[df_tcga_union_landmark['genes_ids']] 
        # Non landmarks
        non_landmark = [str(i) for i in list(df.columns) if int(i) not in list_tcga_union_landmark]
        df_non_landmark = df[non_landmark]

        # # Save
        # pd.DataFrame(data=df_landmark.columns, columns=['gene_id']).to_csv(f"/home/alacan/data_RNAseq_RTCGA/landmark_genes_ids.csv", header=True, index=False)
        # pd.DataFrame(data=df_non_landmark.columns, columns=['gene_id']).to_csv(f"/home/alacan/data_RNAseq_RTCGA/non_landmark_genes_ids.csv", header=True, index=False)

    elif dataset=='gtex': 
        df_descript = pd.read_csv('/home/alacan/GTEx_data/gtex_description.csv', sep=',')
        # filter
        # df_descript['entrez_id'] = df_descript['entrez_id'].astype('str')
        df_landmark = df[df_descript[df_descript.Type=='landmark']['Description'].values] 
        df_non_landmark = df[df_descript[df_descript.Type=='target']['Description'].values] 
    return df_landmark, df_non_landmark


def process_gtex_data(test:bool=False, landmark:bool=False, split_landmark:bool=False, to_tensors:bool=True):
    df_gtex = load_gtex(test)
    # df_gtex = clean_data(df_gtex, keep_cols = ['age','gender', 'tissue_type'])
    # Remove rows with NaN
    df_gtex = df_gtex[~pd.isnull(df_gtex).any(axis=1)]

    #age, gender
    numerical_covs = df_gtex[['age','gender']]

    #convert gender column to 0 and 1
    numerical_covs.loc[numerical_covs['gender'] == "male", 'gender'] = 0
    numerical_covs.loc[numerical_covs['gender'] == "female", 'gender'] = 1

    numerical_covs = numerical_covs.values
    numerical_covs = numerical_covs.astype(np.float32)

    TISSUES = ['Adipose Tissue', 'Adrenal Gland', 'Blood', 'Blood Vessel',
       'Brain', 'Breast', 'Colon', 'Esophagus', 'Heart', 'Liver', 'Lung',
       'Muscle', 'Nerve', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate',
       'Salivary Gland', 'Skin', 'Small Intestine', 'Spleen', 'Stomach',
       'Testis', 'Thyroid', 'Uterus', 'Vagina']

    Tissue_Encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # Init encoder
    Tissue_Encoder.fit(np.unique(TISSUES).reshape(-1,1))

    categorical_covs = df_gtex['tissue_type'].to_numpy()
    #reshape to features vector
    categorical_covs = categorical_covs.reshape((-1,1))
    categorical_covs = Tissue_Encoder.transform(X = categorical_covs)

    # tissues types as one hot
    categorical_covs = categorical_covs.astype(int)

    df_gtex = df_gtex.drop(columns = ['age','gender','tissue_type'])
    
    if split_landmark:
        true_l, true_nl = split_landmark_non_landmark(df_gtex, dataset='gtex')
        true_l = true_l.values.astype(np.float32)
        true_nl = true_nl.values.astype(np.float32)

        if to_tensors:
            true_l = torch.from_numpy(true_l)
            true_nl = torch.from_numpy(true_nl)
            numerical_covs = torch.from_numpy(numerical_covs)
            categorical_covs = torch.from_numpy(categorical_covs.to_array())

        return true_l, true_nl, numerical_covs, categorical_covs

    else:
        if landmark: # Get only 974 landmark genes
            df_landmark = pd.read_csv('/home/alacan/scripts/gerec_pipeline/L974_gtex_description.csv') # Load landmark genes ids
            df_gtex = df_gtex[df_landmark.iloc[0][1:].values] # keep only the genes ids in landmark entrez ids
        
        df_gtex = df_gtex.values
        df_gtex = df_gtex.astype(np.float32)

        #convert to torch tensor
        if to_tensors:
            df_gtex = torch.from_numpy(df_gtex)
            numerical_covs = torch.from_numpy(numerical_covs)
            categorical_covs = torch.from_numpy(categorical_covs.to_array())

        return df_gtex, numerical_covs, categorical_covs
    

def process_gtex_data_with_details(test:bool=False, landmark:bool=False, split_landmark:bool=False, to_tensors:bool=True):
    df_gtex = load_gtex(test)
    # df_gtex = clean_data(df_gtex, keep_cols = ['age','gender', 'tissue_type'])
    # Remove rows with NaN
    df_gtex = df_gtex[~pd.isnull(df_gtex).any(axis=1)]

    #age, gender
    numerical_covs = df_gtex[['age','gender']]

    #convert gender column to 0 and 1
    numerical_covs.loc[numerical_covs['gender'] == "male", 'gender'] = 0
    numerical_covs.loc[numerical_covs['gender'] == "female", 'gender'] = 1

    numerical_covs = numerical_covs.values
    numerical_covs = numerical_covs.astype(np.float32)

    TISSUES = ['Adipose Tissue', 'Adrenal Gland', 'Blood', 'Blood Vessel',
       'Brain', 'Breast', 'Colon', 'Esophagus', 'Heart', 'Liver', 'Lung',
       'Muscle', 'Nerve', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate',
       'Salivary Gland', 'Skin', 'Small Intestine', 'Spleen', 'Stomach',
       'Testis', 'Thyroid', 'Uterus', 'Vagina']

    Tissue_Encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # Init encoder
    Tissue_Encoder.fit(np.unique(TISSUES).reshape(-1,1))

    categorical_covs = df_gtex['tissue_type'].to_numpy()
    #reshape to features vector
    categorical_covs = categorical_covs.reshape((-1,1))
    categorical_covs = Tissue_Encoder.transform(X = categorical_covs)

    # tissues types as one hot
    categorical_covs = categorical_covs.astype(int)

    # tissue_type_details
    tt_details = df_gtex['tissue_type_details'].to_numpy()

    df_gtex = df_gtex.drop(columns = ['age','gender','tissue_type', 'tissue_type_details'])
    
    if split_landmark:
        true_l, true_nl = split_landmark_non_landmark(df_gtex, dataset='gtex')
        true_l = true_l.values.astype(np.float32)
        true_nl = true_nl.values.astype(np.float32)

        if to_tensors:
            true_l = torch.from_numpy(true_l)
            true_nl = torch.from_numpy(true_nl)
            numerical_covs = torch.from_numpy(numerical_covs)
            categorical_covs = torch.from_numpy(categorical_covs.to_array())

        return true_l, true_nl, numerical_covs, categorical_covs, tt_details

    else:
        if landmark: # Get only 974 landmark genes
            df_landmark = pd.read_csv('/home/alacan/scripts/gerec_pipeline/L974_gtex_description.csv') # Load landmark genes ids
            df_gtex = df_gtex[df_landmark.iloc[0][1:].values] # keep only the genes ids in landmark entrez ids
        
        df_gtex = df_gtex.values
        df_gtex = df_gtex.astype(np.float32)

        #convert to torch tensor
        if to_tensors:
            df_gtex = torch.from_numpy(df_gtex)
            numerical_covs = torch.from_numpy(numerical_covs)
            categorical_covs = torch.from_numpy(categorical_covs.to_array())

        return df_gtex, numerical_covs, categorical_covs, tt_details


def get_tcga_datasets(scaler_type:str="standard", landmark=True, split_landmark=False, to_tensors=True):
    # Load train data
    X_train, numerical_covs, y_train = process_tcga_data(test=False, landmark=landmark, split_landmark=split_landmark,to_tensors=to_tensors)
    # Load test data
    X_test, numerical_covs_test, y_test = process_tcga_data(test=True, landmark=landmark, split_landmark=split_landmark,to_tensors=to_tensors)
    # Split
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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
    # X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Turn data into tensors
    X_train = torch.tensor(X_train).type(torch.float)
    # X_val = torch.tensor(X_val).type(torch.float)
    X_test = torch.tensor(X_test).type(torch.float)

    train = data_utils.TensorDataset(X_train, y_train) 
    # val = data_utils.TensorDataset(X_val, y_val)
    test = data_utils.TensorDataset(X_test, y_test) 

    return train, test

def get_datasets_split_landmarks_for_search(dataset:str, landmark:bool=False, split_landmark:bool=True, with_tissues:bool=False):
    """
    """
    # Load train data
    if dataset=='tcga':
        process_func = process_tcga_data
    elif dataset =='gtex':
        process_func = process_gtex_data
    X, y, numerical_covs, tissues = process_func(test=False, landmark=landmark, split_landmark=split_landmark, to_tensors=False)
    # Load test data
    X_test, y_test, numerical_covs_test, tissues_test = process_func(test=True, landmark=landmark, split_landmark=split_landmark, to_tensors=False)

    if with_tissues:
        return X,y,X_test,y_test,tissues, tissues_test
    else:
        return X, y, X_test, y_test

def get_datasets_for_search(dataset:str):
    """
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

    train = data_utils.TensorDataset(X_train, y_train) 
    val = data_utils.TensorDataset(X_val, y_val)
    test = data_utils.TensorDataset(X_test, y_test) 

    return train, val, test

def split_and_scale_split_landmark(X, y, X_test, y_test, scaler_type:str="standard", tissues=None):
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=tissues.argmax(1))

    # Scale the data
    if scaler_type == "standard":
        scaler = StandardScaler()
        scaler_nl = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
        scaler_nl = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
        scaler_nl = RobustScaler()
    elif scaler_type == "maxabs":
        scaler = MaxAbsScaler()
        scaler_nl = MaxAbsScaler()
    else:
        raise Exception("Unknown scaler type")

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train = scaler_nl.fit_transform(y_train)
    y_val = scaler_nl.transform(y_val)
    y_test = scaler_nl.transform(y_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def split_and_scale_datasets_split_landmark(X, y, X_test, y_test, scaler_type:str="standard", tissues=None):
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=tissues.argmax(1))

    # Scale the data
    if scaler_type == "standard":
        scaler = StandardScaler()
        scaler_nl = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
        scaler_nl = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
        scaler_nl = RobustScaler()
    elif scaler_type == "maxabs":
        scaler = MaxAbsScaler()
        scaler_nl = MaxAbsScaler()
    else:
        raise Exception("Unknown scaler type")

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train = scaler_nl.fit_transform(y_train)
    y_val = scaler_nl.transform(y_val)
    y_test = scaler_nl.transform(y_test)

    # Turn data into tensors
    X_train = torch.tensor(X_train).type(torch.float)
    X_val = torch.tensor(X_val).type(torch.float)
    X_test = torch.tensor(X_test).type(torch.float)

    y_train = torch.tensor(y_train).type(torch.float)
    y_val = torch.tensor(y_val).type(torch.float)
    y_test = torch.tensor(y_test).type(torch.float)

    train = data_utils.TensorDataset(X_train, y_train) 
    val = data_utils.TensorDataset(X_val, y_val)
    test = data_utils.TensorDataset(X_test, y_test) 

    return train, val, test

def build_loaders(train, val, test=None, config:dict=None):
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
                                        persistent_workers=True)
        
        return train_loader, val_loader, test_loader 
    else:
        return train_loader, val_loader 

def get_gtex_datasets(scaler_type:str="standard"):
    # Load train data
    X_train, numerical_covs, y_train = process_gtex_data(test=False, landmark=True)
    # Load test data
    X_test, numerical_covs_test, y_test = process_gtex_data(test=True, landmark=True)
    # Split
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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
    # X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # Turn data into tensors
    X_train = torch.tensor(X_train).type(torch.float)
    # X_val = torch.tensor(X_val).type(torch.float)
    X_test = torch.tensor(X_test).type(torch.float)

    train = data_utils.TensorDataset(X_train, y_train) 
    # val = data_utils.TensorDataset(X_valid, y_val)
    test = data_utils.TensorDataset(X_test, y_test) 

    return train, test


# Training callbacks
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience:int=7, verbose:int=0, delta:int=0, path:str='ckpt.pt', trace_func=print):
        """
        Parameters:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, metric, model):
        """ Main function that saves weights at checkpoint if metric has improved. If the metric has not improved after a number of iterations (patience), early stop is forced.
        ----
        Parameters:
            metric (int): metric used to assess performance in training
            model (pytorch model): model being trained
        """
        score = metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.verbose>0:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        """Saves model when metric has improved (in this case it decreases).
        ----
        Parameters:
            metric (int): metric used to assess performance in training
            model (pytorch model): model being trained"""

        if self.verbose>0:
            self.trace_func(f'Validation metric decreased ({self.metric_min:.6f} --> {metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.metric_min = metric
