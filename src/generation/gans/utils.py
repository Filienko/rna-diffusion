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

# Metrics
# sys.path.append(os.path.abspath("../metrics"))
from src.metrics.precision_recall import compute_prdc
from src.metrics.aats import compute_AAts
from src.metrics.correlation_score import gamma_coeff_score
from src.metrics.frechet import compute_frechet_distance_score

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
    # HARDCODED
    #path = "/home/alacan/scripts/diffusion_models/diffusion/diffusion/ddim/sources/datasets/"+dataset+".csv"
    if test:
        path = f"/home/daniilf/rna-diffusion/data/df_test_gtex_L974.csv"
    else:
        path = f"/home/daniilf/rna-diffusion/data/df_train_gtex_L974.csv"
    df = pd.read_csv(path)
    return df

def clean_data(df, keep_cols = ['cancer_type']):
    # Prepare data for basic multiclass prediction
    col_names_not_dna = [col for col in df.columns[1:] if not col.isdigit()]

    for col in keep_cols[1:]:
        print(col)
        print(col_names_not_dna)
        col_names_not_dna.remove(col)

    df = df.drop(columns = col_names_not_dna)
    # Remove rows with NaN
    df = df[~pd.isnull(df).any(axis=1)]

    return df

def process_tcga_data(test:bool=False, landmark:bool=False, numerical_cols=['label'], 
                     categorical_cols=['cancer_type', 'subtype',], tissue_mappings=None):
    """
    Process TCGA data with flexible categorical columns handling
    
    Parameters:
    -----------
    test : bool
        Whether to use test data
    landmark : bool
        Whether to use landmark genes only
    numerical_cols : list
        List of numerical columns to use as covariates
    categorical_cols : list or None
        List of categorical columns to use for one-hot encoding. If None, no categorical features will be used.
    tissue_mappings : dict or None
        Pre-defined mappings for tissue types. If None, will use default TISSUES list.
    
    Returns:
    --------
    tuple
        (gene expression data, numerical covariates, categorical covariates)
    """
    # TODO: convert genes, if want to use landmark
    landmark = False
    df_tcga = load_tcga(test)
    keep_columns = df_tcga.columns
    # df_tcga = clean_data(df_tcga, keep_cols=keep_columns)
    
    # Handle numerical covariates
    if numerical_cols and all(col in df_tcga.columns for col in numerical_cols):
        numerical_covs = df_tcga[numerical_cols]
        numerical_covs = numerical_covs.values
        numerical_covs = numerical_covs.astype(np.float32)
    else:
        numerical_covs = np.zeros((len(df_tcga), 0), dtype=np.float32)
    
    # Handle categorical covariates
    if categorical_cols and all(col in df_tcga.columns for col in categorical_cols):
        # Default tissue mappings if not provided
        if 'tissue_type' in categorical_cols and tissue_mappings is None:
            TISSUES = ['adrenal', 'bladder', 'breast', 'cervical', 'liver', 'colon', 'blood', 
                      'esophagus', 'brain', 'head', 'kidney', 'kidney', 'kidney', 'blood', 
                      'brain', 'liver', 'lung', 'lung', 'lung', 'ovary', 'pancreas', 'kidney', 
                      'prostate', 'rectum', 'soft-tissues', 'skin', 'stomach', 'stomach', 
                      'testes', 'thyroid', 'thymus', 'uterus', 'uterus', 'eye']
        
        # Create and process categorical features
        categorical_arrays = []
        for col in categorical_cols:
            # Get the column data
            col_data = df_tcga[col].to_numpy().reshape(-1, 1)
            
            # Create encoder
            encoder = OneHotEncoder(handle_unknown='ignore')
            
            # Fit encoder with predefined categories if available
            if col == 'tissue_type' and tissue_mappings is not None:
                unique_values = np.array(list(tissue_mappings.keys())).reshape(-1, 1)
                encoder.fit(unique_values)
            elif col == 'tissue_type':
                encoder.fit(np.unique(TISSUES).reshape(-1, 1))
            else:
                # Fit on the actual data for other columns
                encoder.fit(col_data)
            
            # Transform and add to the list
            encoded = encoder.transform(col_data).toarray()
            categorical_arrays.append(encoded)
        
        # Combine all categorical features if any exist
        if categorical_arrays:
            categorical_covs = np.hstack(categorical_arrays)
            categorical_covs = categorical_covs.astype(int)
        else:
            categorical_covs = np.zeros((len(df_tcga), 0), dtype=int)
    else:
        # No categorical columns available/selected
        categorical_covs = np.zeros((len(df_tcga), 0), dtype=int)
    
    # Get expression data
    columns_to_drop = []
    if numerical_cols:
        columns_to_drop.extend([col for col in numerical_cols if col in df_tcga.columns])
    if categorical_cols:
        columns_to_drop.extend([col for col in categorical_cols if col in df_tcga.columns])
    
    true = df_tcga.drop(columns=columns_to_drop) if columns_to_drop else df_tcga.copy()
    
    if landmark:  # Get only 978 landmark genes
        df_tcga_union_landmark = pd.read_csv('/home/daniilf/rna-diffusion/data/tcga_union_978landmark_genes.csv')  # Load landmark genes ids
        df_tcga_union_landmark['genes_ids'] = df_tcga_union_landmark['genes_ids'].astype('str')
        true = true[df_tcga_union_landmark['genes_ids']]  # keep only the genes ids in landmark genes ids
    
    true = true.values
    from sklearn.preprocessing import LabelEncoder
    for col_idx in range(true.shape[1]):
        col_data = true[:, col_idx]
        if not np.issubdtype(col_data.dtype, np.number):
            le = LabelEncoder()
            true[:, col_idx] = le.fit_transform(col_data)
    
    true = true.astype(np.float32)
    
    # Convert to torch tensor
    true = torch.from_numpy(true)
    numerical_covs = torch.from_numpy(numerical_covs)
    categorical_covs = torch.from_numpy(categorical_covs)
    
    return true, numerical_covs, categorical_covs

def process_gtex_data(test:bool=False, landmark:bool=False):
    # print("before", test)
    df_gtex = load_gtex(test)
    # print(f"before{df_gtex.shape}; loaded", df_gtex)
    # keep_columns = ['age','gender','tissue_type','DFFB','ICMT', 'tissue_type_details']
    df_gtex = df_gtex.drop(['tissue_type_details', 'index'], axis=1)
    keep_columns = df_gtex.columns
    # all_columns = df_gtex.columns.tolist()  # Convert to a list
    # print(all_columns)  # Prints the full list without truncation
    # print("OK  keep_columns",  keep_columns)
    df_gtex = clean_data(df_gtex, keep_cols = keep_columns)

    # print(f"before; {df_gtex.shape} cleaned", df_gtex)
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

    Tissue_Encoder = OneHotEncoder(handle_unknown='ignore') # Init encoder
    Tissue_Encoder.fit(np.unique(TISSUES).reshape(-1,1))

    categorical_covs = df_gtex['tissue_type'].to_numpy()
    #reshape to features vector
    categorical_covs = categorical_covs.reshape((-1,1))
    categorical_covs = Tissue_Encoder.transform(X = categorical_covs)

    # tissues types as one hot
    categorical_covs = categorical_covs.astype(int)

    df_gtex = df_gtex.drop(columns = ['age','gender','tissue_type'])
    df_gtex = df_gtex.values
    df_gtex = df_gtex.astype(np.float32)
    # print(f"after {df_gtex.shape}; cleaned", df_gtex)
    # convert to torch tensor
    df_gtex = torch.from_numpy(df_gtex)
    numerical_covs = torch.from_numpy(numerical_covs)
    categorical_covs = torch.from_numpy(categorical_covs.toarray())
    # print("after; clean", df_gtex)
    # print(f"after; {numerical_covs.shape} numerical_covs clean",  numerical_covs)
    # print("after; categorical_cov clean", categorical_covs)
    return df_gtex, numerical_covs, categorical_covs

def process_cambda_data(test:bool=False, landmark:bool=False, numerical_cols=['label'], 
                        categorical_cols=['cancer_type', 'subtype']):
    """
    Process CAMBDA data with flexible categorical columns handling
    
    Parameters:
    -----------
    test : bool
        Whether to use test data
    landmark : bool
        Whether to use landmark genes only
    numerical_cols : list
        List of numerical columns to use as covariates
    categorical_cols : list
        List of categorical columns to use for one-hot encoding
    
    Returns:
    --------
    tuple
        (gene expression data, numerical covariates, categorical covariates)
    """
    df_gtex = load_tcga(test)
    print(df_gtex)
    keep_columns = df_gtex.columns
    # df_gtex = clean_data(df_gtex, keep_cols=keep_columns)
    
    # Process numerical covariates
    if numerical_cols and all(col in df_gtex.columns for col in numerical_cols):
        numerical_covs = df_gtex[numerical_cols]
        
        # Label encode any non-numeric columns in numerical_covs
        from sklearn.preprocessing import LabelEncoder
        for col in numerical_covs.columns:
            if numerical_covs[col].dtype == 'object' or numerical_covs[col].dtype.name == 'category':
                le = LabelEncoder()
                numerical_covs[col] = le.fit_transform(numerical_covs[col])
                
        numerical_covs = numerical_covs.values
        numerical_covs = numerical_covs.astype(np.float32)
    else:
        numerical_covs = np.zeros((len(df_gtex), 0), dtype=np.float32)
    
    # Process categorical covariates
    if categorical_cols and all(col in df_gtex.columns for col in categorical_cols):
        from sklearn.preprocessing import OneHotEncoder
        
        # Process each categorical column
        categorical_arrays = []
        for col in categorical_cols:
            # Get column data
            col_data = df_gtex[col].to_numpy().reshape(-1, 1)
            
            # Create encoder
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoder.fit(col_data)
            
            # Transform and add to list
            encoded = encoder.transform(col_data).toarray()
            categorical_arrays.append(encoded)
        
        # Combine all categorical features if any exist
        if categorical_arrays:
            categorical_covs = np.hstack(categorical_arrays)
            categorical_covs = categorical_covs.astype(int)
        else:
            categorical_covs = np.zeros((len(df_gtex), 0), dtype=int)
    else:
        # No categorical columns available/selected
        categorical_covs = np.zeros((len(df_gtex), 0), dtype=int)
    
    # Drop processed columns to get gene expression data
    columns_to_drop = []
    if numerical_cols:
        columns_to_drop.extend([col for col in numerical_cols if col in df_gtex.columns])
    if categorical_cols:
        columns_to_drop.extend([col for col in categorical_cols if col in df_gtex.columns])

    # Drop the processed columns to get gene expression data
    df_expression = df_gtex.drop(columns=columns_to_drop) if columns_to_drop else df_gtex.copy()

    # Handle string/non-numeric values in the expression data
    # First identify any non-numeric columns
    non_numeric_cols = []
    for col in df_expression.columns:
        # Check if column has any non-numeric values or is of object type
        if df_expression[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(df_expression[col]):
            non_numeric_cols.append(col)

    # If there are non-numeric columns, either convert or drop them
    if non_numeric_cols:
        print(f"Warning: Found non-numeric columns in expression data: {non_numeric_cols}")
        for col in non_numeric_cols:
            le = LabelEncoder()
            df_expression[col] = le.fit_transform(df_expression[col])

    # Check if we have any columns left after dropping non-numeric ones
    if df_expression.shape[1] == 0:
        raise ValueError("No numeric expression data columns remain after processing.")

    # Convert expression data to numpy array and ensure it's float32
    try:
        expression_data = df_expression.values
        expression_data = expression_data.astype(np.float32)
    except ValueError as e:
        print("Error converting expression data to float. Detailed column info:")
        for col in df_expression.columns:
            print(f"Column {col}: dtype={df_expression[col].dtype}, sample={df_expression[col].iloc[0]}")
        raise

    # Convert to torch tensors
    expression_tensor = torch.from_numpy(expression_data)
    numerical_tensor = torch.from_numpy(numerical_covs)
    categorical_tensor = torch.from_numpy(categorical_covs)
    
    return expression_tensor, numerical_tensor, categorical_tensor

def get_tcga_datasets(scaler_type:str="standard"):
    # Load train data
    X_train, numerical_covs, y_train = process_tcga_data(test=False, landmark=True)
    # Load test data
    X_test, numerical_covs_test, y_test = process_tcga_data(test=True, landmark=True)
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

def get_cambda_datasets(scaler_type:str="standard"):
    # Load train data
    X_train, numerical_covs, y_train = process_cambda_data(test=False, landmark=True)
    # Load test data
    X_test, numerical_covs_test, y_test = process_cambda_data(test=True, landmark=True)
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
                                        persistent_workers=False)
        
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


def wasserstein_loss(y_true: torch.tensor, y_pred: torch.tensor):
    """
    Returns Wasserstein loss (product of real/fake labels and critic scores on real or fake data)
    ----
    Parameters:
        y_true (torch.tensor): true labels (either real or fake)
        y_pred (torch.tensor): critic scores on real or fake data
    Returns:
        (torch.tensor): mean product of real labels and critic scores
    """
    return torch.mean(y_true * y_pred)


def generator_loss(fake_score: torch.tensor):
    """
    Returns generator loss i.e the negative scores of the critic on fake data.
    ----
    Parameters:
        fake_score (torch.tensor): critic scores on fake data
    Returns:
        (torch.tensor): generator loss"""

    return wasserstein_loss(-torch.ones_like(fake_score), fake_score)


def discriminator_loss(real_score: torch.tensor, fake_score: torch.tensor):
    """
    Compute and return the wasserstein loss of critic scores on real and fake data i.e: wassertstein_loss = mean(-score_real) + mean(score_fake)
    ----
    Parameters:
        real_score (torch.tensor): critic scores on real data
        fake_score (torch.tensor): critic scores on fake data
    Returns:
        (torch.tensor): wasserstein loss
    """
    real_loss = wasserstein_loss(-torch.ones_like(real_score), real_score)
    fake_loss = wasserstein_loss(torch.ones_like(fake_score), fake_score)

    return real_loss, fake_loss


def load_model(
        model,
        path: str = None,
        location: str = "cuda:0"):
    """
    Loading previously saved model.
    ----
    Parameters:
        model (torch.nn.module): Pytorch model object to load.
        path (str): path to retrieve model weights.
        location (str): device (GPU/CPU) where to load model.
    Return:
        Loaded Pytorch model."""

    assert path is not None, "Please provide a path to load the Generator from."
    try:
        # Load model
        model.load_state_dict(torch.load(path, map_location=location))
        print('Model loaded.')
        return model
    except FileNotFoundError:  # if no model saved at given path
        print(f"No previously saved model at given path {path}.")


def print_training_time(t_begin):
    """
    Compute and print training time in seconds.
    """
    t_end = t.time()
    time_sec = t_end - t_begin
    print(
        f'Time of training: {round(time_sec, 4)} sec = {round(time_sec/60, 4)} minute(s) = {round(time_sec/3600, 4)} hour(s)')
    return time_sec


def save_weights(G, D, G_path: str, D_path: str,
                 hyperparameters_search: bool = False):
    """
    Save current weights at model checkpoint path when called.
    ----
    Parameters:
        G: generator.
        D: dicriminator.
        G_path (str): path to store generator.
        D_path (str): path to store discriminator.
        hyperparameters_search (bool): whether current training is performed for a search.
    """
    if hyperparameters_search:
        # The same storing path is used for a search.
        G_path = './checkpoints/search_gen.pt'
        D_path = './checkpoints/search_disc.pt'
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)

    print(
        f"--------------------\nDiscriminator saved at {D_path} and generator saved at {G_path}.")


def save_history(metric_history, metric_path: str):
    """
    Save metric history as numpy object.
    ----
    Parameters:
        metric_history: list/dict/np.array of training metric history
        metric path (str): path where to store metric history
    """
    # Save metrics history
    np.save(metric_path, metric_history)

def metrics_checkpoints_val(
        x_real,
        x_fake,
        nn,
        list_val,
        list_prec_recall,
        list_dens_cov,
        list_aats):
    """
    Compute metrics of interest at given epochs checkpoints.
    ----
    Parameters:
        x_real (torch.tensor): real data
        x_gen (torch.tensor): generated data
        list_val (list): list of validation score values at each checkpoint. Default None.
        list_prec_recall (list): list of precision and recall values at each checkpoint. Default None.
        list_discrep (list): list of discrepancy values at each checkpoint. Default None.
        list_discrep_real (list): list of discrepancy values at each checkpoint. Default None.
        list_aats (list): list of adversarial accuracy values at each checkpoint. Default None.

    Returns:
        Updated lists of metrics.
    """
    # Compute validation score
    list_val.append(gamma_coeff_score(x_real, x_fake))

    # Precision/recall on validation data
    prec, recall, density, cov = compute_prdc(x_real, x_fake, nn)
    list_prec_recall.append((prec, recall))
    list_dens_cov.append((density, cov))

    # AAts (adversarial accuracy) on validation data
    _, _, adversarial = compute_AAts(real_data=x_real, fake_data=x_fake)
    list_aats.append(adversarial)

    return list_val, list_prec_recall, list_dens_cov, list_aats
    

def epoch_checkpoint_val(x_real, x_gen,
                         list_val: list = None,
                         list_prec_recall: list = None,
                         list_aats: list = None,
                         list_dens_cov:list=None,
                         epoch: int = None):
    """
    Compute metrics of interest and plot of PCA projections at given epochs checkpoints.
    ----
    Parameters:
        x_real (torch.tensor): real data
        x_gen (torch.tensor): generated data
        list_val (list): list of validation score values at each checkpoint. Default None.
        list_prec_recall (list): list of precision and recall values at each checkpoint. Default None.
        list_aats (list): list of adversarial accuracy values at each checkpoint. Default None.
        epoch (int): current epoch iteration. Default None.
        pca_applied (bool): whether PCA reduction was applied on input data. Default False.
        nb_principal_components (int): number of principal components to perform reduction. Default None.
        pca_obj: sklearn PCA object previously trained on train data. Default None.


    Returns:
        Updated lists of metrics.
    """

    # Compute epoch checkpoint metrics
    list_val, list_prec_recall, list_dens_cov, list_aats = metrics_checkpoints_val(x_real, x_gen, list_val, list_prec_recall, list_dens_cov, list_aats)

    return list_val, list_prec_recall, list_dens_cov, list_aats


def epoch_checkpoint_train(x_real, x_gen,
                           nn,
                           list_val_score:list =None,
                           list_prec_recall_train:list = None,
                           list_dens_cov_train:list=None,
                           list_aats_train: list = None,
                           list_frechet_train:list=None,
                           dataset:str=None,
                           device:str=None):
    """
    Compute metrics of interest at given epochs checkpoints.
    ----
    Parameters:
        x_real (torch.tensor): real data
        x_gen (torch.tensor): generated data
        device_pretrained_cls (str): device where to load the pretrained classifier used for frechet distance scores.
        list_prec_recall_train (list): list of precision and recall values at each checkpoint on train data. Default None.
        list_aats_train (list): list of adversarial accuracy values at each checkpoint on train data. Default None.
        list_dens_cov_train (list): list of density and coverage values at each checkpoint on train data. Default None.
        list_frechet_train (list): list of Frechet Score values at each checkpoint on train data. Default None.

    Returns:
        Updated lists.
    """
    # Correlations
    #correlations = gamma_coeff_score(x_real, x_gen)
    correlations = 0.
    list_val_score.append(correlations)

    # Precision/recall on training data
    prec, recall, density, cov = compute_prdc(x_real, x_gen, nn)
    list_prec_recall_train.append((prec, recall))
    list_dens_cov_train.append((density, cov))

    # AAts (adversarial accuracy) on train data
    idx = np.random.choice(len(x_real), 4096, replace=False)
    _, _, adversarial = compute_AAts(real_data=x_real[idx], fake_data=x_gen[idx])
    list_aats_train.append(adversarial)

    # Frechet
    # frechet = compute_frechet_distance_score(x_real, x_gen, dataset, device, to_standardize=False)
    frechet = 0
    list_frechet_train.append(frechet)

    return list_val_score, list_prec_recall_train, list_dens_cov_train, list_aats_train, list_frechet_train


def print_func(metrics_dict: dict = None):
    """
    Print current epoch losses and validation scores.
    ----
    Parameters:
        metrics_dict (dict): dictionnary with metrics values to print. Default None.
    """

    print(
        '-------------\n Epoch {}. Gen loss: {:.2f}. Disc loss: {:.2f}. Val score: {:.2f}.'.format(
            metrics_dict['epoch'],
            round(
                metrics_dict['gen_loss'] /
                metrics_dict['nb_samples'],
                4),
            round(
                metrics_dict['disc_loss'] /
                metrics_dict['nb_samples'],
                4),
            round(
                metrics_dict['val_score'],
                4)))

    print(
        'Precision train: {}. Recall train: {}. Density train: {}. Coverage train: {}. AAts train: {}. FD train:{}. \n ------------------\n'.format(
            metrics_dict['precision_train'],
            metrics_dict['recall_train'],
            metrics_dict['density_train'],
            metrics_dict['coverage_train'],
            metrics_dict['AAts_train'],
            metrics_dict['FD_train']))


def write_log(file_path: str = None, metrics_dict: dict = None):
    """ 
    Write training logs in given path.
    ----
    Parameters:
        file_path (str): path where to write log. Default None.
        metrics_dict (dict): dictionary with information to write. Default None.
    """

    # Open and append
    with open(file_path, 'a') as f:
        f.write(
            '\n Epoch {}. \t Gen loss: {:.2f} \t Disc loss: {:.2f} \t Val score: {:.2f} \n'.format(
                metrics_dict['epoch'],
                round(
                    metrics_dict['gen_loss'] /
                    metrics_dict['nb_samples'],
                    4),
                round(
                    metrics_dict['disc_loss'] /
                    metrics_dict['nb_samples'],
                    4),
                round(
                    metrics_dict['val_score'],
                    4)))

        f.write(
            '\t Precision train: {} \t Recall train: {} \t Density train: {} \t Coverage train: {} \t AAts train: {} \t FD train:{} \n ------------------\n'.format(
                metrics_dict['precision_train'],
                metrics_dict['recall_train'],
                metrics_dict['density_train'],
                metrics_dict['coverage_train'],
                metrics_dict['AAts_train'],
                metrics_dict['FD_train']))


def write_config(file_path: str = None, config: dict = None):
    """ 
    Write model config in given path
    ----
    Parameters:
        file_path (str): path where to write config file. Default None.
        config (dict): dictionary with configuration information. Default None.
    """

    # Open and append file
    with open(file_path, 'a') as f:
        f.write('CONFIGURATION: \n \t Z latent dim: {} \t Nb Epochs: {} \t Batch size: {} \t Nb iters critic: {} \t Lambda penalty: {} \n \t Prob success: {} \t Norm scale: {} \t Optimizer: {} \t LR Gen: {}, \t LR Disc: {}\n \t Activation func: {} \t Negative_slope: {} \t Hidden_dim1_g: {} \t Hidden_dim2_g:{} \t Hidden_dim3_g:{} \n \t Hidden_dim4_g: {} \t Hidden_dim5_g:{} \t Hidden_dim1_d:{} \t Hidden_dim2_d:{}\n -------------------------------------'.format(
            config['latent_dim'], config['epochs'], config['batch_size'], config['iters_critic'], config['lambda_penalty'], config['prob_success'], config['norm_scale'], config['optimizer'], config['lr_g'], config['lr_d'], config['activation'], config['negative_slope'], config['hidden_dim1_g'], config['hidden_dim2_g'], config['hidden_dim3_g'], config['hidden_dim4_g'], config['hidden_dim5_g'], config['hidden_dim1_d'], config['hidden_dim2_d']))


class TrackLoss:
    """
    Callback tracking all components of the loss.
    ----
    Parameters:
        verbose (bool): whether to print information when callback is called. Default False.
        path (str): path where to store loss history dictionary as numpy object. Default 'loss_history.npy'.
        nb_epochs (int): total number of epochs in training. Default 0.
    """

    def __init__(
            self,
            verbose: bool = False,
            path: str = 'loss_history.npy',
            nb_epochs: int = 0):

        self.verbose = verbose
        self.path = path
        self.nb_epochs = nb_epochs
        self.history = {"g_loss_batch":[], "disc_loss_gp":[], "d_loss":[], "gp":[], "disc_loss_batch":[], "g_loss_epoch":[],
                        "disc_loss_epoch":[],  "real_loss":[], "fake_loss": []}

    def __call__(self, hist_dict: dict):
        """
        Main function to call to store current training history. History is saved as a numpy object at the final epoch.
        ----
        Parameters:
            hist_dict (dict): current epoch training history dictionary.
        """
        for key in hist_dict.keys():
            self.history[key].append(hist_dict[key])

        if len(self.history["disc_loss_epoch"]) == self.nb_epochs:
            # Save current training history
            self.save_history()

    def save_history(self):
        """ Saves training history"""
        if self.verbose:
            print(f'All losses components tracked and saved.')
        np.save(self.path, self.history)
