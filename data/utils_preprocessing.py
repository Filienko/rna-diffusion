# Imports
import os
import sys
from numpy import ndarray
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import warnings
import random
warnings.filterwarnings('ignore') #Ignore warnings

# Path (HARD CODED)
PATH = '/data/tcga_files/data_RNAseq_RTCGA'

# SEED
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

COHORTS = ["ACC","BLCA","BRCA","CESC","CHOL","COAD","DLBC","ESCA","GBM", "HNSC","KICH","KIRC","KIRP","LAML","LGG","LIHC","LUAD","LUSC","MESO","OV","PAAD",
            "PCPG","PRAD","READ","SARC","SKCM","STAD",
            "STES","TGCT","THCA","THYM","UCEC","UCS","UVM"]

CANCER_TYPES = ['adrenocortical carcinoma', 'bladder urothelial carcinoma', 'breast invasive carcinoma',
                            'cervical squamous cell carcinoma endocervical adenocarcinoma', 'cholangiocarcinoma', 'colon adenocarcinoma',
                            'lymphoid neoplasm diffuse large b-cell lymphoma', 'esophageal carcinoma', 'glioblastoma multiforme', 'head neck squamous cell carcinoma',
                            'kidney chromophobe', 'kidney renal clear cell carcinoma', 'kidney renal papillary cell carcinoma', 'acute myeloid leukemia',
                            'brain lower grade glioma', 'liver hepatocellular carcinoma', 'lung adenocarcinoma', 'lung squamous cell carcinoma',
                            'mesothelioma', 'ovarian serous cystadenocarcinoma', 'pancreatic adenocarcinoma', 'pheochromocytoma and paraganglioma',
                            'prostate adenocarcinoma', 'rectum adenocarcinoma', 'sarcoma', 'skin cutaneous melanoma', 'stomach adenocarcinoma',
                            'stomach esophageal carcinoma', 'testicular germ cell tumors', 'thyroid carcinoma', 'thymoma', 'uterine corpus endometrial carcinoma',
                            'uterine carcinosarcoma', 'uveal melanoma', 'normal'] # add 'normal' label for healthy samples

TISSUES = ['adrenal', 'bladder', 'breast', 'cervical', 'liver', 'colon', 'blood', 'esophagus', 'brain', 'head',
        'kidney', 'kidney', 'kidney', 'blood', 'brain', 'liver', 'lung', 'lung', 'lung', 'ovary', 'pancreas', 'kidney',
        'prostate','rectum', 'soft-tissues', 'skin', 'stomach', 'stomach', 'testes', 'thyroid', 'thymus', 'uterus',
        'uterus', 'eye']

# Functions

def get_patients_ids(cohort:str=None, path_ids:str=None, path_cancer_types:str=None):
    """
    Retrieve reduced patients IDs of given cohort (order preserved).
    ----
    Parameters:
        cohort (str): given cancer type
        path_ids (str): path where patients IDs have been stored
        path_cancer_types (str): path where patients cancer types have been stored
    Returns:
        reduced_ids_cancer (np.array): truncated bar codes of given cancer type
    """
    #Init patients ids
    patient_ids = np.load(path_ids, allow_pickle=True)
    cancer_types = np.load(path_cancer_types, allow_pickle=True)

    #Extract reduced barcode as in clinical data
    reduced_ids_cancer = np.asarray([patient_ids[i].split('-', 3)[0]+'-'+patient_ids[i].split('-', 3)[1]+'-'+patient_ids[i].split('-', 3)[2] for i in range(len(patient_ids))])
    reduced_ids_cancer = reduced_ids_cancer[cancer_types==cohort]

    return reduced_ids_cancer


def check_duplicates(cohort:str, data:pd.DataFrame):
    """
    Returns duplicates indexes.
    ----
    Parameters:
        cohort (str): cohort for which to retrieve info
        df (pd.DataFrame): input dataframe on which duplicates are found
    Returns:
        dup_to_remove (list): list of duplicates indexes to remove
    """

    reduced_ids_cancer = get_patients_ids(cohort)

    if len(np.unique(reduced_ids_cancer)) != len(reduced_ids_cancer): # check if (same or different) duplicates in barcodes
        print(f'Duplicates found in barcodes for {cohort}.')

        # Find duplicated patients barcodes with pandas.series()
        reduced_ids_cancer_series = pd.Index(reduced_ids_cancer)
        barcode_dup = reduced_ids_cancer_series[reduced_ids_cancer_series.duplicated()]

        # Find if exact duplicates in data that need to be removed
        dup_to_remove = duplicates_to_remove(keys=reduced_ids_cancer_series, duplicated_keys=barcode_dup, data_to_compare=data)

        return dup_to_remove

    else:
        return []


def update_followup(df:pd.DataFrame):
    """Update survival time with number of days to 'death' if event occured.
    ----
    Parameters:
        df (pd.DataFrame): input dataframe on which followup data is to be updated
    Returns:
        df (pd.DataFrame): updated dataframe
    """
    #There are more patients confirmed dead than missing followup information.
    #If double information: always replace missing followup info by 'days to death'
    #Fill NaNs in followup info with days_to_death
    df['followup'].fillna(df['death'].loc[df[df['followup'].isna()].index], inplace=True)
    #Make sure that we keep days_to_death info if double info
    df[df['status']=='dead']['followup'] = df[df['status']=='dead']['death'].values
    
    return df


def remove_nans(df: pd.DataFrame, task:str='covariates'):
    """ Returns dataframe without missing data for specific catgeories.
    ----
    Parameters:
        df (pd.DataFrame): input dataframe in which missing data are removed
        task (str): training task (either 'covariates', 'survival' or 'all')
    Returns:
        df (pd.DataFrame): updated dataframe
    """
    
    if task=='covariates':
        print(f"{len(df[df['age'].isna()].index.values)} missing data for covariate 'age'.")
        print(f"{len(df[df['gender'].isna()].index.values)} missing data for covariate 'gender'.")
        df.drop(df[df['age'].isna()].index.values, inplace=True)
        df.drop(df[df['gender'].isna()].index.values, inplace=True)
        
    elif task=='survival':
        print(f"{len(df[df['followup'].isna()].index.values)} missing data for col 'followup'.")
        print(f"{len(df[df['status'].isna()].index.values)} missing data for col 'status'.")
        print(f"{len(df[df['cancer_stage'].isna()].index.values)} missing data for col 'cancer_stage'.")
        df.drop(df[df['followup'].isna()].index.values, inplace=True)
        df.drop(df[df['status'].isna()].index.values, inplace=True)
        df.drop(df[df['cancer_stage'].isna()].index.values, inplace=True)
        
    elif task=='all':
        print(f"{len(df[df['age'].isna()].index.values)} missing data for covariate 'age'.")
        print(f"{len(df[df['gender'].isna()].index.values)} missing data for covariate 'gender'.")
        print(f"{len(df[df['followup'].isna()].index.values)} missing data for col 'followup'.")
        print(f"{len(df[df['status'].isna()].index.values)} missing data for col 'status'.")
        print(f"{len(df[df['cancer_stage'].isna()].index.values)} missing data for col 'cancer_stage'.")
        df.drop(df[df['age'].isna()].index.values, inplace=True)
        df.drop(df[df['gender'].isna()].index.values, inplace=True)
        df.drop(df[df['followup'].isna()].index.values, inplace=True)
        df.drop(df[df['status'].isna()].index.values, inplace=True)
        df.drop(df[df['cancer_stage'].isna()].index.values, inplace=True)
    
    return df


def process_all(df:pd.DataFrame):
    """ Returns dataframe with all processed variables: duplicates removed, NaNs removed and data encoded.
    ----
    Parameters:
        df (pd.DataFrame): input dataframe on which data is to be processed
    Returns:
        df (pd.DataFrame): processed dataframe
    """
    
    # Check duplicates 
    samples_to_remove = []
    for c in COHORTS:
        df_temp = df[df['cohort']==c]
        samples_to_remove.extend(check_duplicates(c, df_temp))

    # Drop duplicates
    df.drop(samples_to_remove, inplace=True)
    # Drop cancer column (only cancerous samples are kept)
    df.drop('cancer', axis=1, inplace=True)
    # Get corect survival days data
    df = update_followup(df)
    # Remove NaNs
    df = remove_nans(df, task='all')
    # 1-hot encoding for status column
    df['status'] = df['status'].apply(lambda status: 0.0 if status=='alive' else 1.0)
    # 1-hot encoding for gender column
    df['gender'] = df['gender'].apply(lambda gender: 0.0 if gender=='male' else 1.0)
    # Get age in correct type
    df['age'] = df['age'].apply(lambda age: int(age))
    # Encoding cancer stage column
    df['cancer_stage'] = df['cancer_stage'].apply(lambda stage: stage_encoding(stage))

    return df

def process_covariates(df:pd.DataFrame):
    """
    Returns dataframe with all processed covariates: duplicates removed, NaNs removed and data encoded.
    ----
    Parameters:
        df (pd.DataFrame): input dataframe to process
    Returns:
        df (pd.DataFrame): processed dataframe
    """
    # Check duplicates 
    samples_to_remove = []
    for c in COHORTS:
        df_temp = df[df['cohort']==c]
        samples_to_remove.extend(check_duplicates(c, df_temp))

    # Drop duplicates
    df.drop(samples_to_remove, inplace=True)
    # Remove NaNs
    df = remove_nans(df, task='covariates')
    # 1-hot encoding for gender column
    df['gender'] = df['gender'].apply(lambda gender: 0.0 if gender=='male' else 1.0)
    # Get age in correct type
    df['age'] = df['age'].apply(lambda age: int(age))
    # Keep only columns of interest
    df.drop(['followup', 'status', 'race', 'death', 'cancer_stage'], axis=1, inplace=True)

    return df

def process_survival(df:pd.DataFrame):
    """
    Returns dataframe with all processed survival data: duplicates removed, NaNs removed and data encoded.
    ----
    Parameters:
        df (pd.DataFrame): input dataframe to process
    Returns:
        df (pd.DataFrame): processed dataframe
    """
    # Check duplicates 
    samples_to_remove = []
    for c in COHORTS:
        df_temp = df[df['cohort']==c]
        samples_to_remove.extend(check_duplicates(c, df_temp))

    # Drop duplicates
    df.drop(samples_to_remove, inplace=True)
    # Drop cancer column as only cancerous samples are kept
    df.drop('cancer', axis=1, inplace=True)
    # Get corect survival days data
    df = update_followup(df)
    # Remove NaNs
    df = remove_nans(df, task='survival')
    # 1-hot encoding for status column
    df['status'] = df['status'].apply(lambda status: 0.0 if status=='alive' else 1.0)
    # 1-hot encoding for gender column
    df['cancer_stage'] = df['cancer_stage'].apply(lambda stage: stage_encoding(stage))
    # Keep only columns of interest
    df.drop(['age', 'gender', 'race', 'death'], axis=1, inplace=True)
    
    return df

def pca_reduction(data, pca_obj, as_tensors:bool=False):
    """
    """
    data_reduced = pca_obj.transform(data)

    if as_tensors:
        return torch.from_numpy(data_reduced).float()
    else:
        return data_reduced
        

def standardize(x, mean=None, std=None):
    """
    Standardize input with mean and standard deviation.
    ----
    Parameters:
        x (ndarray or float): data of size (nb_samples, nb_vars) to standardize
        mean (ndarray or float): mean parameter(s) for standardization
        std (ndarray or float): std parameter(s) for standardization
    Returns:
        (ndarray or float): standardized input
    """
    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    return (x - mean) / std

def train_test_split(df:pd.DataFrame, split_ratios:list=[0.8, 0.2], shuffle:bool=True):
        """
        Returns train and test split dataframes.
        ----
        Parameters:
            df (pd.DataFrame): dataframe of data to split
            split_ratios (list): list of train-test split ratios (default [0.8, 0.2])
            shuffle (bool): whether to shuffle data before split (default True)
        Returns:
            train_df (pd.DataFrame), test_df (pd.DataFrame): train and test dataframes

        """
        if shuffle:
            # Set random shuffling
            shuffler = np.random.permutation(len(df))
            # Reset indexes
            df.reset_index(drop=True, inplace=True)
            # Shuffle
            df = df.loc[list(shuffler)]

        # Set split ratios
        train_ratio, test_ratio = split_ratios
        assert train_ratio+test_ratio==1, f"Split ratios {split_ratios} must equal to 1."

        # Split dataframe
        train_df = df[:int(len(df)*train_ratio)]
        test_df = df[int(len(df)*train_ratio):]

        return train_df, test_df

def standardize_train(data, as_tensors=True):
    """
    """
    # Init scaler
    scaler = StandardScaler()
    scaler.fit(data)
    standardized_data = scaler.transform(data)
    
    if standardized_data.shape[1]==1: # only 1 feature
        standardized_data = standardized_data.flatten()
    
    if as_tensors:
        standardized_data = torch.from_numpy(standardized_data).float()

    return standardized_data, scaler

def standardize_test(data, scaler, as_tensors=True):
    """
    """
    scaler.fit(data)
    standardized_data = scaler.transform(data)
    
    if standardized_data.shape[1]==1: # only 1 feature
        standardized_data = standardized_data.flatten()
    
    if as_tensors:
        standardized_data = torch.from_numpy(standardized_data).float()

    return standardized_data, scaler


def standardize_split(df_train:pd.DataFrame, df_test:pd.DataFrame, task:str='covariates'):
        """
        Standardize train-test dataframes with training parameters and returns those parameters in a third dataframe.
        ----
        Parameters:
            df_train (pd.DataFrame): train dataframe
            df_test (pd.DataFrame): test dataframe
            task (str): depending on the task, the function standardizes only some specific clinical data. Categorical variables are not standardized.
        Returns:
            df_train (pd.DataFrame), df_test (pd.DataFrame), df_params (pd.DataFrame), mean (np.array), std (np.array)
        """  
        # Get genes id columns
        GENE_IDS = np.load(PATH+'/TCGA_rnaseq_RSEM_gene_ids_full.npy', allow_pickle=True)
        
        # Standardize gene expression data
        mean_exp = np.mean(df_train[list(GENE_IDS)].to_numpy(), axis = 0)
        std_exp = np.std(df_train[list(GENE_IDS)].to_numpy(), axis = 0)

        # Check whether there are null stds and replace by one to avoid division by 0
        std_exp[std_exp ==0.0] = 1.
        
        df_train_final = pd.DataFrame(data=(df_train[list(GENE_IDS)] - mean_exp)/std_exp, columns= list(GENE_IDS))
        df_test_final = pd.DataFrame(data=(df_test[list(GENE_IDS)] - mean_exp)/std_exp, columns= list(GENE_IDS))

        # Fillna if division by zero
        df_train_final.fillna(0,inplace=True)
        df_test_final.fillna(0,inplace=True)

       # Standardize clinical data
        if task=='covariates':
            mean = np.mean(df_train['age'].values, axis=0)
            std = np.std(df_train['age'].values, axis=0)
            df_train['age'] = df_train['age'].apply(lambda x: standardize(x, mean, std))
            df_test['age'] = df_test['age'].apply(lambda x: standardize(x, mean, std))
            df_params = pd.DataFrame({f'age': [mean, std]}, index=['mean', 'std']) # Store standardization parameters

        elif task=='survival':
            mean = np.mean(df_train['followup'].values, axis=0)
            std = np.std(df_train['followup'].values, axis=0)
            df_train['followup'] = df_train['followup'].apply(lambda x: standardize(x, mean, std))
            df_test['followup'] = df_test['followup'].apply(lambda x: standardize(x, mean, std))
            df_params = pd.DataFrame({f'followup': [mean, std]}, index=['mean', 'std']) # Store standardization parameters

        elif task=='all':
            means=[]
            stds=[]
            for col in ['age', 'followup']:
                mean_temp = np.mean(df_train[col].values, axis=0)
                std_temp = np.std(df_train[col].values, axis=0)
                df_train[col] = df_train[col].apply(lambda x: standardize(x, mean_temp, std_temp))
                df_test[col] = df_test[col].apply(lambda x: standardize(x, mean_temp, std_temp))
                means.append(mean_temp)
                stds.append(std_temp)
            df_params = pd.DataFrame({f'{col}': [m, s] for col, m, s in zip(['age', 'followup'], means, stds)}, index=['mean', 'std']) # Store standardization parameters
        
        # Add categorical data
        for col in ['full_barcode', 'tissue_type', 'cancer_type', 'reduced_barcode', 'cohort', 'cancer', 'barcode', 'gender', 'age']:
            df_train_final[col] = df_train[col]
            df_test_final[col] = df_test[col]

        return df_train_final, df_test_final, df_params, mean_exp, std_exp
    

def df_to_data(df:pd.DataFrame, as_type:str='tensor', task:str='covariates'):
    """
    Returns data from dataframe in arrays or tensors.
    ----
    Parameters:
        df (pd.DataFrame): input dataframe
        as_type (str): whether to return data as 'array' or 'tensor' (default 'tensor')
        task (str): training task
    Returns:
        (torch.tensor or np.array): gene expression data and corresponding clinical data
    """

     # Get genes id columns
    GENE_IDS = np.load(PATH+'/TCGA_rnaseq_RSEM_gene_ids_full.npy', allow_pickle=True)

    if task=='all':
        age = df['age'].to_numpy()
        gender = df['gender'].to_numpy()
        cancer_type = df['cancer_type'].to_numpy()
        tissue_type = df['tissue_type'].to_numpy()
        time = df['followup'].to_numpy()
        status = df['status'].to_numpy()
        cancer_stage = df['cancer_stage'].to_numpy()
        cohort = df['cohort'].to_numpy()
        x_exp = df[list(GENE_IDS)].to_numpy()
        # Put infs to 0
        x_exp[np.isinf(x_exp)] = 0

        if as_type=='tensor':
             # Conversion to tensors
            age = torch.from_numpy(age).float()
            gender = torch.from_numpy(gender).long()
            #cancer_type = torch.from_numpy(cancer_type)
            #tissue_type = torch.from_numpy(tissue_type)
            time = torch.from_numpy(time).float()
            status = torch.from_numpy(status).long()
            cancer_stage = torch.from_numpy(cancer_stage).float()
            x_exp = torch.from_numpy(x_exp).float()
        return x_exp, age, gender, cancer_type, tissue_type, time, status, cancer_stage, cohort

    elif task =='covariates':
        age = df['age'].to_numpy()
        gender = df['gender'].to_numpy()
        cancer_type = df['cancer_type'].to_numpy()
        tissue_type = df['tissue_type'].to_numpy()
        labels = df['cancer'].to_numpy()
        cohort = df['cohort'].to_numpy()
        x_exp = df[list(GENE_IDS)].to_numpy()
        # Put infs to 0
        x_exp[np.isinf(x_exp)] = 0

        if as_type=='tensor':
            # Conversion to tensors
            age = torch.from_numpy(age).float()
            gender = torch.from_numpy(gender).long()
            #cancer_type = torch.from_numpy(cancer_type)
            #tissue_type = torch.from_numpy(tissue_type)
            labels = torch.from_numpy(labels).long()
            x_exp = torch.from_numpy(x_exp).float()
        return x_exp, age, gender, cancer_type, tissue_type, labels, cohort

    elif task=='survival':
        age = df['age'].to_numpy()
        gender = df['gender'].to_numpy()
        cancer_type = df['cancer_type'].to_numpy()
        tissue_type = df['tissue_type'].to_numpy()
        time = df['followup'].to_numpy()
        status = df['status'].to_numpy()
        cancer_stage = df['cancer_stage'].to_numpy()
        cohort = df['cohort'].to_numpy()
        x_exp = df[list(GENE_IDS)].to_numpy()
        # Put infs to 0
        x_exp[np.isinf(x_exp)] = 0
        
        if as_type=='tensor':
            # Conversion to tensors
            age = torch.from_numpy(age).float()
            gender = torch.from_numpy(gender).long()
            #cancer_type = torch.from_numpy(cancer_type)
            #tissue_type = torch.from_numpy(tissue_type)
            time = torch.from_numpy(time).float()
            status = torch.from_numpy(status).long()
            cancer_stage = torch.from_numpy(cancer_stage).float()
            x_exp = torch.from_numpy(x_exp).float()
        return x_exp, age, gender, cancer_type, tissue_type, time, status, cancer_stage, cohort


def duplicates_to_remove(keys:pd.Series, duplicated_keys: pd.Series , data_to_compare:pd.DataFrame):
    """
    Finds duplicated samples indexes to remove.
    ----
    Parameters:
        keys (pd.Series): series of all keys in data
        duplicated_keys (pd.Series): series of duplicated keys previously found
        data_to_compare (ndarray): array of data used to define if there are exact duplicated or just duplicated keys
    Returns:
        dup_to_remove (list): list of exact duplicates indexes if found, else empty list
    """

    #One dataset can have more than 1 key duplicated. Therefore, we need to check each unique key:
    dup_to_remove = []

    for KEY in duplicated_keys.unique():
        #Look where we have duplicates of given barcode
        key_dup_idx = np.where(keys == KEY)[0]

        #Check if gene expression data is different for each sample with duplicated key
        for i in range(len(key_dup_idx)):
            og_data = data_to_compare[key_dup_idx[i]] #init original duplicate

            #Loop over similar duplicates
            for idx in key_dup_idx[i+1:]:
                if (og_data == data_to_compare[idx]).sum() == og_data.shape[0]:
                    dup_to_remove.append(idx)

    #Remove exact duplicates if any
    if len(dup_to_remove)>0:
        #Drop duplicates in data
        print('Exact duplicate same data was found.')

    return dup_to_remove



def check_missing_data(cohort:str, clinical_data_barcodes:np.array):
    """
    Finds whether barcodes from gene expression data and clinical data are missing.
    ----
    Parameters:
        cohort (str): given cancer type
        clinical_data (np.array): patients barcodes from clinical data
    Returns (list): list of missing clinical data indexes to remove from gene expression data
    """
    reduced_ids_cancer = get_patients_ids(cohort)

    #Remove patients for which we don't have clinical data
    miss_to_remove = []
    for index, barcode in enumerate(reduced_ids_cancer):
        if barcode.lower() not in clinical_data_barcodes:
            miss_to_remove.append(index)

    return miss_to_remove


def check_inconsistencies(cohort:str, df:pd.DataFrame):
    """
    Finds whether there are inconsistent values (e.g negative number of survival days...etc).
    ----
    Parameters:
        cohort (str): given cohort
        df (pd.DataFrame): survival dataframe
    Returns (list): list of inconsistent data indexes found, else empty list
    """
    FOLLOWUP_COL =''
    BARCODE_COL = ''

    inconst_barcodes = df[df[FOLLOWUP_COL]<'0'][BARCODE_COL].values

    if len(inconst_barcodes)>0:
        reduced_ids_cancer = get_patients_ids(cohort)

        inconst_to_remove = []
        for barcode in inconst_barcodes:
            indexes = np.where(reduced_ids_cancer==barcode.upper())[0]
            inconst_to_remove.extend(indexes)

        return inconst_to_remove

    else:
        return []


def stage_encoding(stage:str or float):
    """
    Encoding cancer stages from 'string' type to value.
    ----
    Parameters:
        stage (str or float): cancer stage in clinical data (float type if NaN)
    Returns:
        stage_encod (int): cancer stage as an integer
    """

    if type(stage) is not float: # Exclude NaNs
        level = stage.lower() # Lower

        if level == 'i/ii nos': # Special case
            stage_encod = 2
        else:
            stage_encod = level.count('i')+level.count('v')*3

        return int(stage_encod)


def stage_t_encoding(stage_t:str or float):
    """
    Encode pathologic cancer stage from 'string' to value.
    ----
    Parameters:
        cancer_stage(str): pathologic cancer stage of primary tumor
    Returns:
        stage_t_encode (int): cancer stage as an integer"""

    if type(stage_t) is not float: #Exclude NaNs
        level = stage_t.lower() #Lower

        if level == 'i/ii nos': #Special case
            stage_t_encod = 2
        else:
            if 'a' in level:
                stage_t_encod = int(level.split('t')[1].split('a')[0])
            elif 'b' in level:
                stage_t_encod = int(level.split('t')[1].split('b')[0])
            elif 'c' in level:
                stage_t_encod = int(level.split('t')[1].split('c')[0])
            elif 'd' in level:
                stage_t_encod = int(level.split('t')[1].split('d')[0])
            elif 'x' in level or 's':
                stage_t_encod = 0
            else:
                stage_t_encod = int(level.split('t')[1])

        return int(stage_t_encod)


def categorical_labels(labels, as_:str='categorical', nb_class:int=34):
    """ Returns labels as categorical data (integers forPytorch models or one-hot encoding).
    ----
    Parameters:
        as_ (str): whether to return labels as categorical integers or one hot encoded integers.
        nb_class (int): number of classes in labels
    Returns:
        cancer_types (np.array): cancer types as categories or one-hot encoded
    """
    df_CT = pd.DataFrame(labels, columns=['cancer'])
    #1-hot encoding
    df_CT_1hot = pd.get_dummies(df_CT.cancer, prefix='type')
    cancer_types = df_CT_1hot.to_numpy()

    if as_=='one_hot':
        return cancer_types
    elif as_=='categorical':
        idx = np.arange(nb_class).reshape(1, nb_class)
        return np.repeat(idx, len(cancer_types), axis=0)[cancer_types==1]

    
def one_hot_to_categoricals(labels, nb_class:int=34):
    """ Returns labels as categorical data (integers for Pytorch models).
    ----
    Parameters:
        labels (np.array or torch.tensor): one-hot encoded labels
        nb_class (int): number of classes in labels
    """
    idx = np.arange(nb_class).reshape(1, nb_class)
    return np.repeat(idx, len(labels), axis=0)[labels==1]
