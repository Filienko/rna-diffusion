import numpy as np
import torch
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
from sklearn.metrics import roc_auc_score


# Metrics
def confusion_matrix(labels:np.array, predicted_labels:np.array, nb_class:int=2):
    """ Computes and returns confusion matrix.
    ----
    Parameters:
        labels (np.array): true labels
        predicted_labels (np.array): predicted_labels
        nb_class (int): number of classes to classify
    Returns:
        confusion_matrix (np.array) of size (nb_class, nb_class)
    """
    # Init matrix
    confusion_matrix = np.zeros((nb_class, nb_class), dtype=np.int64)

    # Fill matrix
    for i in range(len(labels)):
        confusion_matrix[int(labels[i]), int(predicted_labels[i])] += 1

    return confusion_matrix


def compute_auc(y_true:torch.tensor, y_pred:torch.tensor):
    """ Compute validation AUC based on true labels and model predictions.
    ----
    Parameters:
        y_true (torch.tensor): true labels
        y_pred (torch.tensor): predicted logits
        classes (str): whether the training class is 'binary' and 'multiclass'
    Returns
        auc (float): AUC
    """
    # Stack values
    #y_true = torch.hstack(y_true)
    #y_pred = torch.hstack(y_pred)

    # For AUC computation, y_pred must be probabilities for each class and sum to 1
    y_pred = torch.vstack(y_pred)
    y_pred = torch.softmax(y_pred, dim=-1)
        
    auc = roc_auc_score(y_true, y_pred, multi_class='ovo') # One-vs-one algorithm for AUC computation

    return auc

def get_class_weights(train_loader, val_loader):
    tissue_labels = []
    for i, (_, labels) in enumerate(train_loader):
        tissue_labels.append(labels.argmax(1))
    
    for i, (_, labels) in enumerate(val_loader):
        tissue_labels.append(labels.argmax(1))

    tissue_labels = torch.hstack(tissue_labels)
    tissue_weights = torch.tensor([len(tissue_labels)/(tissue_labels==t).sum().item() for t in range(len(torch.unique(tissue_labels)))])
    
    return tissue_weights


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

def clean_data(df, keep_cols = ['tissue_type']):
    # Prepare data for basic binary prediction
    col_names_not_dna = [col for col in df.columns if not col.isdigit()]

    for col in keep_cols:
        col_names_not_dna.remove(col)

    df = df.drop(columns = col_names_not_dna)
    # Remove rows with NaN
    df = df[~pd.isnull(df).any(axis=1)]

    return df

def process_tcga_data(test:bool=False, landmark:bool=False):
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

    Tissue_Encoder = OneHotEncoder(handle_unknown='ignore') # Init encoder
    Tissue_Encoder.fit(np.unique(TISSUES).reshape(-1,1))

    categorical_covs = df_tcga['tissue_type'].to_numpy()
    #reshape to features vector
    categorical_covs = categorical_covs.reshape((-1,1))
    categorical_covs = Tissue_Encoder.transform(X = categorical_covs)
    print(categorical_covs.shape)

    #tissues types as one hot
    categorical_covs = categorical_covs.astype(int)

    true = df_tcga.drop(columns = ['age','gender','cancer','tissue_type'])
    
    if landmark: # Get only 978 landmark genes
        df_tcga_union_landmark = pd.read_csv('/home/alacan/scripts/gerec_pipeline/tcga_union_978landmark_genes.csv') # Load landmark genes ids
        df_tcga_union_landmark['genes_ids'] = df_tcga_union_landmark['genes_ids'].astype('str')
        true = true[df_tcga_union_landmark['genes_ids']] # keep only the genes ids in landmark genes ids

    else:
        df_tcga_union_landmark = pd.read_csv('/home/alacan/scripts/gerec_pipeline/tcga_union_978landmark_genes.csv') # Load landmark genes ids
        list_tcga_union_landmark = df_tcga_union_landmark['genes_ids'].values
        df_tcga_union_landmark['genes_ids'] = df_tcga_union_landmark['genes_ids'].astype('str')
        # Landmarks
        landmark = df_tcga_union_landmark['genes_ids'].values.flatten()
        # Non landmarks
        non_landmark = np.array([str(i) for i in list(true.columns) if int(i) not in list_tcga_union_landmark])
        # Order genes
        true = true[np.append(landmark, non_landmark)]
        # debug
        print(true.shape)
    
    true = true.values
    true = true.astype(np.float32)

    #convert to torch tensor
    true = torch.from_numpy(true)
    numerical_covs = torch.from_numpy(numerical_covs)
    categorical_covs = torch.from_numpy(categorical_covs.toarray())

    return true, numerical_covs, categorical_covs

def process_gtex_data(test:bool=False, landmark:bool=False):
    df_gtex = load_gtex(test)
    # Remove rows with NaN
    df_gtex = df_gtex[~pd.isnull(df_gtex).any(axis=1)]

    #debug
    print(df_gtex.shape)

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

    df_gtex = df_gtex.drop(columns = ['age','gender','tissue_type', 'ID'])
    
    # Order by landmark genes and target genes
    df_descript = pd.read_csv('/home/alacan/GTEx_data/gtex_description.csv', sep=',')
    # filter
    genes_ids = np.append(df_descript[df_descript.Type=='landmark']['Description'].values.flatten(), df_descript[df_descript.Type=='target']['Description'].values.flatten())
    df_gtex = df_gtex[genes_ids].values
    df_gtex = df_gtex.astype(np.float32)

    #debug
    print(df_gtex.shape)

    # convert to torch tensor
    df_gtex = torch.from_numpy(df_gtex)
    numerical_covs = torch.from_numpy(numerical_covs)
    categorical_covs = torch.from_numpy(categorical_covs.toarray())

    return df_gtex, numerical_covs, categorical_covs

def get_tcga_datasets(scaler_type:str="standard"):
    # Load train data
    X, numerical_covs, y = process_tcga_data(test=False, landmark=True)
    # Load test data
    X_test, numerical_covs_test, y_test = process_tcga_data(test=True, landmark=True)
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

def get_datasets_for_search(dataset:str):
    """
    """
    # Load train data
    if dataset=='tcga':
        process_func = process_tcga_data
    elif dataset =='gtex':
        process_func = process_gtex_data
    X, numerical_covs, y = process_func(test=False, landmark=False)
    # Load test data
    X_test, numerical_covs_test, y_test = process_func(test=True, landmark=False)

    return X, y, X_test, y_test

def split_and_scale_datasets(X, y, X_test, y_test, scaler_type:str="standard"):
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
    else:
        test_loader = None

    return train_loader, val_loader, test_loader 


def get_debug_datasets(config):
    """
    Build debugging train and test datasets based on Normal random variables.
    """

    # Generate random data
    X_train, X_valid, X_test = np.random.random((1000, 20531)), np.random.random((500, 20531)), np.random.random((500, 20531))
    # Labels
    TISSUES = np.array(['adrenal', 'bladder', 'breast', 'cervical', 'liver', 'colon', 'blood', 'esophagus', 'brain', 'head', 'kidney', 'kidney', 'kidney', 'blood', 'brain', 'liver', 'lung', 'lung', 'lung', 'ovary', 'pancreas', 'kidney', 'prostate','rectum', 'soft-tissues', 'skin', 'stomach', 'stomach', 'testes', 'thyroid', 'thymus', 'uterus', 'uterus', 'eye'])
    y_train, y_valid, y_test = TISSUES[np.random.randint(0, len(TISSUES), (1000,))].reshape(-1,1), TISSUES[np.random.randint(0, len(TISSUES), (500,))].reshape(-1,1), TISSUES[np.random.randint(0, len(TISSUES), (500,))].reshape(-1,1)
    # One hot encoding
    Tissue_Encoder = OneHotEncoder(handle_unknown='ignore') # Init encoder
    Tissue_Encoder.fit(np.unique(TISSUES).reshape(-1,1))
    
    y_train, y_valid, y_test = Tissue_Encoder.transform(X=y_train), Tissue_Encoder.transform(X=y_valid), Tissue_Encoder.transform(X=y_test)

    # Scale the data
    if config.data.scaler_type == "standard":
        scaler = StandardScaler()
    elif config.data.scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif config.data.scaler_type == "quantile":
        # scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(20000 // 30, 1000), 10), random_state=0, subsample=1000000000)
        scaler = QuantileTransformer(output_distribution='normal', n_quantiles=config.data.n_quantiles, random_state=42)
    elif config.data.scaler_type == "robust":
        scaler = RobustScaler()
    elif config.data.scaler_type == "maxabs":
        scaler = MaxAbsScaler()
    elif config.data.scaler_type == "custom":
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = MinMaxScaler()
    elif config.data.scaler_type == "none":
        scaler = None
    else:
        raise Exception("Unknown scaler type")


    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # print('X_train', X_train.shape)
    # print('y_train', y_train.shape)

    # #print('X_train', X_train)
    # print('y train', np.array(y_train))
    # print('y train[0]', y_train[0])

    # Turn data into tensors
    X_train = torch.tensor(X_train).type(torch.float)
    X_valid = torch.tensor(X_valid).type(torch.float)
    X_test = torch.tensor(X_test).type(torch.float)
    y_train = torch.from_numpy(y_train.toarray())
    y_valid = torch.from_numpy(y_valid.toarray())
    y_test = torch.from_numpy(y_test.toarray())


    train_tensor = data_utils.TensorDataset(X_train, y_train) 
    test_tensor = data_utils.TensorDataset(X_test, y_test) 
    valid_tensor = data_utils.TensorDataset(X_valid, y_valid)

    return train_tensor, test_tensor

def get_gtex_datasets(scaler_type:str="standard"):
    # Load train data
    X, numerical_covs, y = process_gtex_data(test=False, landmark=True)
    # Load test data
    X_test, numerical_covs_test, y_test = process_gtex_data(test=True, landmark=True)
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # Turn data into tensors
    X_train = torch.tensor(X_train).type(torch.float)
    X_val = torch.tensor(X_val).type(torch.float)
    X_test = torch.tensor(X_test).type(torch.float)

    train = data_utils.TensorDataset(X_train, y_train) 
    val = data_utils.TensorDataset(X_valid, y_val)
    test = data_utils.TensorDataset(X_test, y_test) 

    return train, val, test


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
        self.metric_min = -np.Inf
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
        elif score <= self.best_score + self.delta:
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
            self.trace_func(f'Validation metric increased ({self.metric_min:.6f} --> {metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.metric_min = metric
