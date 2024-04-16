# Imports
import torch
import pandas as pd
import numpy as np
import random
import warnings
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
# Utils variables and functions
from data.utils_preprocessing import COHORTS, TISSUES, CANCER_TYPES, process_all, process_covariates, process_survival, train_test_split, df_to_data, categorical_labels, one_hot_to_categoricals, standardize, pca_reduction, standardize_train, standardize_test
from sklearn.preprocessing import OneHotEncoder

# Ignore pandas warnings
warnings.filterwarnings('ignore') 

# SEED
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Dataloader reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() # should be 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)


# Loader class
class TCGALoader(object):
    """
    Class to build full TCGA dataset and load training data.
    ----
    """
    def __init__(self):
        # Init
        self.PATH = '/data/tcga_files/data_RNAseq_RTCGA'
        self.PATHFOLDER = '/gdac.broadinstitute.org_{}.Mer'
        self.PATHFILE = '/{}.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt'
        self.PATH_CLINICALFOLDER = '/clinical/gdac.broadinstitute.org_{}.Merge_Clinical.Level_1.2016012800.0.0'
        self.PATH_CLINICALFILE = '/{}.clin.merged.txt'
        # List of cohorts, cancer types and tissues
        self.COHORTS = COHORTS
        self.CANCER_TYPES = CANCER_TYPES
        self.TISSUES = TISSUES

        # Full dictionary of cancer types
        full_vocab = [] # Flatten list of all tissue types and create vocab dictionary
        for c in self.CANCER_TYPES:
            full_vocab.extend(c.split())
        vocab = Counter(full_vocab) # create a dictionary
        vocab = sorted(vocab, key=vocab.get, reverse=True)
        self.full_vocab_cancer_types_size = len(vocab)
        # map words to unique indices
        self.word2idx_cancer_types = {word: ind for ind, word in zip(range(1, self.full_vocab_cancer_types_size+1), vocab)}
        self.word2idx_cancer_types['none'] = 0 # For zero padding

        #  Full dictionary of tissue types
        full_vocab = []
        for c in self.TISSUES:
            full_vocab.extend(c.split())
        vocab_tissue_types = Counter(full_vocab) # create a dictionary
        vocab_tissue_types = sorted(vocab_tissue_types, key=vocab_tissue_types.get, reverse=True)
        self.vocab_tissue_types_size = len(vocab_tissue_types)
        # Map words to unique indices
        self.word2idx_tissue_types = {word: ind for ind, word in enumerate(vocab_tissue_types)}

        # Columns
        self.BARCODE_COL = 'patient.bcr_patient_barcode' # Patients already reduced barcodes
        self.AGE_COL = 'patient.age_at_initial_pathologic_diagnosis' # Age at diagnosis
        self.AGE_COL2 = 'patient.days_to_birth' # Other variable of age if previous one is not in dataframe
        self.GENDER_COL = 'patient.gender' # Gender
        self.RACE_COL = 'patient.race_list.race' # Race (can also add ethnicity)
        self.FOLLOWUP_COL = 'patient.days_to_last_followup' # Number of days between the diagnosis date and the date of the patient's last follow-up appointment or contact.
        self.STATUS_COL = 'patient.vital_status' # Patient status: alive or dead
        self.DEATH_COL ='patient.days_to_death' # Number of days between diagnosis and death if event occurred
        self.CANCER_STAGE_PATH = 'patient.stage_event.pathologic_stage' # Cancer stage if available for cohort
        self.CANCER_STAGE_CLIN = 'patient.stage_event.clinical_stage' # Cancer stage (clinical) if available for cohort. Similar to pathological stage.
        # Columns list
        self.CLINICAL_COLUMNS = [self.BARCODE_COL, self.AGE_COL, self.AGE_COL2, self.GENDER_COL, self.RACE_COL, self.FOLLOWUP_COL, self.STATUS_COL, self.DEATH_COL,
                                self.CANCER_STAGE_PATH, self.CANCER_STAGE_CLIN]
        # Clinical data labels
        self.LABELS = ['barcode', 'age1', 'age2', 'gender', 'race', 'followup', 'status', 'death', 'cancer_stage_pathological', 'cancer_stage_clinical']
        # Build a dictionary for easier fetching
        self.dict_columns = {i :j for i,j in zip(self.LABELS, self.CLINICAL_COLUMNS)}
        # Standardization parameters
        self.df_params = None


    def load_clinical_data(self, cohort:str):
        """
        Load clinical data of interest for a given cohort.
        ----
        Parameters:
            cohort (str): cancer type from which we want to retrieve the data
        Returns:
            DF (pd.DataFrame): survival data of interest
        """

        try:
            df_temp = pd.read_csv(self.PATH+self.PATH_CLINICALFOLDER.format(cohort)+self.PATH_CLINICALFILE.format(cohort), sep='\t', header=None)
        except FileNotFoundError:
            df_temp = pd.read_csv(self.PATH+self.PATH_CLINICALFOLDER.format(cohort)+'g'+self.PATH_CLINICALFILE.format(cohort), sep='\t', header=None)

        DF = df_temp.transpose()
        NEW_COLS = list(DF.loc[0].values)
        DF.rename(columns={i: j for i,j in zip(DF.columns, NEW_COLS)}, inplace=True)
        DF.drop(0, inplace=True)

        return DF


    def fetch(self, label:str='age', cohort:str='all'):
        """
        Returns specific clinical data for all patients in one cohort type or all patients in all cohorts.
        ----
        Parameters:
            label (str): data to fetch (either 'age', 'cancer_stage')
            cohort (str): cohort for which to retrieve the data
        Returns:
            df (pd.DataFrame): dataframe with patients barcodes, cohorts and data fetched

        """
        # Assert correct label was selected
        POSSIBLE_LABELS = list(set(self.LABELS) - set(['age1', 'age2', 'cancer_stage_pathological', 'cancer_stage_clinical'])) + ['age', 'cancer_stage']
        assert label in POSSIBLE_LABELS, f"'{label} was not found. To correctly fetch data, please select a label argument in the following list: {POSSIBLE_LABELS}.'"

        if cohort=='all':
            # Build first dataframe
            df = self.load_clinical_data(self.COHORTS[0]) # Load full dataset
            if label=='age':
                df = self.add_age(df)
                df = df[[self.dict_columns['barcode'], label]]
                df.rename(columns={self.dict_columns['barcode']: 'barcode'}, inplace = True)

            elif label =='cancer_stage':
                df = self.add_cancer_stage(df)
                df = df[[self.dict_columns['barcode'], label]]
                df.rename(columns={self.dict_columns['barcode']: 'barcode'}, inplace = True)

            else:
                df = df[[self.dict_columns['barcode'], self.dict_columns[label]]] # Keep only the corresponding column
                df.rename(columns={self.dict_columns['barcode']: 'barcode', self.dict_columns[label]: label}, inplace=True) # Rename column with short label

            df['cohort'] = [self.COHORTS[0]]*len(df) # Add a column to keep track of cohorts

            for c in self.COHORTS[1:]:
                df_temp = self.load_clinical_data(c)
                if label=='age':
                    df_temp = self.add_age(df_temp)
                    df_temp = df_temp[[self.dict_columns['barcode'], label]]
                    df_temp.rename(columns={self.dict_columns['barcode']: 'barcode'}, inplace = True)
                elif label =='cancer_stage':
                    df_temp = self.add_cancer_stage(df_temp)
                    df_temp = df_temp[[self.dict_columns['barcode'], label]]
                    df_temp.rename(columns={self.dict_columns['barcode']: 'barcode'}, inplace = True)
                else:
                    df_temp = df_temp[[self.dict_columns['barcode'], self.dict_columns[label]]]
                    df_temp.rename(columns={self.dict_columns['barcode']: 'barcode', self.dict_columns[label]: label}, inplace=True)
                df_temp['cohort'] = [c]*len(df_temp)
                df= pd.concat([df, df_temp], ignore_index=True)

        else:
            assert cohort in self.COHORTS, f"'{cohort}' not found. Please select a 'cohort' in the following list: {self.COHORTS}."
            df = self.load_clinical_data(cohort)
            if label=='age':
                df = self.add_age(df)
                df = df[[self.dict_columns['barcode'], label]]
                df.rename(columns={self.dict_columns['barcode']: 'barcode'}, inplace = True)
            elif label =='cancer_stage':
                df = self.add_cancer_stage(df)
                df = df[[self.dict_columns['barcode'], label]]
                df.rename(columns={self.dict_columns['barcode']: 'barcode'}, inplace = True)
            else:
                df = df[[self.dict_columns['barcode'], self.dict_columns[label]]]
                df.rename(columns={self.dict_columns['barcode']: 'barcode', self.dict_columns[label]: label}, inplace=True)

        return df


    def add_cancer_stage(self, df:pd.DataFrame):
        """
        Returns a dataframe with the cancer stage data with less missing data if any. If cancer_stage column is not found, returns column of NaNs.
        ---- 
        Parameters:
            df (pd.DataFrame): dataframe on which cancer stage column will be added
        Returns:
            df (pd.DataFrame): dataframe with additional column
        """

        if self.dict_columns['cancer_stage_pathological'] in list(df.columns.values):
            # Check which column has the less NaNs
            col_to_keep = np.argmin([df[self.dict_columns['cancer_stage_pathological']].isna().sum(),
                                        df[self.dict_columns['cancer_stage_clinical']].isna().sum()])
            cancer_stage_cols = [self.dict_columns['cancer_stage_pathological'], self.dict_columns['cancer_stage_clinical']]
            # Rename column with short labels
            df.rename(columns={cancer_stage_cols[col_to_keep]: 'cancer_stage'}, inplace=True)

        else:
            # Add column of Nans
            df['cancer_stage'] = np.array([np.nan]*len(df))

        return df


    def add_age(self, df:pd.DataFrame):
        """
        Returns a dataframe with age column with the less missing data if any.
        ----
        Parameters:
            df (pd.DataFrame): dataframe on which age column will be added
        Returns:
            df (pd.DataFrame): dataframe with additional column
        """
        convert_age = lambda x: (-1)*x/365 # Function to convert 'days to birth' into age

        if self.dict_columns['age1'] in list(df.columns.values):
            # Check which column has the less NaNs. If same number of missing values, argmin returns the first index.
            col_to_keep = np.argmin([df[self.dict_columns['age1']].isna().sum(),
                                        df[self.dict_columns['age2']].isna().sum()])
            cancer_stage_cols = [self.dict_columns['age1'], self.dict_columns['age2']]
            # Rename column with short labels
            df.rename(columns={cancer_stage_cols[col_to_keep]: 'age'}, inplace=True)
            df['age'] = df['age'].astype(float) # Convert to float for compatibility

            # 'days to birth' was kept, convert to age
            if col_to_keep == 2:
                df['age'] = convert_age(df['age'])

        else:
            # Rename column with short labels
            df.rename(columns={self.dict_columns['age2']: 'age'}, inplace=True)
            df['age'] = df['age'].astype(float) # Convert to float for compatibility
            df['age'] = convert_age(df['age']) # Convert negative days to age

        return df


    def fetch_all(self, cohort:str):
        """
        Returns a dataframe with all clinical information of interest for given cohort.
        ----
        Parameters:
            cohort (str): cohort for which the clinical data is needed. If 'all', then all clinical data for all cohorts is returned.
        Returns:
            df (pd.DataFrame): dataframe with patients barcodes, gender, age, race, followup days, cancer stage, death status and event
        """
        COLS_OF_INTEREST = [self.BARCODE_COL, self.GENDER_COL, self.RACE_COL, self.FOLLOWUP_COL, self.STATUS_COL, self.DEATH_COL]
        LABELS_OF_INTEREST = ['barcode', 'gender', 'race', 'followup', 'status', 'death']

        if cohort=='all':
            # Build first dataframe
            df = self.load_clinical_data(self.COHORTS[0]) # Load full dataframe
            df.rename(columns={i: j for i,j in zip(COLS_OF_INTEREST, LABELS_OF_INTEREST)}, inplace=True) # Rename column with short labels
            df = self.add_age(df) # Add age
            df = self.add_cancer_stage(df) # Add cancer stage column
            df = df[LABELS_OF_INTEREST+['age', 'cancer_stage']] # Keep only the columns of interest
            df['cohort'] = [self.COHORTS[0]]*len(df) # Add a column to keep track of cohorts

            for c in self.COHORTS[1:]:
                df_temp = self.load_clinical_data(c)
                # Rename column with short labels
                df_temp.rename(columns={i: j for i,j in zip(COLS_OF_INTEREST, LABELS_OF_INTEREST)}, inplace=True)
                df_temp = self.add_age(df_temp)
                df_temp = self.add_cancer_stage(df_temp)
                df_temp = df_temp[LABELS_OF_INTEREST+['age', 'cancer_stage']] # Keep only the columns of interest
                df_temp['cohort'] = [c]*len(df_temp)
                df = pd.concat([df, df_temp], ignore_index=True)

        else:
            df = self.load_clinical_data(cohort) # Load full dataframe
            df.rename(columns={i: j for i,j in zip(COLS_OF_INTEREST, LABELS_OF_INTEREST)}, inplace=True) # Rename column with short labels
            df = self.add_age(df) # Add age
            df = self.add_cancer_stage(df) # Add cancer stage data
            df = df[LABELS_OF_INTEREST+['age', 'cancer_stage']] # Keep only the columns of interest

        return df


    def fetch_covariates(self, cohort:str):
        """
        Returns a dataframe of covariates used in the GAN training.
        ----
        Parameters:
            cohort (str): cohort for which to fetch the data from, if 'all' is selected then data for all cohorts is returned
        Returns:
            df (pd.DataFrame): dataframe with patients barcodes and covariates columns

        """
        if cohort=='all':
            # Build first dataframe
            df = self.load_clinical_data(self.COHORTS[0]) # Load full dataframe
            df = self.add_age(df) # Add age
            df = df[[self.dict_columns['barcode'], 'age', self.dict_columns['gender']]] # Keep only covariates of interest
            df.rename(columns={self.dict_columns['barcode']: 'barcode', self.dict_columns['gender']: 'gender'}, inplace=True) # Rename column with short labels
            df['cohort'] = [self.COHORTS[0]]*len(df) # Add a column to keep track of cohorts

            for c in self.COHORTS[1:]:
                df_temp = self.load_clinical_data(c) # Load full dataframe
                df_temp = self.add_age(df_temp) # Add age
                df_temp = df_temp[[self.dict_columns['barcode'], 'age', self.dict_columns['gender']]] # Keep only covariates of interest
                df_temp.rename(columns={self.dict_columns['barcode']: 'barcode', self.dict_columns['gender']: 'gender'}, inplace=True) # Rename column with short labels
                df_temp['cohort'] = [c]*len(df_temp) # Add a column to keep track of cohorts
                df = pd.concat([df, df_temp], ignore_index=True)

        else:
            # Load full dataframe
            df = self.load_clinical_data(cohort)
            df = self.add_age(df)
            df = df[[self.dict_columns['barcode'], 'age', self.dict_columns['gender']]]
            df.rename(columns={self.dict_columns['barcode']: 'barcode', self.dict_columns['gender']: 'gender'}, inplace=True)

        return df


    def build_full_dataframe(self, path:str):
        """
        Returns full dataframe with all gene expression data.
        ----
        Parameters:
            path (str): path where preprocessed data has been stored.
        Returns:
            df (pd.DataFrame): dataframe with patients barcodes, gene expression data, cancer types and cancer labels (normal or cancer)
        """
        # Load data
        patient_ids = np.load(path+'/TCGA_rnaseq_RSEM_patients_ids_full.npy', allow_pickle=True)
        cancer_types = np.load(path+'/TCGA_rnaseq_RSEM_cancer_types_full.npy', allow_pickle=True)
        exp_data = np.load(path+'/TCGA_rnaseq_RSEM_preprocess_full.npy', allow_pickle=True)
        gene_ids = np.load(path+'/TCGA_rnaseq_RSEM_gene_ids_full.npy', allow_pickle=True)
        labels = np.load(path+'/TCGA_rnaseq_RSEM_labels_full.npy', allow_pickle=True)

        # Get all clinical data
        df_all = self.fetch_all(cohort='all')
        df_exp = pd.DataFrame(data=exp_data, columns= gene_ids)
        df_exp.insert(0, 'full_barcode' , patient_ids)
        df_exp.insert(1, 'cohort' , cancer_types)
        df_exp.insert(2, 'cancer' , labels)

        # Extract reduced barcode as in clinical data
        reduced_ids_cancer = np.asarray([(patient_ids[i].split('-', 3)[0]+'-'+patient_ids[i].split('-', 3)[1]+'-'+patient_ids[i].split('-', 3)[2]).lower() for i in range(len(patient_ids))])
        df_exp.insert(1, 'reduced_barcode' , reduced_ids_cancer)
        # Merge both datasets by keeping only samples for which we have both expression data and clinical data
        df_all.drop('cohort', axis=1, inplace=True)
        df = pd.merge(df_exp, df_all, how='inner', left_on='reduced_barcode', right_on='barcode')

        # Add cancer types and tissues
        dict_cancer_types = {i : j for i,j in zip(self.COHORTS, self.CANCER_TYPES)}
        df.insert(1, 'cancer_type' ,[dict_cancer_types[df['cohort'].loc[i]] for i in range(len(df['cohort']))])
        dict_tissues_types = {i : j for i,j in zip(self.COHORTS, self.TISSUES)}
        df.insert(1, 'tissue_type' , [dict_tissues_types[df['cohort'].loc[i]] for i in range(len(df['cohort']))])
        # Add 'normal' category to cancer types where label = 0
        df['cancer_type'].loc[df[df['cancer']==0].index.values] = 'normal'

        return df

    def encode_cancer_type(self, cancer_type:str):
        """
        Returns encoded and padded cancer type (size 6,)
        cancer_type (str): cancer type of max size (6,)
        """
        # Split cancer type
        x = cancer_type.split()
        # Encode with vocabulary dict (=vectorize)
        encoded = torch.LongTensor([self.word2idx_cancer_types[word] for word in x])
        # Zero-padding to max length (6)
        pad = (0, 6-len(encoded))
        encoded = torch.nn.functional.pad(encoded, pad, "constant", 0)

        return encoded

    def one_hot_encoder(self, on='tissues'):
        """
        Init 1-hot encoder for the given data type and fit the encoder on the list of data type (either tissues or cancer).
        ----
        Parameters:
            on (str): on what data to encode (either tissues or cancer)
        """
        if on=='tissues':
            self.Tissue_Encoder = OneHotEncoder(handle_unknown='ignore') # Init encoder
            self.Tissue_Encoder.fit(np.unique(self.TISSUES).reshape(-1,1))
        elif on=='cancer':
            self.Cancer_Encoder = OneHotEncoder(handle_unknown='ignore') # Init encoder
            self.Cancer_Encoder.fit(np.unique(self.CANCER_TYPES).reshape(-1,1))


    def one_hot_encode_tissue_type(self, tissue_types:str):
        """
        Perform one-hot encoding on input tissue types.
        ----
        Parameters:
            tissue_types (np.array): tissue types to encode
        Returns:
            (np.array): tissue types one-hot encoded
        """
        # Init encoder
        self.one_hot_encoder(on='tissues') 
        # Encode tissue types
        return self.Tissue_Encoder.transform(tissue_types.reshape(-1,1)).toarray()

    def one_hot_encode_cancer_type(self, cancer_types):
        """
        Perform one-hot encoding on input cancer types.
        ----
        Parameters:
            cancer_types (np.array): cancer types to encode
        Returns:
            (np.array): cancer types one-hot encoded
        """
        # Init encoder
        self.one_hot_encoder(on='cancer')
        # Encode cancer types
        return self.Cancer_Encoder.transform(cancer_types.reshape(-1,1)).toarray()


    def encode_tissue_type(self, tissue_type:str):
        """
        Returns encoded tissue type as vectorize words (size 1)
        ----
        Parameters:
            tissue_type (str): tissue type to encode
        Returns:
            encoded (torch.tensor): tensor of tissue type encoded as vector
        """
        # Split tissue type
        x = tissue_type.split()
        # Encode with vocabulary dict (=vectorize)
        encoded = torch.LongTensor([self.word2idx_tissue_types[word] for word in x])
        return encoded


    def process_all_data(self, path:str):
        """
        Returns processed dataframe with all gene expression data and clinical data of interest for all cohorts.
        ----
        Parameters:
            path (str): path where full raw dataframe is located
        Returs:
            df (pd.DataFrame): preprocessed dataframe
        """
        # Get full dataframe
        df = pd.read_csv(path)
        # Process all variables
        df = process_all(df)

        return df


    def process_all_covariates(self, path:str):
        """
        Processing clinical data used in GAN training: age, gender, tissue type, cancer label and cancer type.
        ----
        Parameters:
            path (str): path where full raw dataframe is located
        Returs:
            df (pd.DataFrame): preprocessed dataframe of covariates
        """
        # Get full dataframe
        df = pd.read_csv(path)
        # Process covariates
        df = process_covariates(df)

        return df

    def process_all_survival(self, path:str):
        """
        Processing clinical data used for survival tasks: survival time, status, death event.
        ----
        Parameters:
            path (str): path where full raw dataframe is located
        Returs:
            df (pd.DataFrame): preprocessed dataframe of survival data
        """
        # Get full dataframe
        df = pd.read_csv(path)
        # Process covariates
        df = process_survival(df)

        return df

    def train_data(self, task:str='covariates'):
        """
        Returns train data given the training task.
        ----
        Parameters:
            task (str): training task (either 'covariates', 'survival' or 'all')
        Returns:
            df_train (pd.DataFrame): train dataframe of given task
        """
        # Get train data
        df_train = pd.read_csv(self.PATH+f'/train_df_{task}.csv')

        return df_train


    def training_splits(self, task:str='covariates', shuffle:bool=True, split_ratios:list=[0.8, 0.2]):
        """
        Performs a train-test split given split ratios, standardize the dataframes, save standardization parameters and returns the train and test dataframes.
        ----
        Parameters:
            task (str): training task (either 'covariates', 'survival' or 'all')
            shuffle (bool): whether to randomly shuffle data for the train-test split (default True)
            split_ratios (list): train-test split ratios (default [0.8, 0.2])
        Returns:
            df_train (pd.DataFrame), df_val (pd.DataFrame): standardized dataframes resulting from a train-test split
        """
        # Get training dataframe
        df = self.train_data(task)

        # Train-validation split
        df_train, df_val = train_test_split(df, split_ratios=split_ratios, shuffle=shuffle)

        return df_train, df_val


    def save_standardization_params(self, path:str=None, print_params_location=True):
        """
        Saves previously computed standardization params at given path or default path.
        Gene expression standardization parameters are saved in numpy.array files; clinical variables parameters are saved in a .csv file.
        ----
        Parameters:
            path (str): root folder path where to store all parameters files
            print_params_location (bool): whether to print paths where parameters are saved (default True)
        """
        # Paths where to store parameters
        if path is None:
            path = self.PATH
        path_mean_exp = path+'/standardization_params_mean_exp.npy'
        path_std_exp = path+'/standardization_params_std_exp.npy'
        path_params = path+'/standardization_df_params.csv'

        # Save
        self.df_params.to_csv(path_params, index=False)
        np.save(path_mean_exp, self.mean_exp)
        np.save(path_std_exp, self.std_exp)

        # Display files location
        if print_params_location:
            print(f"Gene expression mean parameters were saved at {path_mean_exp}.")
            print(f"Gene expression std parameters were saved at {path_std_exp}.")
            print(f"Clinical data parameters were saved at {path_params}.")


    def training_loaders(self, batch_size:int, task:str='covariates', shuffle:bool=True, subset:int=None, standardize:bool=True, split_ratios=[0.8, 0.2]):
        """
        Returns train data and validation pytorch data loaders given the training task and batch size.
        ----
        Parameters:
            batch_size (int): size of loaded batches
            task (str): training task (either 'covariates', 'survival' or 'all')
            shuffle (bool): whether to shuffle training data in train loader (default True). This option is set to False for validation loader.
        Returns:
            train_dataloader (torch.DataLoader), val_dataloader (torch.DataLoader): train and validation loaders
        """
        # Get train-validation splits dataframes
        df_train, df_val = self.training_splits(task, split_ratios=split_ratios)
        # Batch size
        self.batch_size=batch_size

        # Get data and build loaders
        if task=='all':
            x_exp, age, gender, cancer_type, tissue_type, time, status, cancer_stage, cohort = df_to_data(df_train, as_type='tensor', task=task)
            x_exp_val, age_val, gender_val, cancer_type_val, tissue_type_val, time_val, status_val, cancer_stage_val, cohort_val = df_to_data(df_val, as_type='tensor', task=task)
            # Encode cancer types and tissue types
            encoded_cancer_types = torch.from_numpy(self.one_hot_encode_cancer_type(cancer_type)).long()
            encoded_tissue_types = torch.from_numpy(self.one_hot_encode_tissue_type(tissue_type)).long()
            encoded_cancer_types_val = torch.from_numpy(self.one_hot_encode_cancer_type(cancer_type)).long()
            encoded_tissue_types_val = torch.from_numpy(self.one_hot_encode_tissue_type(tissue_type_val)).long()

            # Standardize
            if standardize:
                x_exp, self.scaler_x_exp_ = standardize_train(x_exp, as_tensors=True)
                age, self.scaler_age_ = standardize_train(age.reshape(-1, 1), as_tensors=True)
                x_exp_val, _ = standardize_test(x_exp_val, self.scaler_x_exp_, as_tensors=True)
                age_val, _ = standardize_test(age_val.reshape(-1, 1), self.scaler_age_, as_tensors=True)

            if subset is not None:
                # Only subset of genes
                x_exp = x_exp[:, :subset]
                x_exp_val = x_exp_val[:, :subset]
                print("New training set shape:", x_exp.shape)

            # Build loaders
            train_dataloader = DataLoader(TensorDataset(x_exp, age, gender, encoded_cancer_types, encoded_tissue_types, time, status, cancer_stage), batch_size=batch_size, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)
            val_dataloader = DataLoader(TensorDataset(x_exp_val, age_val, gender_val, encoded_cancer_types_val, encoded_tissue_types_val, time_val, status_val, cancer_stage_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        
        elif task=='covariates':
            x_exp, age, gender, cancer_type, tissue_type, labels, cohort = df_to_data(df_train, as_type='tensor', task=task)
            x_exp_val, age_val, gender_val, cancer_type_val, tissue_type_val, labels_val, cohort_val = df_to_data(df_val, as_type='tensor', task=task)

            # Encode cancer types and tissue types
            encoded_cancer_types = torch.from_numpy(self.one_hot_encode_cancer_type(cancer_type)).long()
            encoded_tissue_types = torch.from_numpy(self.one_hot_encode_tissue_type(tissue_type)).long()
            encoded_cancer_types_val = torch.from_numpy(self.one_hot_encode_cancer_type(cancer_type_val)).long()
            encoded_tissue_types_val = torch.from_numpy(self.one_hot_encode_tissue_type(tissue_type_val)).long()

            # Standardize
            if standardize:
                x_exp, self.scaler_x_exp_ = standardize_train(x_exp, as_tensors=True)
                age, self.scaler_age_ = standardize_train(age.reshape(-1, 1), as_tensors=True)
                x_exp_val, _ = standardize_test(x_exp_val, self.scaler_x_exp_, as_tensors=True)
                age_val, _ = standardize_test(age_val.reshape(-1, 1), self.scaler_age_, as_tensors=True)

            if subset is not None:
                # Only subset of genes
                x_exp = x_exp[:, :subset]
                x_exp_val = x_exp_val[:, :subset]

                print("New training set shape:", x_exp.shape)

            # Build loaders
            train_dataloader = DataLoader(TensorDataset(x_exp, age, gender, encoded_cancer_types, encoded_tissue_types, labels), batch_size=batch_size, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)
            val_dataloader = DataLoader(TensorDataset(x_exp_val, age_val, gender_val, encoded_cancer_types_val, encoded_tissue_types_val, labels_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        
        elif task=='survival':
            x_exp, age, gender, cancer_type, tissue_type, time, status, cancer_stage, cohort = df_to_data(df_train, as_type='tensor', task=task)
            x_exp_val, age_val, gender_val, cancer_type_val, tissue_type_val, time_val, status_val, cancer_stage_val, cohort_val = df_to_data(df_val, as_type='tensor', task=task)

            # Encode cancer types and tissue types
            encoded_cancer_types = torch.from_numpy(self.one_hot_encode_cancer_type(cancer_type)).long()
            encoded_tissue_types = torch.from_numpy(self.one_hot_encode_tissue_type(tissue_type)).long()
            encoded_cancer_types_val = torch.from_numpy(self.one_hot_encode_cancer_type(cancer_type_val)).long()
            encoded_tissue_types_val = torch.from_numpy(self.one_hot_encode_tissue_type(tissue_type_val)).long()
            
            # Standardize
            if standardize:
                # Fit
                x_exp, self.scaler_x_exp_ = standardize_train(x_exp, as_tensors=True)
                age, self.scaler_age_ = standardize_train(age.reshape(-1, 1), as_tensors=True)
                cancer_stage, self.scaler_stage_ = standardize_train(cancer_stage.reshape(-1, 1), as_tensors=True)
                time, self.scaler_time_ = standardize_train(time.reshape(-1, 1), as_tensors=True)
                # Transform
                x_exp_val, _ = standardize_test(x_exp_val, self.scaler_x_exp_, as_tensors=True)
                age_val, _ = standardize_test(age_val.reshape(-1, 1), self.scaler_age_, as_tensors=True)
                cancer_stage_val, _ = standardize_test(cancer_stage_val.reshape(-1, 1), self.scaler_stage_, as_tensors=True)
                time_val, _ = standardize_test(time_val.reshape(-1, 1), self.scaler_time_, as_tensors=True)
            
            if subset is not None:
                # Only subset of genes
                x_exp = x_exp[:, :subset]
                x_exp_val = x_exp_val[:, :subset]
                print("New training set shape:", x_exp.shape)

            
            train_dataloader = DataLoader(TensorDataset(x_exp, age, gender, encoded_cancer_types, encoded_tissue_types, time, status, cancer_stage), batch_size=batch_size, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)
            val_dataloader = DataLoader(TensorDataset(x_exp_val, age_val, gender_val, encoded_cancer_types_val, encoded_tissue_types_val, time_val, status_val, cancer_stage_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

        return train_dataloader, val_dataloader


    def test_standardize(self, task:str='covariates'):
        """
        Returns test dataframe with current standardization parameters.
        ----
        Parameters:
            task (str): training task (either 'covariates', 'survival' or 'all')
        Returns:
            df_test (pd.DataFrame): standardized test dataframe
        """
        assert self.df_params is not None, "Standardization parameters have to be previously computed. Please call 'training_splits' function before."

        # Get test data
        df_test = pd.read_csv(self.PATH+f'/test_df_{task}.csv')

        # Standardize clinical data
        if task=='covariates':
            mean = self.df_params[f'age'][0]
            std = self.df_params[f'age'][1]
            df_test[f'age'] = df_test[f'age'].apply(lambda x: standardize(x, mean, std))

        elif task=='survival':
            mean = self.df_params[f'followup'][0]
            std = self.df_params[f'followup'][1]
            df_test[f'followup'] = df_test[f'followup'].apply(lambda x: standardize(x, mean, std))

        elif task=='all':
            for col in ['age', 'followup']:
                mean_temp = self.df_params[f'{col}'][0]
                std_temp = self.df_params[f'{col}'][1]
                df_test[col]= df_test[col].apply(lambda x: standardize(x, mean_temp, std_temp))

        # Get gene ids
        GENE_IDS = np.load(self.PATH+'/TCGA_rnaseq_RSEM_gene_ids_full.npy', allow_pickle=True)

        # Standardize gene expression data
        mean_exp = self.mean_exp
        std_exp = self.std_exp
        df_test[list(GENE_IDS)] = (df_test[list(GENE_IDS)].to_numpy() - mean_exp)/std_exp

        # Put Nans to 0 if any (because of division by zero if std is null)
        df_test[list(GENE_IDS)].fillna(0, inplace=True)

        return df_test


    def test_loader(self, batch_size:int, task:str='covariates', shuffle:bool=False, standardize:bool=True, subset:int=None):
        """
        Returns test data loader given the training task and batch size.
        ----
        Parameters:
            batch_size (int): size of loaded batches
            task (str): training task
            shuffle (bool): whether to shuffle data in test loader (default False)
            standardize (bool): whether to standardize the test data with training parameters (default True)
        Returns:
            test_loader (torch.DataLoader): test data loader
        """
        # Get test data
        df_test = pd.read_csv(self.PATH+f'/test_df_{task}.csv')

        #Get data and build loaders
        if task=='all':
            x_exp, age, gender, cancer_type, tissue_type, time, status, cancer_stage, cohort = df_to_data(df_test, as_type='tensor', task=task)
            # Encode cancer types and tissue types
            encoded_cancer_types = torch.from_numpy(self.one_hot_encode_cancer_type(cancer_type)).long()
            encoded_tissue_types = torch.from_numpy(self.one_hot_encode_tissue_type(tissue_type)).long()

            # Standardize
            if standardize:
                x_exp, _ = standardize_test(x_exp, self.scaler_x_exp_, as_tensors=True)
                age, _ = standardize_test(age.reshape(-1, 1), self.scaler_age_, as_tensors=True)

            if subset is not None:
                # Only subset of genes
                x_exp = x_exp[:, :subset]
                print("New training set shape:", x_exp.shape)
            # Loader
            test_loader = DataLoader(TensorDataset(x_exp, age, gender,  encoded_cancer_types, encoded_tissue_types, time, status, cancer_stage), batch_size=batch_size, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)
        
        elif task=='covariates':
            x_exp, age, gender, cancer_type, tissue_type, labels, cohort = df_to_data(df_test, as_type='tensor', task=task)
            # Encode cancer types and tissue types
            encoded_cancer_types = torch.from_numpy(self.one_hot_encode_cancer_type(cancer_type)).long()
            encoded_tissue_types = torch.from_numpy(self.one_hot_encode_tissue_type(tissue_type)).long()

            # Standardize
            if standardize:
                x_exp, _ = standardize_test(x_exp, self.scaler_x_exp_, as_tensors=True)
                age, _ = standardize_test(age.reshape(-1, 1), self.scaler_age_, as_tensors=True)

            if subset is not None:
                # Only subset of genes
                x_exp = x_exp[:, :subset]
                print("New training set shape:", x_exp.shape)
            # Loader
            test_loader = DataLoader(TensorDataset(x_exp, age, gender, encoded_cancer_types, encoded_tissue_types, labels), batch_size=batch_size, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)
        
        elif task=='survival':
            x_exp, age, gender, cancer_type, tissue_type, time, status, cancer_stage, cohort = df_to_data(df_test, as_type='tensor', task=task)
            
            # Encode cancer types and tissue types
            encoded_cancer_types = torch.from_numpy(self.one_hot_encode_cancer_type(cancer_type)).long()
            encoded_tissue_types = torch.from_numpy(self.one_hot_encode_tissue_type(tissue_type)).long()

            # Standardize
            if standardize:
                x_exp, _ = standardize_test(x_exp, self.scaler_x_exp_, as_tensors=True)
                age, _ = standardize_test(age.reshape(-1, 1), self.scaler_age_, as_tensors=True)
                cancer_stage, _ = standardize_test(cancer_stage.reshape(-1, 1), self.scaler_stage_, as_tensors=True)
                time, _ = standardize_test(time.reshape(-1, 1), self.scaler_time_, as_tensors=True)

            if subset is not None:
                # Only subset of genes
                x_exp = x_exp[:, :subset]
                print("New training set shape:", x_exp.shape)
            # Loader
            test_loader = DataLoader(TensorDataset(x_exp, age, gender, encoded_cancer_types, encoded_tissue_types, time, status, cancer_stage), batch_size=batch_size, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)

        return test_loader


    def test_data(self, task:str='covariates', as_type:str='tensor'):
        """
        Returns test data given the training task.
        ----
        Parameters:
            task (str): training task
            as_type (str): data type as tensors or arrays
        Returns:
            torch.tensors of gene expression data and clinical data given the training task
        """
        # Get test data
        df_test = pd.read_csv(self.PATH+f'/test_df_{task}.csv')

        #Get data and build loaders
        if task=='all':
            x_exp, age, gender, cancer_type, tissue_type, time, status, cancer_stage, cohort = df_to_data(df_test, as_type=as_type, task=task)
            return x_exp, age, gender, cancer_type, tissue_type, time, status, cancer_stage
        elif task=='covariates':
            x_exp, age, gender,cancer_type, tissue_type, labels, cohort = df_to_data(df_test, as_type=as_type, task=task)
            return x_exp, age, gender,cancer_type, tissue_type, labels
        elif task=='survival':
            x_exp, age, gender, cancer_type, tissue_type, time, status, cancer_stage, cohort = df_to_data(df_test, as_type=as_type, task=task)
            return x_exp, age, gender, cancer_type, tissue_type, time, status, cancer_stage


    def fetch_real_train_test_data(self, apply_pca:bool=False, subset:int=None, standardize_train_set:bool=True, standardize_test_set:bool=False, standardize_covariates:bool=True):
        """
        Fetch real training and testing data and returns standardized gene expression data with covariates.
        ----
        Parameters:
            apply_pca (bool): whether to apply PCA reduction on TCGA data. Default False.
            subset (int): number of genes in the subset of total genes to consider. If 'None', all the genes are kept. Default None.
            standardize_train_set (bool): whether to standardize to 0 mean and 1 of variance the training data. Default True.
            standardize_test_set (bool): whether to standardize the test data with train dtaa mean and standard deviation. Default False.
            standardize_covariates (bool): whether to standardize the training numerical conditional covariates (e.g age) to 0 mean and variance of 1.
        Returns:
            Two tuples of training gene expression data with covariates and testing data with covariates.
            
            x_exp (torch.tensor): gene expression data
            age (torch.tensor): age
            gender (torch.tensor): gender
            cancer_type (torch.tensor): cancer type
            encoded_tissue_types (torch.tensor): encoded tissue types
            labels (torch.tensor): cancer labels
        """
        # Get as much covariates as possible from training data
        df_train = self.train_data(task='covariates')
        # Get largest test data
        df_test = pd.read_csv(self.PATH+f'/test_df_covariates.csv')
        #df_train, df_test, df_params, mean_exp, std_exp = standardize_split(df, df_test, task='covariates')
        x_exp, age, gender, cancer_type, tissue_type, labels, cohort = df_to_data(df_train, as_type='tensor', task='covariates')
        x_exp_val, age_val, gender_val, cancer_type_val, tissue_type_val, labels_val, cohort_val = df_to_data(df_test, as_type='tensor', task='covariates')

        # Standardize covs
        if standardize_covariates:
            age, self.scaler_age_ = standardize_train(age.reshape(-1, 1) , as_tensors=True)
            age_val, _ = standardize_test(age_val.reshape(-1, 1), self.scaler_age_, as_tensors=True)

        # Standardize
        if standardize_train_set:
            x_exp, self.scaler_x_exp_ = standardize_train(x_exp, as_tensors=True)

            if standardize_test_set:
                x_exp_val, _ = standardize_test(x_exp_val, self.scaler_x_exp_, as_tensors=True)
                
        if subset is not None:
            # Only subset of genes
            x_exp = x_exp[:, :subset]
            x_exp_val = x_exp_val[:, :subset]
            print(f"New dataset shape on subset of size {subset}:", x_exp.shape)
        
        # Apply PCA reduction
        if apply_pca:
            x_exp = pca_reduction(x_exp, self.pca, as_tensors=True)
            x_exp_val = pca_reduction(x_exp_val, self.pca, as_tensors=True)
            
        # Encode cancer types and tissue types
        cancer_type= torch.from_numpy(categorical_labels(cancer_type, as_='categorical', nb_class=35)).long() # Get categorical labels for cancer types (34 + 'normal')
        cancer_type_val = torch.from_numpy(categorical_labels(cancer_type_val, as_='categorical', nb_class=35)).long() # Get categorical labels for cancer types (34 + 'normal')
        encoded_tissue_types = torch.from_numpy(self.one_hot_encode_tissue_type(tissue_type)).long()
        encoded_tissue_types_val = torch.from_numpy(self.one_hot_encode_tissue_type(tissue_type_val)).long()
        
        # Turn cancer types into categorical features (not one hot encoded)
        tissue_type = torch.from_numpy(one_hot_to_categoricals(encoded_tissue_types, nb_class=24)).long()
        tissue_type_val = torch.from_numpy(one_hot_to_categoricals(encoded_tissue_types_val, nb_class=24)).long()

        return (x_exp, age, gender, cancer_type, encoded_tissue_types, tissue_type, labels, cohort), (x_exp_val, age_val, gender_val, cancer_type_val, encoded_tissue_types_val, tissue_type_val, labels_val, cohort_val)

