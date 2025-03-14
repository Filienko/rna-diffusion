# Imports
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from time import process_time
import random
random.seed(1)

class RNADataProcessor():
    def __init__(self, counts_file, subtypes_file, output_dir='./processed_data'):
        """
        Initialize RNADataProcessor with paths to input files
        
        Parameters:
        -----------
        counts_file : str
            Path to the counts file with gene expression data
        subtypes_file : str
            Path to the file containing sample subtypes
        output_dir : str
            Directory to save processed data files
        """
        self.counts_file = counts_file
        self.subtypes_file = subtypes_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Processor initialized with:\n- Counts file: {counts_file}\n- Subtypes file: {subtypes_file}")
    
    @staticmethod
    def standardize(x, mean=None, std=None):
        """
        Standardize data to have mean 0 and std 1
        
        Parameters:
        -----------
        x : numpy array, shape (nb_samples, nb_vars)
            Data to standardize
        mean : numpy array or None
            Mean values. If None, calculated from x
        std : numpy array or None
            Standard deviation values. If None, calculated from x
        
        Returns:
        --------
        numpy array
            Standardized data
        """
        if mean is None:
            mean = np.mean(x, axis=0)
        if std is None:
            std = np.std(x, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        return (x - mean) / std

    def load_data(self):
        """
        Load data from the counts and subtypes files
        
        Returns:
        --------
        tuple (counts_df, subtypes_df)
            DataFrames containing counts and subtypes data
        """
        print("Loading data files...")
        try:
            # For counts file, we need to parse it properly - assuming tab separated or space separated
            counts_df = pd.read_csv(self.counts_file, sep='\t')
            
            # Load subtypes file
            subtypes_df = pd.read_csv(self.subtypes_file)
            
            return counts_df, subtypes_df
        
        except Exception as e:
            print(f"Error loading data: {e}")
            # Try alternative separator for counts file
            try:
                counts_df = pd.read_csv(self.counts_file, sep=',')
                subtypes_df = pd.read_csv(self.subtypes_file)
                return counts_df, subtypes_df
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                raise
    
    def preprocess(self):
        """
        Preprocess the RNA-seq data
        
        Returns:
        --------
        dict
            Dictionary containing processed data
        """
        t1 = process_time()
        print("Starting preprocessing...")
        
        # Load data
        counts_df, subtypes_df = self.load_data()
        
        # Clean up column names in counts dataframe if needed
        # Assuming the first column contains gene identifiers
        gene_id_col = counts_df.columns[0]
        sample_cols = [col for col in counts_df.columns if col != gene_id_col]
        
        # Extract gene IDs and expression data
        gene_ids = counts_df[gene_id_col].values
        
        # Check if there's a Hybridization column
        if 'Hybridization' in counts_df.columns:
            hybridization = counts_df['Hybridization'].values
        else:
            hybridization = None
        
        # Convert expression data to numeric and handle any non-numeric values
        samples_data = counts_df[sample_cols].apply(pd.to_numeric, errors='coerce')
        
        # Replace NaNs with zeros
        samples_data.fillna(0, inplace=True)
        
        # Process subtype information
        # Make sure sample IDs match between counts and subtypes
        common_samples = list(set(sample_cols).intersection(set(subtypes_df['samplesID'])))
        if len(common_samples) == 0:
            print("Warning: No matching sample IDs between counts and subtypes files")
            # Try to match by removing version numbers or other common differences in sample naming
            # This would need customization based on your actual data format
        
        # Extract common samples
        subtypes_filtered = subtypes_df[subtypes_df['samplesID'].isin(common_samples)]
        samples_data_filtered = samples_data[common_samples]
        
        # Create labels from subtypes
        cancer_types = subtypes_filtered['cancer_type'].values
        subtypes = subtypes_filtered['Subtype'].values
        patients_ids = subtypes_filtered['samplesID'].values
        
        # Create tumor/normal labels if possible (based on TCGA barcode convention)
        # TCGA barcodes: If the 4th element starts with a number < 10, it's a tumor sample
        labels = []
        for patient_id in patients_ids:
            try:
                # Check if it's a TCGA barcode format
                if '-' in patient_id and len(patient_id.split('-')) >= 4:
                    sample_type = patient_id.split('-')[3]
                    if sample_type.startswith(('01', '02', '03', '04', '05', '06', '07', '08', '09')):
                        labels.append(1)  # Tumor
                    else:
                        labels.append(0)  # Normal
                else:
                    # Default to 1 (tumor) if format is unknown
                    labels.append(1)
            except:
                labels.append(1)  # Default to tumor
        
        # Convert data to numpy arrays for further processing
        expression_data = samples_data_filtered.values.T  # Transpose to get samples as rows
        
        # Save processed data
        result = {
            'expression_data': expression_data,
            'gene_ids': gene_ids,
            'hybridization': hybridization,
            'patient_ids': patients_ids,
            'cancer_types': cancer_types,
            'subtypes': subtypes,
            'labels': np.array(labels)
        }
        
        # Save to files
        self.save_processed_data(result)
        
        t_end = process_time()
        print(f'Preprocessing complete in {round(t_end - t1, 5)} seconds.')
        
        return result
    
    def save_processed_data(self, data_dict):
        """
        Save processed data to output directory
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary containing data to save
        """
        # Create CSV for the main dataset
        df_main = pd.DataFrame(data_dict['expression_data'])
        df_main.index = data_dict['patient_ids']
        
        # Add metadata columns
        df_main['cancer_type'] = data_dict['cancer_types']
        df_main['subtype'] = data_dict['subtypes']
        df_main['label'] = data_dict['labels']
        
        # Save the main dataframe
        main_output = os.path.join(self.output_dir, 'processed_dataset.csv')
        df_main.to_csv(main_output)
        
        # Save metadata separately
        meta_df = pd.DataFrame({
            'patient_id': data_dict['patient_ids'],
            'cancer_type': data_dict['cancer_types'],
            'subtype': data_dict['subtypes'],
            'label': data_dict['labels']
        })
        meta_output = os.path.join(self.output_dir, 'metadata.csv')
        meta_df.to_csv(meta_output, index=False)
        
        # Save gene information
        gene_df = pd.DataFrame({'gene_id': data_dict['gene_ids']})
        if data_dict['hybridization'] is not None:
            gene_df['hybridization'] = data_dict['hybridization']
        gene_output = os.path.join(self.output_dir, 'gene_info.csv')
        gene_df.to_csv(gene_output, index=False)
        
        # Save numpy arrays for easier loading in machine learning tasks
        np.save(os.path.join(self.output_dir, 'expression_data.npy'), data_dict['expression_data'])
        np.save(os.path.join(self.output_dir, 'cancer_types.npy'), data_dict['cancer_types'])
        np.save(os.path.join(self.output_dir, 'subtypes.npy'), data_dict['subtypes'])
        np.save(os.path.join(self.output_dir, 'labels.npy'), data_dict['labels'])
        np.save(os.path.join(self.output_dir, 'patient_ids.npy'), data_dict['patient_ids'])
        
        print(f"Saved processed data to {self.output_dir}")

def train_test_split(df, split_ratios=[0.8, 0.2], shuffle=True, random_state=42):
    """
    Split a dataframe into train and test sets
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame to split
    split_ratios : list
        Proportions for train and test sets. Must sum to 1.
    shuffle : bool
        Whether to shuffle the data before splitting
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple (df_train, df_test)
        Train and test DataFrames
    """
    assert sum(split_ratios) == 1, "Split ratios must sum to 1"
    
    # Get indices and shuffle if needed
    indices = np.arange(len(df))
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)
    
    # Calculate split points
    split_points = [int(len(df) * ratio) for ratio in split_ratios]
    split_points = np.cumsum(split_points)
    
    # Split indices
    train_indices = indices[:split_points[0]]
    test_indices = indices[split_points[0]:]
    
    # Create datasets
    df_train = df.iloc[train_indices].copy().reset_index(drop=True)
    df_test = df.iloc[test_indices].copy().reset_index(drop=True)
    
    print(f"Split dataset: Train set size = {len(df_train)}, Test set size = {len(df_test)}")
    
    return df_train, df_test

# Usage example
if __name__ == "__main__":
    # Replace with your actual file paths
    # counts_file = "/home/daniilf/rna-diffusion/data/combined/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv"
    # subtypes_file = "/home/daniilf/rna-diffusion/data/combined/TCGA-COMBINED_primary_tumor_subtypes.csv"
    output_dir = "./processed_tcga_data"
    counts_file = "/home/daniilf/rna-diffusion/data/brca/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv"
    subtypes_file = "/home/daniilf/rna-diffusion/data/brca/TCGA-BRCA_primary_tumor_subtypes.csv"
 
    # Create processor and preprocess data
    processor = RNADataProcessor(counts_file, subtypes_file, output_dir)
    processed_data = processor.preprocess()
    
    # Load the processed dataset for train/test split
    df_final = pd.read_csv(os.path.join(output_dir, 'processed_dataset.csv'))
    
    # Split into train and test sets
    df_train, df_test = train_test_split(df_final, split_ratios=[0.8, 0.2], shuffle=True)
    
    # Save train and test sets
    df_train.to_csv(os.path.join(output_dir, 'train_df.csv'), index=False)
    df_test.to_csv(os.path.join(output_dir, 'test_df.csv'), index=False)
    
    print("Processing complete!")

