""" Processing GTEx data """

import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-path_gtex", "--path_gtex",
                        dest = "path_gtex",
                        type = str,
                        required = True,
                        help="Specify the path to GTEx dataset (required).")

parser.add_argument("-path_tissues", "--path_tissues",
                        dest = "path_tissues",
                        type = str,
                        required = True,
                        help="Specify the path to GTEx tissues information (required).")

parser.add_argument("-path_clinical", "--path_clinical",
                        dest = "path_clinical",
                        type = str,
                        required = True,
                        help="Specify the path to GTEx clinical information (required).")

parser.add_argument("-coding_genes", "--coding_genes",
                        dest = "coding_genes",
                        type = str,
                        required = True,
                        help="Specify whether to keep only coding genes (required).")

parser.add_argument("-landmark_only", "--landmark_only",
                        dest = "landmark_only",
                        type = str,
                        required = True,
                        help="Specify whether to keep only landmark genes (required).")


args = parser.parse_args()
file_name = args.path_gtex
clinical_file_name = args.path_clinical
tissues_file_name = args.path_tissues
coding_only = args.coding_genes
landmark_only = args.landmark_only

print(f'File Size is {os.stat(file_name).st_size / (1024 * 1024)} MB')

def main():

    ################################## 

    # Load data (remove 2 first rows)
    df = pd.read_table(file_name, sep='\t', header=2)

    print(f"Dataset has {df.shape[0]} variables and {df.shape[1]} samples.")

    # Doublons
    assert np.unique(df.columns).shape == df.columns.shape, "Warning: dupplicates found in patients IDs"

    ##################################

    # Coding genes
    if coding_only=='y':
        df_ensembl_coding = pd.read_csv('map_ensembl_codants.csv', sep=',')
        # Keep ensembl IDs as integer values but string type
        df['Name'] = df['Name'].apply(lambda x: str(x.split('.')[0]))
        # Filter with only coding genes ensembl IDs 
        df = df[df['Name'].isin(df_ensembl_coding['ENSEMBL'].values)]

    ##################################

    # Tissues details for each sample
    """
    SAMPID: patient ID
    SMTS: tissue
    SMTSD: tissue details
    """
    df_tissues = pd.read_table(tissues_file_name, sep='\t', header=0)
    df_tissues = df_tissues[['SAMPID', 'SMTS', 'SMTSD']]

    # Clinical data (age, sex) for each donor
    df_clinical = pd.read_table(clinical_file_name, sep='\t', header=0)

    # Add common key for covariates merger
    df_clinical['patient_id'] = [df_clinical['SUBJID'].iloc[i].split('-')[1] for i in range(len(df_clinical))]
    # Add common key
    df_tissues['patient_id'] = [df_tissues['SAMPID'].iloc[i].split('-')[1] for i in range(len(df_tissues))]
    # Merge
    df_covariates = df_tissues.merge(df_clinical, left_on='patient_id', right_on='patient_id')

    # Drop patient id
    df_covariates = df_covariates.drop('patient_id', axis=1)
    # Drop death
    df_covariates = df_covariates.drop('DTHHRDY', axis=1)
    # Drop one key
    df_covariates = df_covariates.drop('SUBJID', axis=1)
    # Replace sex by male/female
    df_covariates.loc[df_covariates['SEX'] == 1, 'SEX'] = "male"
    df_covariates.loc[df_covariates['SEX'] == 2, 'SEX'] = "female"
    # Convert age to float
    df_covariates['AGE']=[float(df_covariates['AGE'][i].split('-')[0]) for i in range(len(df_covariates))]
    # Rename columns age, sex
    df_covariates = df_covariates.rename(columns={"AGE": "age", "SEX": "gender", "SMTS":"tissue_type", "SMTSD":"tissue_type_details"}) 
    name_vars = df_covariates.transpose().index[1:].values # index vars names
    # Rebuild covariates dataframe
    df_covariates = pd.DataFrame(data=df_covariates.transpose().loc[['tissue_type', 'tissue_type_details', 'gender', 'age']].values,
                                    columns = df_covariates.transpose().loc['SAMPID'].values)
    # Add name column
    df_covariates.insert(0, column='Name', value=name_vars)
    # Add name column
    df_covariates.insert(1, column='Description', value=np.zeros_like(df_covariates['Name'].values))

    ##################################
    if landmark_only=='y':
        # Load landmark genes
        df_landmark = pd.read_csv('L978.csv')
        df_landmark['EntrezID'] = df_landmark['EntrezID'].astype(float)
        # Mapping of ensembl IDs to entrezIds (using grex R package https://nanx.me/grex/articles/grex.html)
        df_map = pd.read_csv('map_entrezid_ensembl_gtex.csv')
        df_landmark_mapping = df_map.merge(df_landmark, right_on ='EntrezID', left_on ='entrez_id', how='inner') # there should be 974 remaining genes

        # Remove versionning of Ensembl IDs
        df['ensembl'] = [e.split('.')[0] for e in df['Name']]
        # Merge
        df = df.merge(df_landmark_mapping, right_on ='ensembl_id', left_on ='ensembl', how='inner')

    ##################################

    # Mapping expression and covariates
    df = pd.concat([df, df_covariates], join="inner")

    # Transpose to obtain a dataframe of size (n_samples, n_genes):
    df = df.transpose()

    # Rename columns with EntrezIds
    df.columns=list(np.concatenate((df.iloc[1].values.flatten()[:-4],df.iloc[0].values.flatten()[-4:])))

    print(df)

    ##################################

    # Keep only cohorts with more than 100 samples per tissue type
    count_tissues = np.zeros_like(np.unique(df['tissue_type']))
    for i in range(len(count_tissues)):
        count_tissues[i] = (df['tissue_type'].values== np.unique(df['tissue_type'])[i]).sum().item()

    for t in np.unique(df['tissue_type'])[count_tissues <100]:
        idx_to_drop = df.loc[df['tissue_type'] == t].index.values
        df = df.drop(idx_to_drop)
    
    df = df.reset_index()

    # columns as type float
    for col in df.columns[1:-5]:
        df[col] = df[col].astype(float) 

    ##################################
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(df, df['tissue_type'], test_size=5000, random_state=42)

    # Save
    if landmark_only=='y':
        X_train.to_csv('df_train_gtex_L974.csv', header=True, index=False)
        X_val.to_csv('df_test_gtex_L974.csv', header=True, index=False)
    else:
        X_train.to_csv('df_train_gtex.csv', header=True, index=False)
        X_val.to_csv('df_test_gtex.csv', header=True, index=False)

    print(f"Final GTEx train set holds {X_train.shape[0]} samples.")
    print(f"Final GTEx test set holds {X_val.shape[0]} samples.")


# Run function    
if __name__ == '__main__':
    main()