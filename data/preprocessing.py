"""
Get clinical data or RNAseq from TCGA datasets obtained with RTCGA package on R.

"""

# Imports
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') # Ignore warnings of pandas
from time import process_time
import random
random.seed(1)
from os.path import expanduser
USER_HOME = expanduser("~")


class RNADataProcessor():
    def __init__(self, _type_='clinical', cancer_type='all', dataset='tcga'):
        """ Init RNAseqDATA class with data type 'clinical' or 'data' for gene expressions and cancer type with 'all' by default
        ----------
        _type_ (str): data type either 'clinical' or RNAseq 'data'
        cancer_type (str): 'all' cancer types is the only option for now
        dataset (str): default is tcga, else 'microarrays'

        """
        assert _type_.lower() in ['clinical', 'data']
        assert cancer_type.lower() in ['all']
        assert dataset.lower() in ['tcga']


        self.type = _type_
        self.cancer_type = cancer_type
        self.dataset= dataset.lower()

        if dataset.lower() =='tcga':
            if not os.path.exists('./tcga_files'):
                print("Loader could not find 'tcga_files' folder..")
            else:
                print(f"Loader ready.")

            self.PATH = './tcga_files'
            self.PATHFOLDER = '/gdac.broadinstitute.org_{}.Mer'
            self.PATHFILE = '/{}.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt'
            self.PATH_CLINICALFOLDER = '/clinical/gdac.broadinstitute.org_{}.Merge_Clinical.Level_1.2016012800.0.0'
            self.PATH_CLINICALFILE = '/{}.clin.merged.txt'
            self.COHORTS = ["ACC","BLCA","BRCA","CESC","CHOL","COAD","DLBC","ESCA","GBM", "HNSC","KICH","KIRC","KIRP",
"LAML","LGG","LIHC","LUAD","LUSC","MESO","OV","PAAD","PCPG","PRAD","READ","SARC","SKCM","STAD","STES","TGCT","THCA","THYM","UCEC","UCS","UVM"] #COADREAD, KIPAN and GBMLGG REMOVED

    def preprocessing(self):
        """Preprocessing according to dataset type
        ---"""

        if self.dataset=='tcga':
            self.preprocessing_tcga()


    @staticmethod
    def standardize(x, mean=None, std=None):
        """
        Shape x: (nb_samples, nb_vars)
        """
        if mean is None:
            mean = np.mean(x, axis=0)
        if std is None:
            std = np.std(x, axis=0)
        return (x - mean) / std


    def preprocessing_tcga(self):
        """Preprocessing of data either clinical or rnaseq for TCGA data
        ---------------------- """

        """Loading individual  dataset"""
        if self.type=='clinical':
            #Function to read CSV
            clin_UVM = pd.read_csv(self.PATH+self.PATH_CLINICALFOLDER.format('UVM')+self.PATH_CLINICALFILE.format('UVM'), sep='\t', header=None)

        elif self.type=='data' and self.cancer_type =='all':

            t1 = process_time()

            for COHORT in self.COHORTS:

                #Load cohort dataframe
                try:
                    df_temp = pd.read_csv(self.PATH+self.PATHFOLDER.format(COHORT)+self.PATHFILE.format(COHORT), sep='\s+')
                except FileNotFoundError:
                    df_temp = pd.read_csv(self.PATH+self.PATHFOLDER.format(COHORT)+'g'+self.PATHFILE.format(COHORT), sep='\s+')

                print(f'[Callback]| Starting preprocessing for {COHORT}.')

                """Preprocessing of data"""
                PATIENTS_IDs = list(df_temp.columns[2:])
                #Get tuples of genes ids hybridization
                GENE_IDs = np.asarray([['','']]+[x.split('|') for x in list(df_temp['Hybridization'])[1:]]) #Same order check
                #New df
                df = df_temp[['REF']+PATIENTS_IDs]
                #Old columns misplaced
                OLD_COLS = list(df.columns)
                #Drop last column with NaN
                #print(OLD_COLS[-1]) #to drop
                df.drop(OLD_COLS[-1], inplace=True, axis=1)
                #Rename columns
                df.rename(columns={x:y for x,y in zip(OLD_COLS[:-1], PATIENTS_IDs )}, inplace=True)
                #Add columns with gene IDs
                df.insert(0, 'Hybridization' ,GENE_IDs[:,0])
                df.insert(1, 'Gene_id', GENE_IDs[:,1])
                #Drop string row of 'normalized count'
                df.drop(0,axis=0,inplace=True)

                # Save gene ids once
                if COHORT=='ACC':
                    np.save(self.PATH+'/TCGA_rnaseq_RSEM_gene_ids_full.npy', GENE_IDs[:,1])
                    np.save(self.PATH+'/TCGA_rnaseq_RSEM_hybridization_full.npy', GENE_IDs[:,0])

                #Get labels according to bar code: 4th element in barcode is a number+letter like '01A', if number <10 then cancer, else: not cancer
                LABELS = [1 if int(patient_id.split('-')[3][0:2]) < 10 else 0 for patient_id in PATIENTS_IDs]
                #print(LABELS)

                #Get data
                #convert to floats
                df[PATIENTS_IDs] =df[PATIENTS_IDs].apply(pd.to_numeric)
                SAMPLES = df[PATIENTS_IDs].to_numpy().transpose() #We transpose to have 1 row per patient

                #Remove nans
                SAMPLES[np.isnan(SAMPLES)] = 0.0

                #Keep only cancerous tissues for multiclass classification (patients with disease)
                DIS_SAMPLES = SAMPLES[np.asarray(LABELS)==1]


                if COHORT != 'ACC':
                    #We don't keep cancer type for non cancerous tissues
                    cancer_types_multiclass.extend(len(np.asarray(LABELS)[np.asarray(LABELS)==1])*[COHORT])
                    cancer_types.extend(len(np.asarray(LABELS))*[COHORT])
                    samples = np.append(samples, SAMPLES, axis=0)
                    dis_samples = np.append(dis_samples, DIS_SAMPLES, axis=0)
                    labels = np.append(labels, np.asarray(LABELS), axis=0)
                    patients_ids = np.append(patients_ids, PATIENTS_IDs, axis=0)

                elif COHORT =='ACC':
                    samples=SAMPLES
                    dis_samples = DIS_SAMPLES
                    labels = LABELS
                    patients_ids = PATIENTS_IDs
                    #Save cancer type for multiclass labels of cancerous tissues only
                    cancer_types_multiclass = len(np.asarray(LABELS)[np.asarray(LABELS)==1])*[COHORT]
                    cancer_types = len(np.asarray(LABELS))*[COHORT]


            print(f"Preprocessing on TCGA data is complete.\n---> Saving {samples.shape[0]} RSEM RNAseq samples...")
            #Save data in local folder
            np.save(self.PATH+'/TCGA_rnaseq_RSEM_preprocess_full.npy', samples)
            np.save(self.PATH+'/TCGA_rnaseq_RSEM_preprocess_multiclass_full.npy', dis_samples)
            np.save(self.PATH+'/TCGA_rnaseq_RSEM_labels_full.npy', labels)
            np.save(self.PATH+'/TCGA_rnaseq_RSEM_patients_ids_full.npy', patients_ids)
            np.save(self.PATH+'/TCGA_rnaseq_RSEM_cancer_types_full.npy', cancer_types)
            np.save(self.PATH+'/TCGA_rnaseq_RSEM_cancer_types_multiclass_full.npy', cancer_types_multiclass)

            t_end = process_time()
            print(f'[Callback]| Preprocessing took {round(t_end - t1, 5)} seconds to run.')


Processor = RNADataProcessor(_type_='data', cancer_type='all', dataset='tcga')
Processor.preprocessing()

df = Loader.build_full_dataframe('./tcga_files')
df.to_csv('./tcga_files/TCGA_rnaseq_RSEM_full_dataset.csv', index=False)
df_final = Loader.process_all_covariates('./tcga_files/TCGA_rnaseq_RSEM_full_dataset.csv')

from utils_preprocessing import train_test_split

df_train, df_test = train_test_split(df_final, split_ratios=[0.8, 0.2], shuffle=True)

df_train.to_csv('./tcga_files/train_df_covariates.csv', index=False)
df_test.to_csv('./tcga_files/test_df_covariates.csv', index=False)


