import numpy as np

# Model architecture 

# We keep the same paths to save results
PWD = '/home/alacan/scripts/gerec_pipeline/src/reconstruction'
RES_DIR = PWD+'/results/model_{}.pth'
LOG_DIR = PWD+'/logs/logs_{}/'

CONFIG1 = {
            'hidden_dim1' : 1024,
            'hidden_dim2' : 2048,
            'hidden_dim3' : 4096,
            'input_dim' : 978,
            'output_dim' : 19553, # tcga
            'activation' : 'relu',
            'dropout' : 0.5,
            'BN' : True,
            'optimizer' : 'adam',
            'lr' : 0.001,
            'momentum':0.9,
            'nb_nn':50,
            'batch_size' : 2048,
            'epochs' : 250,
            'path' : RES_DIR.format('tcga'),
            'path_logs': LOG_DIR.format('tcga'),
            'num_workers':2,
}

CONFIG2 = {
            'hidden_dim1' : 1024,
            'hidden_dim2' : 2048,
            'hidden_dim3' : 4096,
            'input_dim' : 974,
            'output_dim' : 17717, # gtex
            'activation' : 'relu',
            'dropout' : 0.5,
            'BN' : True,
            'optimizer' : 'adam',
            'lr' : 0.001,
            'momentum':0.9,
            'nb_nn':50,
            'batch_size' : 2048,
            'epochs' : 250,
            'path' : RES_DIR.format('gtex'),
            'path_logs': LOG_DIR.format('gtex'),
            'num_workers':2,
}


# Final configurations dictionary
LIST_CONFIGS = [CONFIG1, CONFIG2]
CONFIGS = {i:j for i,j in zip(np.arange(1,len(LIST_CONFIGS)+1), LIST_CONFIGS)}