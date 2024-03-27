import numpy as np

# Model architecture  

# We keep the same paths to save results
PWD = './src/classification/large_dim'
RES_DIR = PWD+'/results/model_{}.pth'

CONFIG1 = {'dataset': 'tcga',
            'hidden_dim1' : 1024,
            'hidden_dim2' : 1024,
            'input_dim' : 20531,
            'output_dim' : 24, # tcga
            'classification_task' : 'tissue_type',
            'nb_classes' : 24,
            'activation' : 'relu',
            'dropout' : 0.5,
            'BN' : False,
            'optimizer' : 'sgd',
            'lr' : 0.0125,
            'batch_size' : 64,
            'epochs' : 150,
            'path' : RES_DIR,
            'num_workers':0,
}

CONFIG2 = {'dataset':'gtex',
            'hidden_dim1' : 2048,
            'hidden_dim2' : 8192,
            'input_dim' : 18691,
            'output_dim' : 26, # gtex
            'classification_task' : 'tissue_type',
            'nb_classes' : 26,
            'activation' : 'relu',
            'dropout' : 0.5,
            'BN' : False,
            'optimizer' : 'adam',
            'lr' : 0.0005,
            'batch_size' : 1024,
            'epochs' : 150,
            'path' : RES_DIR,
            'num_workers':0,
}


# Final configurations dictionary
LIST_CONFIGS = [CONFIG1, CONFIG2]
CONFIGS = {i:j for i,j in zip(np.arange(1,len(LIST_CONFIGS)+1), LIST_CONFIGS)}