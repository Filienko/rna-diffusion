# CVAE architecture and training hyperparameters

# BEST config 
CONFIG = {
# Architecture
'latent_dim': 32,
'x_dim': 978,
'output_dim': 978,
'embedding_dim': 2,
'list_dims_encoder':[],
'list_dims_decoder':[],
'n_blocks':2,
'vocab_size': 24,
# Training
'batch_size':2048,
'epochs':1000,
'optimizer': 'adam',
'momentum':0.9,
'lr': 5e-4,
'activation': 'leaky_relu',
'negative_slope':0.05,
'bn': True,
'sn': False,
# Logs
'num_workers': 4,
'epochs_checkpoints': [],
'checkpoint_dir':'./src/generation/vae/checkpoints', 
'log_dir': './src/generation/vae/logs',
'fig_dir':'./src/generation/vae/figures',
# Metrics
'step': 5,
'nb_nn': 10,
}