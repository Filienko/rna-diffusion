# WGAN-GP architecture and training hyperparameters

# BEST config 
CONFIG = {
# Architecture
'latent_dim': 32,
'x_dim': 978,
'output_dim': 1,
'embedded_dim': 2,
'hidden_dim1_g': 64,
'hidden_dim2_g': 512,
'hidden_dim3_g': 1024,
'hidden_dim4_g': None,
'hidden_dim5_g': None,
'hidden_dim1_d': 512,
'hidden_dim2_d': 256,
'vocab_size': 24,
# Training
'batch_size':2048,
'epochs':1000,
'optimizer': 'adam',
'lr_g': 5e-4,
'lr_d':5e-3,
'activation': 'leaky_relu',
'negative_slope':0.05,
'iters_critic': 5,
'lambda_penalty': 10., 
'prob_success': 0., 
'norm_scale':0.5,
'bn': True,
'sn': False,
# Logs
'num_workers': 4,
'epochs_checkpoints': [],
'checkpoint_dir':'./src/generation/gans/results/checkpoints',
'log_dir': './src/generation/gans/logs',
'fig_dir':'./src/generation/gans/figures',
# Metrics
'step': 5,
'nb_nn': 10,
}
