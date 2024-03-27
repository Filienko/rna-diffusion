
##### ---------------- Best params GTEx 974 landmark genes ---------------- #####
python main_vae.py -dataset 'gtex' -gpu_device "cuda:0" -gpu_device2 "cuda:1"
##### ---------------- Best params TCGA 978 landmark genes ---------------- #####
python main_vae.py -dataset 'tcga' -gpu_device "cuda:0" -gpu_device2 "cuda:1"


##### ---------------- Reverse val: Best params GTEx 974 landmark genes ---------------- #####
python reverse_val_vae.py -dataset 'gtex' -with_best_params "y" -nb_runs 15 -gpu_vae "cuda:0" -gpu_mlp "cuda:1"
##### ---------------- Reverse val: Best params TCGA 978 landmark genes ---------------- #####
python reverse_val_vae.py -dataset 'tcga' -with_best_params "y" -nb_runs 15 -gpu_vae "cuda:0" -gpu_mlp "cuda:1"


##### ---------------- Reconstruct VAE-generated data with Linear Regression ---------------- #####
python reconstruction_from_vae.py -dataset 'gtex' -nb_runs 5 -nb_nn 50 -gpu_device "cuda:0" -device2 "cpu"
python reconstruction_from_vae.py -dataset 'tcga' -nb_runs 5 -nb_nn 50 -gpu_device "cuda:0" -device2 "cpu"
##### ---------------- Reconstruct VAE-generated data with MLP ---------------- #####
python reconstruction_mlp_from_vae.py -dataset 'gtex' -nb_runs 5 -nb_nn 50 -gpu_device "cuda:0" -device2 "cuda:1"
python reconstruction_mlp_from_vae.py -dataset 'tcga' -nb_runs 5 -nb_nn 50 -gpu_device "cuda:0" -device2 "cuda:1"


##### ---------------- Reverse val: VAE TCGA 978 landmark genes + reconstructed genes---------------- #####
python reverse_val_from_reconstruction.py -dataset 'tcga' -gen_model 'vae' -recon_model 'lr' -nb_runs 15 -gpu_device 'cuda:0'
python reverse_val_from_reconstruction.py -dataset 'tcga' -gen_model 'vae' -recon_model 'mlp' -nb_runs 15 -gpu_device 'cuda:0'
##### ---------------- Reverse val: VAE GTEx 974 landmark genes + reconstructed genes---------------- #####
python reverse_val_from_reconstruction.py -dataset 'gtex' -gen_model 'vae' -recon_model 'lr' -nb_runs 15 -gpu_device 'cuda:0'
python reverse_val_from_reconstruction.py -dataset 'gtex' -gen_model 'vae' -recon_model 'mlp' -nb_runs 15 -gpu_device 'cuda:0'