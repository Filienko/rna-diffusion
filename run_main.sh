
##### ---------------- Best params GTEx 974 landmark genes ---------------- #####
python main_ddim.py --config gtex_landmark_ddim_baseline.yml --with_best_params "y" --exp exp_best_params_landmark_gtex/ --doc tcga --sample --fid --timesteps 1000 --eta 0 --ni --device "[0]" --device2 "cuda:1"
##### ---------------- Best params TCGA 978 landmark genes ---------------- #####
python main_ddim.py --config tcga_landmark_ddim_baseline.yml --with_best_params "y" --exp exp_best_params_landmark_tcga/ --doc tcga --sample --fid --timesteps 1000 --eta 0 --ni --device "[0]" --device2 "cuda:1"


##### ---------------- Reverse val: Best params GTEx 974 landmark genes ---------------- #####
python reverse_validation.py --config gtex_landmark_ddim_baseline.yml --with_best_params "y" --exp exp_best_params_landmark_gtex/ --doc tcga --sample --fid --timesteps 1000 --eta 0 --ni --device "[0]" --device2 "cuda:1"
##### ---------------- Reverse val: Best params TCGA 978 landmark genes ---------------- #####
python reverse_validation.py --config tcga_landmark_ddim_baseline.yml --with_best_params "y" --exp exp_best_params_landmark_tcga/ --doc tcga --sample --fid --timesteps 1000 --eta 0 --ni --device "[0]" --device2 "cuda:1"


##### ---------------- Reconstruct DDIM-generated data with Linear Regression ---------------- #####
python reconstruction_from_ddim.py --config gtex_landmark_ddim_baseline.yml --with_best_params "y" --exp exp_best_params_landmark_gtex/ --doc tcga --sample --fid --timesteps 1000 --eta 0 --ni --device "[0]" --device2 "cuda:1"
python reconstruction_from_ddim.py --config tcga_landmark_ddim_baseline.yml --with_best_params "y" --exp exp_best_params_landmark_tcga/ --doc tcga --sample --fid --timesteps 1000 --eta 0 --ni --device "[0]" --device2 "cuda:1"
##### ---------------- Reconstruct DDIM-generated data with MLP ---------------- #####
python reconstruction_mlp_from_ddim.py --config gtex_landmark_ddim_baseline.yml --with_best_params "y" --exp exp_best_params_landmark_gtex/ --doc tcga --sample --fid --timesteps 1000 --eta 0 --ni --device "[0]" --device2 "cuda:1"
python reconstruction_mlp_from_ddim.py --config tcga_landmark_ddim_baseline.yml --with_best_params "y" --exp exp_best_params_landmark_tcga/ --doc tcga --sample --fid --timesteps 1000 --eta 0 --ni --device "[0]" --device2 "cuda:1"


##### ---------------- Reverse val: Best params TCGA 978 landmark genes + reconstructed genes---------------- #####
python reverse_val_from_reconstruction.py -dataset 'tcga' -gen_model 'ddim' -recon_model 'lr' -nb_runs 15 -gpu_device 'cuda:0'
python reverse_val_from_reconstruction.py -dataset 'tcga' -gen_model 'ddim' -recon_model 'mlp' -nb_runs 15 -gpu_device 'cuda:0'
##### ---------------- Reverse val: Best params GTEx 974 landmark genes + reconstructed genes---------------- #####
python reverse_val_from_reconstruction.py -dataset 'gtex' -gen_model 'ddim' -recon_model 'lr' -nb_runs 15 -gpu_device 'cuda:0'
python reverse_val_from_reconstruction.py -dataset 'gtex' -gen_model 'ddim' -recon_model 'mlp' -nb_runs 15 -gpu_device 'cuda:0'