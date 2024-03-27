# Imports
import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import random
import time as t
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.utils.data as data_utils

# import torch.utils.tensorboard as tb
from src.generation.ddim.runners.diffusion import Diffusion

from src.reconstruction.model import LR, mae, mse
from src.reconstruction.utils import get_datasets_split_landmarks_for_search
from src.metrics.precision_recall import compute_prdc, get_precision_recall
from src.metrics.aats import compute_AAts
from src.metrics.correlation_score import gamma_coeff_score


# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)


torch.set_printoptions(sci_mode=False) #Je sais pas vraiment ce que ca va changer pour nous


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--with_best_params", type=str, required=True, help="Run model with best hyperparameters obtained with search."
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Which GPU to use."
    )
    parser.add_argument(
        "--device2",
        type=str,
        required=True,
        help="Which GPU to use for frechet."
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--noisepredi", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")
    args = parser.parse_args()
    # Log path to pretrained model
    # args.log_path = os.path.join(args.exp, "logs", args.doc)
    args.log_path = os.path.join("/home/alacan/scripts/diffusion_models/diffusion/ddim/", args.exp, "logs", args.doc)

    # parse config file
    # with open(os.path.join("configs", args.config), "r") as f:
    with open(os.path.join("./src/generation/ddim/","configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    
    # Best params config
    if args.with_best_params=='y':
        best_params = pd.read_csv(f"./src/generation/ddim/results/tissue_search_{config['data']['dataset']}.csv", sep=',')
        config['diffusion']['beta_schedule'] = str(best_params['beta_schedule'].item())
        config['model']['d_layers'] = [int(best_params['dim_layers'].item()), int(best_params['dim_layers'].item())]
        config['model']['dropout'] = float(best_params['dropout'].item())
        config['optim']['lr'] = float(best_params['lr'].item())
        config['model']['is_time_embed'] = bool(best_params['time_sinus_embed'].item())
    
    new_config = dict2namespace(config)
    # tb_path = os.path.join(args.exp, "tensorboard", args.doc) #TODO? No tensorboard pour l'instant

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    # Samples folder
    args.image_folder = os.path.join("./src/generation/ddim/", args.exp, "image_samples", args.image_folder)
    # args.image_folder = os.path.join(args.exp, "image_samples", args.image_folder)
    print("args.image_folder:", args.image_folder)

    # add device
    #device = [torch.device(args.device[i]) for i in args.device]
    list_devices = args.device.strip("[]").split(",")
    #device = [torch.device(int(list_devices[i])) for i in range(len(list_devices))]
    device = [int(list_devices[i]) for i in range(len(list_devices))]
    logging.info("Using device (s): {}".format(device))
    new_config.device = device
    new_config.device2 = args.device2

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config, config

#fonction qui transforme un arbre de dictionnaires en arbre de namespaces (recursivement)
def dict2namespace(config): 
    namespace = argparse.Namespace() #Namespace is a simple container object that provides attribute-style access to its members.
    for key, value in config.items():
        if isinstance(value, dict): #si on a un dictionnaire, on le transforme en namespace (recursivement)
            new_value = dict2namespace(value)
        else: #sinon on ajoute juste la valeur dans le namespace
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    # Config diffusion model
    ARGS, CONFIG, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(ARGS.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(ARGS.comment))

    DATASET = CONFIG.data.dataset_frechet.lower()
    NB_NN = 50

    # Load true data in original dimensions
    print("----> Loading true data")
    X, y, _, _, tissues_train, _ = get_datasets_split_landmarks_for_search(DATASET, landmark=False, split_landmark=True, with_tissues=True)
    # Scale the data
    scaler = StandardScaler()
    scaler_nl = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler_nl.fit_transform(y)

    # Full true data
    full_true = np.concatenate((X, y), axis=1)

    # Instantiate model
    runner = Diffusion(ARGS, CONFIG, device=CONFIG.device)

    # Init metrics
    PREC, REC, DENS, COV, AATS, CORR = [],[],[],[],[],[]

    # Loop over runs
    print(f"----> Start with 5 runs")
    for i in range(5):
        # Generate landmark genes
        print("----> Sampling images")
        runner.sample()

        # Load generated data
        x_fake = pd.read_csv(f"{ARGS.image_folder}/samples.csv", sep=',', header=None).to_numpy()
        y_fake = pd.read_csv(f"{ARGS.image_folder}/samples_label.csv", sep=',', header=None).to_numpy()

        # Unscale MaxAbs
        scales = np.load(f'./src/generation/ddim/results/{CONFIG.data.dataset_frechet.lower()}_landmark_scales.npy', allow_pickle=True)
        x_fake = x_fake*scales

        # Standardize (reduce and center)
        # Scale the data
        scaler_fake = StandardScaler()
        x_fake = scaler_fake.fit_transform(x_fake)

        # Reconstruct target genes
        print("----> Load reconstructing model")
        GeRec = LR(pretrained_path=f'./src/reconstruction/results/lr_{DATASET}.joblib')
        # Reconstruct
        start = t.time()
        preds_fake = GeRec(x_fake)
        recon_time = t.time() - start
        
        # Concatenate
        x_fake = np.concatenate((x_fake, preds_fake), axis=1)

        # Precision/Recall/Density/Coverage
        print("PRDC...")
        # prec, recall, dens, cov = compute_prdc(full_true, x_fake, NB_NN)
        prec, recall = get_precision_recall(torch.from_numpy(full_true).float(), torch.from_numpy(x_fake).float(), [NB_NN])
        PREC.append(prec)
        REC.append(recall)
        print("prec, recall:", prec, recall)
        # Adversarial accuracy
        print("AATS...")
        idx = np.random.choice(len(full_true), 2048, replace=False) # Sample random data
        _, _, aa = compute_AAts(real_data=full_true[idx], fake_data=x_fake[idx])
        # Correlations
        print("Correlations...")
        corr = gamma_coeff_score(full_true[idx], x_fake[idx])
        AATS.append(aa)
        CORR.append(corr)

        # Save reconstructed data
        if i==0:
            if DATASET=='tcga':
                landmark_genes_id = pd.read_csv(f"/home/alacan/data_RNAseq_RTCGA/landmark_genes_ids.csv").values.flatten()
                non_landmark_genes_id = pd.read_csv(f"/home/alacan/data_RNAseq_RTCGA/non_landmark_genes_ids.csv").values.flatten()

            elif DATASET=='gtex':
                df_descript = pd.read_csv('/home/alacan/scripts/gerec_pipeline/gtex_description.csv', sep=',')
                # df_descript['entrez_id'] = df_descript['entrez_id'].astype('str')
                landmark_genes_id = df_descript[df_descript.Type=='landmark']['Description'].values.flatten()
                non_landmark_genes_id = df_descript[df_descript.Type=='target']['Description'].values.flatten()
            
            pd.DataFrame(data=x_fake, columns=np.concatenate((landmark_genes_id, non_landmark_genes_id))).to_csv(f'./src/reconstruction/results/fake_from_ddim_reconstructed_lr_{DATASET}.csv')
            pd.DataFrame(data=y_fake.argmax(1).reshape(-1,1), columns=['tissue_type']).to_csv(f'./src/reconstruction/results/fake_from_ddim_reconstructed_lr_{DATASET}_labels.csv')
    
        # memory
        x_fake = []

    # Save results
    print(f"----> Store best results.")
    data = [recon_time]
    data_cols = ['reconstruction_time']

    df_res = pd.DataFrame(columns=data_cols,
                            data=np.array([data]))

    df_res['precision'], df_res['precision_std'] = np.mean(PREC), np.std(PREC)
    df_res['recall'], df_res['recall_std'] = np.mean(REC), np.std(REC)
    df_res['aats'], df_res['aats_std'] = np.mean(AATS), np.std(AATS)
    df_res['correlation'], df_res['correlation_std'] = np.mean(CORR), np.std(CORR)

    df_res.to_csv(f'./src/reconstruction/results/results_lr_{DATASET}_from_ddim.csv', index=False)
    print(f"----> End. ")

if __name__ == "__main__":
    sys.exit(main())
    pass