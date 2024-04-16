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

# import torch.utils.tensorboard as tb
from runners.diffusion import Diffusion

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)


torch.set_printoptions(sci_mode=False) 


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
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    
    
    # Best params config
    if args.with_best_params=='y':
        best_params = pd.read_csv(f"./results/tissue_search_{config['data']['dataset']}.csv", sep=',')
        config['diffusion']['beta_schedule'] = str(best_params['beta_schedule'].item())
        config['model']['d_layers'] = [int(best_params['dim_layers'].item()), int(best_params['dim_layers'].item())]
        config['model']['dropout'] = float(best_params['dropout'].item())
        config['optim']['lr'] = float(best_params['lr'].item())
        config['model']['is_time_embed'] = bool(best_params['time_sinus_embed'].item())
    
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc) 

    if args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

        if args.fid:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples", args.image_folder
            )
            print("args.image_folder:", args.image_folder)
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input(
                        f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                    )
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)
                else:
                    print("Output image folder exists. Program halted.")
                    sys.exit(0)

    else:
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

    # add device
    list_devices = args.device.strip("[]").split(",")
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

# Tree of dicts to tree of namespaces
def dict2namespace(config): 
    namespace = argparse.Namespace() # Namespace is a simple container object that provides attribute-style access to its members.
    for key, value in config.items():
        if isinstance(value, dict): 
            new_value = dict2namespace(value)
        else: 
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    """
    Main function to run search.
    """
    ARGS, CONFIG, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(ARGS.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(ARGS.comment))

    PATH_RESULTS_DATAFRAME = './results/results_ddim.csv'

    # Instantiate model
    runner = Diffusion(ARGS, CONFIG, device=CONFIG.device)

    # Model training
    start_time = t.time()
    runner.train() 
    end_time = t.time()

    train_time =end_time - start_time
    print(f"Train time: {round(train_time, 2)} sec. (= {round(train_time/60, 2)} min.)")  

    runner = [] # free memory

    ############### Prec/Recall/Density/Coverage/AAts 5 runs on train data ###############

    # Instantiate model again
    runner = Diffusion(ARGS, CONFIG, device=CONFIG.device)

    PREC = []
    RECALL = []
    DENSITY = []
    COVERAGE = []
    AATS = []
    CORR = []
    FD =[]

    for i in range(5):
        # Sampling and store images
        print("----> Sampling images")
        runner.sample()

        # Retrieve metrics
        PREC.append(runner.precision)
        RECALL.append(runner.recall)
        DENSITY.append(runner.density)
        COVERAGE.append(runner.coverage)
        AATS.append(runner.adversarial)
        FD.append(runner.frechet)
        CORR.append(runner.correlation_score)

    # Statistics
    final_prec = np.mean(PREC)    
    final_prec_std = np.std(PREC)    
    
    final_rec = np.mean(RECALL)    
    final_rec_std = np.std(RECALL)    
    
    final_dens = np.mean(DENSITY)    
    final_dens_std = np.std(DENSITY)    
    
    final_cov = np.mean(COVERAGE)    
    final_cov_std = np.std(COVERAGE)  
    
    final_aats = np.mean(AATS)    
    final_aats_std = np.std(AATS)  

    final_corr = np.mean(CORR)    
    final_corr_std = np.std(CORR) 

    final_fd = np.mean(FD)    
    final_fd_std = np.std(FD)  

    # save results
    dict_res_5runs = {'precision': [final_prec, final_prec_std],
                        'recall': [final_rec, final_rec_std],
                        'density': [final_dens, final_dens_std]  ,
                        'coverage': [final_cov, final_cov_std]  ,
                        'aats':  [final_aats, final_aats_std],
                        'correlation_score':[final_corr, final_corr_std],
                         'frechet': [final_fd, final_fd_std] }
    np.save(ARGS.log_path+'/dict_res_metrics_5runs.npy', dict_res_5runs)

    # Loss
    PATH_LOSS = f'{ARGS.log_path}/loss.csv'
    df = pd.read_csv(PATH_LOSS, sep =',')
    
    # Save config + results in csv file
    print("------> Saving csv results file")
    d = {c: config[k][c] for k in config.keys() for c in config[k].keys()}
    d['d_layers']=[d['d_layers']]
    d['attn_resolutions']=[d['attn_resolutions']]
    # Add keys
    d['model_folder_path'] = ARGS.log_path
    d['comment'] = 'Best params'
    d['correlation_score'] = [[final_corr, final_corr_std]]
    d['precision'] = [[final_prec, final_prec_std]] # metric mean + std
    d['recall'] = [[final_rec, final_rec_std]]
    d['density'] = [[final_dens, final_dens_std]]
    d['coverage'] = [[final_cov, final_cov_std]]
    d['aats'] = [[final_aats, final_aats_std]]
    d['fid_tissue'] = [[final_fd, final_fd_std]] 
    d['total_time'] = df['total_time'].values[-1].item()
    d['mean_epoch_time'] = np.mean(df['epoch_time'].values).item()
    d['noise_avg'] = df['avg'].values[-1].item()
    d['loss'] = df['loss'].values[-1].item()

    # build csv
    df_temp = pd.DataFrame(data=d, index=[0])

    # Load results dataframe
    df = pd.read_csv(PATH_RESULTS_DATAFRAME, sep =',')
    
    # Merge dataframes
    df = pd.concat([df, df_temp])
    df.to_csv(PATH_RESULTS_DATAFRAME, sep =',', header=True, index=False)

#This will call the function main() and when main finishes, it will exit giving the system the return code that is the result of main()
#If you execute a Python script directly, __name__ is set to "__main__", but if you import it from another script, it is not.
if __name__ == "__main__":
    sys.exit(main())
    pass