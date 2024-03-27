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

from src.generation.ddim.utils import build_loaders, split_and_scale_datasets, get_datasets_for_search

# import torch.utils.tensorboard as tb
from src.generation.ddim.runners.diffusion import Diffusion

# Import classifier config and hyperparameters
from src.metrics.pretrained_mlp import TissuePredictor, get_class_weights, CONFIGS

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
    """
    """
    # Config diffusion model
    ARGS, CONFIG, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(ARGS.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(ARGS.comment))

    # Config classifier
    if 'tcga' in CONFIG.data.dataset:
        CONFIG_CLS = CONFIGS[1]
    elif 'gtex' in CONFIG.data.dataset:
        CONFIG_CLS = CONFIGS[2]
    CONFIG_CLS['device'] = CONFIG.device2
    # New path to store mlp
    CONFIG_CLS['path'] = f'./src/generation/ddim/results/cls_fake_{CONFIG.data.dataset_frechet.lower()}.pth'

    print("----> Loading true test data")
    # Load test data
    _, _, X_test, y_test = get_datasets_for_search(CONFIG.data.dataset_frechet)

    # Instantiate model
    runner = Diffusion(ARGS, CONFIG, device=CONFIG.device)

    # Init
    ACC_TF_TF, F1_TF_TF, AUC_TF_TF = [], [], []
    ACC_TT_TF, F1_TT_TF, AUC_TT_TF = [], [], []
    ACC_TF_TT, F1_TF_TT, AUC_TF_TT = [], [], []

    for i in range(15):
        # Sampling and store images
        print("----> Sampling images")
        runner.sample()

        # Load generated data
        x_gen = pd.read_csv(f"{ARGS.image_folder}/samples.csv", sep=',', header=None).to_numpy()
        y = pd.read_csv(f"{ARGS.image_folder}/samples_label.csv", sep=',', header=None).to_numpy()

        # Unscale MaxAbs
        scales = np.load(f'./src/generation/ddim/results/{CONFIG.data.dataset_frechet.lower()}_landmark_scales.npy', allow_pickle=True)
        x_gen = x_gen*scales

        train, val, test = split_and_scale_datasets(x_gen, y, X_test, y_test, scaler_type="standard")

        # Build fake data loaders
        train_loader_fake, val_loader_fake, test_loader = build_loaders(train, 
                                                                        val, 
                                                                        test,
                                                                        config=CONFIG_CLS)

        # Weights
        TISSUE_WEIGHTS = get_class_weights(train_loader_fake, val_loader_fake)

        ########## 3. Load classifier #########
        cls = TissuePredictor(CONFIG_CLS, CONFIG.data.dataset_frechet.lower())

        ########## 4. Train classifier #########
        # Model training
        cls.train(train_loader_fake, 
                    val_loader_fake,  
                    verbose=2, 
                    class_weights=TISSUE_WEIGHTS)

        # Test on fake val data
        acc_val, f1_score_val, auc_val, cm_val = cls.test(val_loader_fake, output='metric')

        ACC_TF_TF.append(acc_val)
        F1_TF_TF.append(f1_score_val)
        AUC_TF_TF.append(auc_val)

        # Test on true test data
        acc_test, f1_score_test, auc_test, cm_test = cls.test(test_loader, output='metric')

        ACC_TF_TT.append(acc_test)
        F1_TF_TT.append(f1_score_test)
        AUC_TF_TT.append(auc_test)
        print('acc_train_fake_test_true:', acc_test)

        ########## 5. Test pretrained classifier on generated data #########

        # Path where pretrained weights are stored    
        path_model = f'/home/alacan/scripts/classification/landmarks/results/model_{CONFIG.data.dataset_frechet.lower()}.pth'
        # Instantiate model
        cls = TissuePredictor(CONFIG_CLS, CONFIG.data.dataset_frechet.lower())
        # Load pretrained model
        cls.model.load_state_dict(torch.load(path_model, map_location=CONFIG.device2))

        acc_train_true_test_fake, f1_score_train_true_test_fake, auc_train_true_test_fake, cm_train_true_test_fake = cls.test(train_loader_fake, output='metric')
        print('acc_train_true_test_fake:', acc_train_true_test_fake)
        ACC_TT_TF.append(acc_train_true_test_fake)
        F1_TT_TF.append(f1_score_train_true_test_fake)
        AUC_TT_TF.append(auc_train_true_test_fake)

    
    # Save results
    d = {
    'acc_train_fake_test_fake': ACC_TF_TF,
    'f1_score_train_fake_test_fake': F1_TF_TF,
    'auc_train_fake_test_fake': AUC_TF_TF,

    'acc_train_fake_test_true':  ACC_TF_TT,
    'f1_score_train_fake_test_true': F1_TF_TT,
    'auc_train_fake_test_true': AUC_TF_TT,

    'acc_train_true_test_fake': ACC_TT_TF,
    'f1_score_train_true_test_fake': F1_TT_TF,
    'auc_train_true_test_fake': AUC_TT_TF,
    }

    df_res = pd.DataFrame(data=d,
                        columns= d.keys())
    # Save
    df_res.to_csv(f'./src/generation/ddim/results/reverse_validation_{CONFIG.data.dataset_frechet}_eta_{ARGS.eta}.csv')
    print(f"----> End. ")

#This will call the function main() and when main finishes, it will exit giving the system the return code that is the result of main()
#If you execute a Python script directly, __name__ is set to "__main__", but if you import it from another script, it is not.
if __name__ == "__main__":
    sys.exit(main())
    pass