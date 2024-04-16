import os
import sys
import logging
import time
import glob
import csv
import datetime

import numpy as np
import pandas as pd
import tqdm
import torch
import torch.utils.data as data

from src.generation.ddim.models.diffusion_ddim import ModelDDIM
from src.generation.ddim.models.ema import EMAHelper
from src.generation.ddim.functions import count_parameters, get_optimizer, get_scheduler, get_scheduler_warmup
from src.generation.ddim.functions.losses import loss_registry
from src.generation.ddim.datasets import get_dataset

###### Additions
from torch.cuda.amp import autocast, GradScaler
###### Metrics
from src.metrics.precision_recall import compute_prdc
from src.metrics.aats import compute_AAts
from src.metrics.frechet import compute_frechet_distance_score
from src.metrics.correlation_score import gamma_coeff_score


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    '''
        beta_schedule: A string that specifies which schedule to use for the beta values. There are five options:
            "quad": A quadratic schedule where the beta values start at beta_start and increase quadratically until they reach beta_end over the course of num_diffusion_timesteps.
            "linear": A linear schedule where the beta values start at beta_start and increase linearly until they reach beta_end over the course of num_diffusion_timesteps.
            "const": A constant schedule where the beta values are equal to beta_end for all num_diffusion_timesteps.
            "jsd": A schedule inspired by Jensen-Shannon Divergence, where the beta values start at 1/num_diffusion_timesteps and increase until they reach 1 over the course of num_diffusion_timesteps.
            "sigmoid": A schedule where the beta values follow a sigmoid curve that starts at beta_start and ends at beta_end over the course of num_diffusion_timesteps.
        beta_start: The starting value of beta.
        beta_end: The ending value of beta.
        num_diffusion_timesteps: The number of diffusion timesteps, which determines the length of the array of beta values returned by the function. 
    '''

    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.model = None

        # Parallelization
        if len(device) > 1:
            self.device = torch.device('cuda')
        elif len(device)==1:
            self.device = torch.device(device[0])
        else:
            print("Error with specified device:", self.device)

        print("Init on device (s): ", self.device)

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0) # alphas
        alphas_cumprod_prev = torch.cat( 
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = ( 
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log() 
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log() 

    def train(self):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(config)

        train_loader = data.DataLoader(
                                    dataset,
                                    batch_size=config.training.batch_size,
                                    shuffle=True,
                                    num_workers=config.data.num_workers,
                                    pin_memory=True,
                                    prefetch_factor=2,
                                    persistent_workers=config.data.persistent_workers)
        
        self.config.n_batchs = len(train_loader)
        print(f"NUMBER OF BATCHES: {self.config.n_batchs}")

        #--Load Model--
        self.model = ModelDDIM(config)

        #--Nb Parameters
        self.nb_params = count_parameters(self.model)
        print(f"NUMBER OF PARAMETERS: Model has {self.nb_params} to train.")

        # Parallelization
        print("config.device", config.device)
        if len(config.device) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config.device)
            self.device = torch.device('cuda')
        # To GPU
        self.model = self.model.to(self.device)

        #--Scaler for Mixed Precision--
        scaler = GradScaler()

        #--Optimizer--
        optimizer = get_optimizer(self.config, self.model.parameters())

        #--Scheduler--
        if self.config.optim.use_scheduler:
            lr_scheduler = get_scheduler(self.config, optimizer)

        #--Warmup--
        if self.config.optim.use_scheduler and self.config.optim.use_warmup:
            print("Using WARMUP")
            warmup_scheduler = get_scheduler_warmup(self.config, optimizer)

        #--EMA Helper--
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            self.model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
        else:
            # delete loss csv
            if os.path.exists(os.path.join(self.args.log_path, "loss.csv")):
                os.remove(os.path.join(self.args.log_path, "loss.csv"))
            # save csv
            with open(os.path.join(self.args.log_path, "loss.csv"), "a") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "loss", "avg", "epoch_time", "total_time", "grad_norm_down_layer", "grad_norm_mid_layer", "grad_norm_last_layer", "lr"])
                f.close()

            # delete metrics csv
            if os.path.exists(os.path.join(self.args.log_path, "metrics.csv")):
                os.remove(os.path.join(self.args.log_path, "metrics.csv"))
            # save csv
            with open(os.path.join(self.args.log_path, "metrics.csv"), "a") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "precision", "recall", "density", "coverage", "aats", "frechet"])
                f.close()

        epoch_tracker = -1

        train_start_time = time.time()
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                step += 1

                x = x.to(self.device)
                y = y.to(self.device)
                
                e = torch.randn_like(x).to(self.device)
                b = self.betas.to(self.device)

                # for each sample in a batch we select a random timestep to train all timesteps at once
                t = torch.randint( 
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,) 
                ).to(self.device)
                # we also select all the opposite timsteps (antithetic sampling) 
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n] # concatenation of all timesteps
                '''
                The reason for concatenating the t tensor with self.num_timesteps - t - 1 
                is to ensure that the selected time steps are spread out across the input sequence. 
                This helps to prevent the model from overfitting to specific parts of the input sequence.
                '''
                
                # Loss is computed under autocast env
                loss, avg = loss_registry[config.model.type](self.model, x, t, e, b, y)

                # estimate remaining time
                remaining_epochs = self.config.training.n_epochs - epoch
                time_per_epoch = (time.time() - train_start_time) / (epoch + 1)
                remaining_time = remaining_epochs * time_per_epoch
                remaining_time_str = str(datetime.timedelta(seconds=remaining_time))

                logging.info(
                    f"epoch: {epoch}, step: {step}, loss: {loss.item():.4f}, avg: {avg.item():.4f}, time: {data_time / (i+1):.3f}, total time: {time.time() - train_start_time:.3f}, remaining time: {remaining_time_str}"
                )

                optimizer.zero_grad()

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward()
                #loss.backward()

                try:
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                #optimizer.step()
                
                # Updates the scale for next iteration.
                scaler.update()

                # pytorch_warmup https://github.com/Tony-Y/pytorch_warmup
                if self.config.optim.use_scheduler and self.config.optim.use_warmup:
                    if i < len(train_loader)-1:
                        with warmup_scheduler.dampening():
                            pass

                 # Add loss to csv if first of epoch
                if epoch > epoch_tracker:
                    epoch_tracker = epoch

                    csv_info = []
                
                    csv_info.append(epoch)
                    csv_info.append("%.3f" % loss.item())
                    csv_info.append("%.4f" % avg.item())
                    csv_info.append("%.3f" % (time.time() - data_start))
                    csv_info.append("%.2f" % (time.time() - train_start_time))

                    # New model architecture
                    csv_info.append("0")
                    csv_info.append("0")
                    csv_info.append("0")

                    # Learning rate
                    csv_info.append("%.10f" % optimizer.param_groups[0]['lr'])


                    # save csv
                    with open(os.path.join(self.args.log_path, "loss.csv"), "a") as f:
                        w = csv.writer(f)
                        w.writerow(csv_info)
                        f.close()

                if self.config.model.ema:
                    ema_helper.update(self.model)

                if step % self.config.training.snapshot_freq == 0 or epoch >= self.config.training.n_epochs - 1:
                    states = [
                        self.model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    print("Saved checkpoint at step {}".format(step) + "at path" + self.args.log_path + "ckpt_{}.pth".format(step))

                data_start = time.time()

                #pytorch_warmup https://github.com/Tony-Y/pytorch_warmup
                if self.config.optim.use_scheduler and self.config.optim.use_warmup:
                    with warmup_scheduler.dampening():
                        lr_scheduler.step()
                        print("Learning rate decayed to {}".format(optimizer.param_groups[0]["lr"]))
                elif self.config.optim.use_scheduler:
                    lr_scheduler.step()

            # Compute metrics
            if epoch % self.config.training.metrics_step == 0 and epoch>0:
                self.checkpoints_metrics(epoch, train_loader)

    def checkpoints_metrics(self, epoch, train_loader):
        """ 
        Compute metrics at given checkpoints during training.
        """

        self.model.eval()
        ####### SAMPLING #######
        #START SAMPLING INFO
        # delete csv
        if os.path.exists(os.path.join(self.args.image_folder, "sampling_info.csv")):
            os.remove(os.path.join(self.args.image_folder, "sampling_info.csv"))
        # save csv
        with open(os.path.join(self.args.image_folder, "sampling_info.csv"), "a") as f:
            w = csv.writer(f)
            w.writerow(["total_time", "total_sample", "batch_size"])
            f.close()

        start_time = time.time()
        # CSV to store generated samples
        csv_filename = '/samples.csv'
        csv_filename_label = '/samples_label.csv'
        csv_ = None
        csv_label = None
        img_id = 0

        # delete csv if existing
        if os.path.exists(os.path.join(self.args.image_folder, csv_filename)):
            os.remove(os.path.join(self.args.image_folder, csv_filename))
       
        print(f"starting from image {img_id}")
        total_n_samples = self.config.sampling.total_samples #how many batches to generate
        n_rounds = int(np.ceil((total_n_samples - img_id) / self.config.sampling.batch_size)) #combien de rounds pour gen les images

        last_save = img_id
        with torch.no_grad():
            for i, (_, y) in tqdm.tqdm(enumerate(train_loader), desc="Generating image samples for metrics evaluation.", total=n_rounds):
                n = y.size(0)
                x = torch.randn(
                    n,
                    self.config.data.image_size, 
                    device=self.device,
                )
                n_classes = self.config.data.num_classes
                y = y.to(self.device)
                x = self.sample_image(x, self.model, y=y) # generating
                x = x.cpu().numpy() 
                x = x.reshape(x.shape[0], -1) 

                try:
                    csv_ = np.concatenate((csv_, x), axis=0)
                except:
                    csv_ = x

                # save labels
                try:
                    csv_label = np.concatenate((csv_label, y.cpu().numpy()), axis=0)
                except:
                    csv_label = y.cpu().numpy()
                
                if img_id - last_save >= 1000 or img_id == total_n_samples - 1 or i >= n_rounds - 1:
                    df = pd.DataFrame(csv_)
                    print("self.args.image_folder:", self.args.image_folder)
                    df.to_csv(f"{self.args.image_folder}/samples.csv", index=False, header=False)
                    last_save = img_id
                    df = pd.DataFrame(csv_label)
                    df.to_csv(f"{self.args.image_folder}/samples_label.csv", index=False, header=False)

                    print(f"saved {img_id} images")

                img_id += n

            batch_size = self.config.sampling.batch_size
            total_sample = self.config.sampling.total_samples
            total_time = time.time() - start_time

            csv_info = []
            csv_info.append("%.3f" % total_time)
            csv_info.append(total_sample)
            csv_info.append(batch_size)

            # save csv
            with open(os.path.join(self.args.image_folder, "sampling_info.csv"), "a") as f:
                w = csv.writer(f)
                w.writerow(csv_info)
                f.close()

            ####### METRICS #######
            # Retrieve full dataset and labels
            true_samples = []
            for b, _ in train_loader:
                true_samples.append(b)
            true_samples = torch.cat(true_samples, 0).numpy()

            # Generated samples
            print("------> Load fake samples")
            PATH_SAMPLES = f"{self.args.image_folder}/samples.csv"
            fake_samples = pd.read_csv(PATH_SAMPLES, sep =',', header=None)
            fake_samples = fake_samples.to_numpy()

            print("------> Compute metrics")
            # Precision/recall
            prec, recall, dens, cov = compute_prdc(true_samples, fake_samples, self.config.data.nb_nn)
            print("k=10 | Prec:{} Recall:{} Density:{} Coverage:{}".format(round(prec,4), round(recall,4), round(dens,4), round(cov,4)))
            # Sample random data
            idx = np.random.choice(len(true_samples), 2048, replace=False)
            # Adversarial accuracy
            _, _, self.adversarial = compute_AAts(real_data=true_samples[idx], fake_data=fake_samples[idx])
            print(f"AATS: {round(self.adversarial, 4)}")

            self.precision, self.recall, self.density, self.coverage = prec, recall, dens, cov

            # Frechet
            self.frechet = compute_frechet_distance_score(true_samples, fake_samples, dataset=self.config.data.dataset_frechet, device=self.config.device2, to_standardize=True)

            ####### SAVE METRICS #######
            csv_metrics = []
            csv_metrics.append(epoch)
            csv_metrics.append("%.3f" % prec)
            csv_metrics.append("%.3f" % recall)
            csv_metrics.append("%.3f" % dens)
            csv_metrics.append("%.3f" % cov)
            csv_metrics.append("%.3f" % self.adversarial)
            csv_metrics.append("%.3f" % self.frechet)
            
            # save csv 
            with open(os.path.join(self.args.log_path, "metrics.csv"), "a") as f:
                w = csv.writer(f)
                w.writerow(csv_metrics)
                f.close()

        # Back to training
        self.model.train()


    def sample(self):
        
        if self.model is None:
            if self.config.model.model == "ddim":
                self.model = ModelDDIM(self.config)
                # To GPU
                self.model = self.model.to(self.device)

        start_time = time.time()

        # loading last checkpoints if no pretraining
        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                                    map_location=self.device)
                print("Loaded checkpoint ckpt.pth")
            else:
                states = torch.load(os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.device
                )
                print("Loaded checkpoint ckpt_{}.pth".format(self.config.sampling.ckpt_id))
            
            # Parallelization
            if len(self.config.device) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.config.device)
                self.device = torch.device('cuda')
                # To GPU
                self.model = self.model.to(self.device)

            self.model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(self.model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(self.model)
            else:
                ema_helper = None

        self.model.eval()

        # Sampling images for metric computations
        if self.args.fid:
            self.sample_fid(self.model, self.config.sampling.compute_metrics) #on pourra utiliser notre propre distance de frechet ici
            #START SAMPLING INFO
            #delete csv
            if os.path.exists(os.path.join(self.args.image_folder, "sampling_info.csv")):
                os.remove(os.path.join(self.args.image_folder, "sampling_info.csv"))
            #save csv
            with open(os.path.join(self.args.image_folder, "sampling_info.csv"), "a") as f:
                w = csv.writer(f)
                w.writerow(["total_time", "total_sample", "batch_size"])
                f.close()

            config = self.config
            batch_size = config.sampling.batch_size
            total_sample = config.sampling.total_samples
            total_time = time.time() - start_time

            csv_info = []
                    
            csv_info.append("%.3f" % total_time)
            csv_info.append(total_sample)
            csv_info.append(batch_size)

            #save csv
            with open(os.path.join(self.args.image_folder, "sampling_info.csv"), "a") as f:
                w = csv.writer(f)
                w.writerow(csv_info)
                f.close()
            
        # Sampling diffusion steps as sequences of a sample
        elif self.args.sequence:
            self.sample_sequence(self.model)
        # Sampling noise predictions at each time step
        elif self.args.noisepredi:
            self.sample_noisepredi(self.model)
        else:
            raise NotImplementedError("Sample procedure not defined")


    def sample_noisepredi(self, model):
        config = self.config
        csv_filename = "noise_predis.csv"
        # delete csv if it exists
        if os.path.exists(os.path.join(self.args.image_folder, csv_filename)):
            os.remove(os.path.join(self.args.image_folder, csv_filename))
        
        dataset, test_dataset = get_dataset(config)
        
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=config.data.persistent_workers)
        

        avg_list = []
        with torch.no_grad():
            for epoch in tqdm.tqdm(
                range(config.diffusion.num_diffusion_timesteps), desc="Predicting Average Noise for each timestep."
            ):

                avg_value = 0

                for i, (x, y) in enumerate(train_loader):
                    n = x.size(0)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    e = torch.randn_like(x)
                    b = self.betas

                    t = torch.full((n,), epoch, dtype=torch.long, device=self.device)
                    # Loss is computed under autocast env
                    loss, avg = loss_registry[config.model.type](model, x, t, e, b, y)

                    avg_value += avg.item() * n

                avg_value /= len(train_loader.dataset)
                avg_list.append(avg.item())

        with open(os.path.join(self.args.image_folder, csv_filename), "a") as f:
            w = csv.writer(f)
            w.writerow(avg_list)
            f.close()

    # Sampling for metrics
    def sample_fid(self, model, compute_metrics:bool=False):
        config = self.config
        csv_filename = '/samples.csv'
        csv_filename_label = '/samples_label.csv'
        csv = None
        csv_label = None
        img_id = 0
       
        print(f"starting from image {img_id}")
        total_n_samples = config.sampling.total_samples

        # load real data
        dataset, test_dataset = get_dataset(config)
        
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=config.data.persistent_workers)
        
        # how many batches to generate
        total_n_samples = dataset.__len__()
        n_rounds = int(np.ceil((total_n_samples - img_id) / config.sampling.batch_size)) 

        last_save = img_id
        with torch.no_grad():

            for i, (_, y) in tqdm.tqdm(enumerate(train_loader), desc="Generating image samples for metrics evaluation.", total=n_rounds):

                n = y.size(0)

                x = torch.randn(
                    n,
                    config.data.image_size, 
                    device=self.device,
                )

                n_classes = config.data.num_classes 
                y = y.to(self.device)

                x = self.sample_image(x, model, y=y) # generation
                x = x.cpu().numpy() 
                x = x.reshape(x.shape[0], -1) 

                try:
                    csv = np.concatenate((csv, x), axis=0)
                except:
                    csv = x

                # save labels
                try:
                    csv_label = np.concatenate((csv_label, y.cpu().numpy()), axis=0)
                except:
                    csv_label = y.cpu().numpy()
                
                # save
                if img_id - last_save >= 1000 or img_id == total_n_samples - 1 or i >= n_rounds - 1:
                    df = pd.DataFrame(csv)
                    print("self.args.image_folder:", self.args.image_folder)
                    df.to_csv(f"{self.args.image_folder}/samples.csv", index=False, header=False)
                    last_save = img_id

                    df = pd.DataFrame(csv_label)
                    df.to_csv(f"{self.args.image_folder}/samples_label.csv", index=False, header=False)

                    print(f"saved {img_id} images")

                img_id += n

        if compute_metrics:
            # Compute similarity metrics on fake and true data
            print("------> Load fake samples")
            # Generated samples
            PATH_SAMPLES = f"{self.args.image_folder}/samples.csv"
            fake_samples = pd.read_csv(PATH_SAMPLES, sep =',', header=None)
            fake_samples = fake_samples.to_numpy()

            # Retrieve full dataset and labels
            true_samples = torch.stack([dataset[i][0] for i in range(len(dataset))]).numpy()
            true_labels = torch.stack([dataset[i][1] for i in range(len(dataset))]).numpy()

            print("------> Compute metrics")
            # Precision/recall
            prec_10, recall_10, dens_10, cov_10 = compute_prdc(true_samples, fake_samples, 10)
            print("k=10 | Prec:{} Recall:{} Density:{} Coverage:{}".format(round(prec_10,4), round(recall_10,4), round(dens_10,4), round(cov_10,4)))
            # Sample random data
            idx = np.random.choice(len(true_samples), 2048, replace=False)
            # Adversarial accuracy
            _, _, self.adversarial = compute_AAts(real_data=true_samples[idx], fake_data=fake_samples[idx])

            self.precision, self.recall, self.density, self.coverage = prec_10, recall_10, dens_10, cov_10

            # Frechet
            self.frechet = compute_frechet_distance_score(true_samples, fake_samples, dataset=self.config.data.dataset_frechet, device=self.config.device2, to_standardize=True)
            
            # Correlation score
            self.correlation_score = gamma_coeff_score(true_samples, fake_samples)
            # End        

    def sample_sequence(self, model):
        """
        Generating at each diffusion step.
        """
        config = self.config

        x = torch.randn(
            1024,
            config.data.image_size,
            device=self.device,
        )

        # Random classes
        n_classes = config.data.num_classes
        y = torch.randint(0, n_classes, (1024, 1), device=self.device)
        y = torch.nn.functional.one_hot(y, num_classes=n_classes)
        y = y.squeeze(1)

        with torch.no_grad():
            x = self.sample_image(x, model, last=False, y=y)

        x0s = x[1]
        x = x[0]
        # save 6 intermediary steps
        steps_to_save = 6

        steps = np.linspace(0, len(x) - 1, steps_to_save).astype(int)
        steps_x0 = np.linspace(0, len(x0s) - 1, steps_to_save).astype(int)
        
        for i in tqdm.tqdm(range(steps_to_save)):

            csv = x[steps[i]]
            csv = csv.cpu().numpy()
            df = pd.DataFrame(csv)
            df.to_csv(f"{self.args.image_folder}/samples_seq_{str(i)}.csv", index=False, header=False)

            csv = x0s[steps_x0[i]]
            csv = csv.cpu().numpy() 
            df = pd.DataFrame(csv)
            df.to_csv(f"{self.args.image_folder}/samples_seq_x0_{str(i)}.csv", index=False, header=False)

            print(f"saved images for step {str(steps[i])}")

    def sample_interpolation(self, model):
        #TODO: sample interpolation
        pass

    def sample_image(self, x, model, last=True, y=None):
        """
        Global function for generation.
        """
        if self.args.sample_type == "generalized":
            # uniform skip
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
                
            # quadratic skip
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from src.generation.ddim.functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, y=y, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from src.generation.ddim.functions.denoising import ddpm_steps

            # DDPM STEPS
            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError

        # if last = True, we keep only the last denoised sample, otherwise we take all the intermediary diffusion steps
        if last:
            x = x[0][-1]

        return x

    def test(self):
        pass

    def sample_cond_on_y(self, labels=None):
        """
        Generation with covariates conditioning.
        """
        # Load architecture
        if self.model is None:
            if self.config.model.model == "ddim":
                self.model = ModelDDIM(self.config)
                self.model = self.model.to(self.device) # To GPU

        start_time = time.time()
        # Load model weights
        states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                            map_location=self.device)
        
        self.model.load_state_dict(states[0], strict=True)
        print(f"Loaded checkpoint: {os.path.join(self.args.log_path, 'ckpt.pth')}")

        # Load EMA
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(self.model)
        else:
            ema_helper = None

        self.model.eval()

        # Sampling
        csv_filename = '/samples.csv'
        csv_filename_label = '/samples_label.csv'
        csv_ = None
        csv_label = None
        img_id = 0
       
        print(f"starting from image {img_id}")
        # Same number of generated samples as labels
        total_n_samples = labels.shape[0]
        print(f"Number of samples to generate: {total_n_samples}")
        
        # How many batches to generate
        n_rounds = int(np.ceil((total_n_samples - img_id) / self.config.sampling.batch_size))
        last_save = img_id
        
        with torch.no_grad():
            for i, step in tqdm.tqdm(enumerate(range(0, len(labels), self.config.sampling.batch_size)), desc="Sampling...", total=n_rounds):
                y = labels[step:self.config.sampling.batch_size+step] # batch of labels
                n = y.size(0)
                x = torch.randn(n, self.config.data.image_size, device=self.device,)
                n_classes = self.config.data.num_classes # y condition: one hot label 
                y = y.to(self.device)

                x = self.sample_image(x, self.model, y=y) 
                x = x.cpu().numpy() 
                x = x.reshape(x.shape[0], -1) 

                try:
                    csv_ = np.concatenate((csv_, x), axis=0)
                except:
                    csv_ = x

                # save labels
                try:
                    csv_label = np.concatenate((csv_label, y.cpu().numpy()), axis=0)
                except:
                    csv_label = y.cpu().numpy()
                
                # save
                if img_id - last_save >= 1000 or img_id == total_n_samples - 1 or i >= n_rounds - 1:
                    df = pd.DataFrame(csv_)
                    print("self.args.image_folder:", self.args.image_folder)
                    df.to_csv(f"{self.args.image_folder}/samples.csv", index=False, header=False)
                    last_save = img_id

                    df = pd.DataFrame(csv_label)
                    df.to_csv(f"{self.args.image_folder}/samples_label.csv", index=False, header=False)

                    print(f"saved {img_id} images")

                img_id += n

        # START SAMPLING INFO
        # delete csv
        if os.path.exists(os.path.join(self.args.image_folder, "sampling_info.csv")):
            os.remove(os.path.join(self.args.image_folder, "sampling_info.csv"))
        # save csv
        with open(os.path.join(self.args.image_folder, "sampling_info.csv"), "a") as f:
            w = csv.writer(f)
            w.writerow(["total_time", "total_sample", "batch_size"])
            f.close()

        config = self.config
        batch_size = config.sampling.batch_size
        total_sample = config.sampling.total_samples
        total_time = time.time() - start_time

        csv_info = []
                
        csv_info.append("%.3f" % total_time)
        csv_info.append(total_sample)
        csv_info.append(batch_size)

        # save csv
        with open(os.path.join(self.args.image_folder, "sampling_info.csv"), "a") as f:
            w = csv.writer(f)
            w.writerow(csv_info)
            f.close()

        return csv_, csv_label


    def sample_sequence_from_y(self, y, steps_to_save:int=6):
        """
        Generation of intermediary diffusion steps with covariate conditioning.
        """
        # Load architecture
        if self.model is None:
            if self.config.model.model == "ddim":
                self.model = ModelDDIM(self.config)
                self.model = self.model.to(self.device) # To GPU

        
        # Load model weights
        states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                            map_location=self.device)
        
        self.model.load_state_dict(states[0], strict=True)
        print(f"Loaded checkpoint: {os.path.join(self.args.log_path, 'ckpt.pth')}")

        # Load EMA
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(self.model)
        else:
            ema_helper = None

        self.model.eval()

        # Config
        config = self.config
        # N samples
        nb_samples = y.shape[0]
        # Input noise
        x = torch.randn(
            nb_samples,
            config.data.image_size,
            device=self.device)
        y = y.to(self.device)

        start_time = time.time()
        with torch.no_grad():
            x = self.sample_image(x, self.model, last=False, y=y)
        end_time = time.time()
        self.sampling_time = end_time - start_time
        print('sampling_time:', self.sampling_time)

        x0s = x[1] # predicted x0s
        x = x[0] # denoised x
        
        steps = np.linspace(0, len(x) - 1, steps_to_save).astype(int)
        steps_x0 = np.linspace(0, len(x0s) - 1, steps_to_save).astype(int)
        
        for i in tqdm.tqdm(range(steps_to_save)):
            csv = x[steps[i]]
            csv = csv.cpu().numpy() 
            df = pd.DataFrame(csv)
            df.to_csv(f"{self.args.image_folder}/samples_seq_{str(i)}.csv", index=False, header=False)

            csv = x0s[steps_x0[i]]
            csv = csv.cpu().numpy() 
            df = pd.DataFrame(csv)
            df.to_csv(f"{self.args.image_folder}/samples_seq_x0_{str(i)}.csv", index=False, header=False)

            print(f"saved images for step {str(steps[i])}")

