# Imports
import os
import sys
import random
import time as t
import datetime
import torch
import torch.nn as nn
import torch.optim
import torch.autograd as autograd
import torch.nn.utils.spectral_norm as SN
from src.generation.gans.utils import *
from torch.cuda.amp import autocast, GradScaler

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)


def apply_SN(layer, apply: bool = False):
    """
    Apply spectral normalization on layers.
    """
    if apply:
        return SN(layer)
    else:
        return layer


class Generator(nn.Module):
    """
    Generator class
    """

    def __init__(self, latent_dim: int,
                 embedd_dim: int,
                 hidden_dim1: int,
                 hidden_dim2: int,
                 output_dim: int = None,
                 hidden_dim3: int = None,
                 vocab_size: int = None,
                 activation_func: str = 'relu',
                 negative_slope: float = 0.1):
        super().__init__()

        """Parameters:
            latent_dim (int): dimension of latent noise vector z
            embedd_dim (int): dimension of categorical embedded variables (cancer types or tissue types)
            numerical_dim (int): dimension of numerical variables
            hidden_dim1 (int): dimension of 1st hidden layer
            hidden_dim2 (int): dimension of 2nd hidden layer
            hidden_dim3 (int): dimension of 3rd hidden layer
            output_dim (int): dimension of generated data (nb of genes)
            vocab_size (int): size of vocabulary for cancer/tissue embeddings
            activation_func (str): activation function between hidden layers (either 'relu' or 'leaky_relu'). Default 'relu'.
            negative_slope (float): slope of leaky relu activation. Default 0.1.
        """

        # Dimensions
        self.vocab_size = vocab_size
        self.embedd_dim = embedd_dim
        self.latent_dim = latent_dim
        # We concatenate conditional covariates with gene expression variables
        self.input_dim = latent_dim + self.embedd_dim * self.vocab_size
        # Layers params
        self.output_dim = output_dim
        self.hidden_dim1 = hidden_dim1 
        self.hidden_dim2 = hidden_dim2 
        self.hidden_dim3 = hidden_dim3 

        # Embedding layers for tissue type/cancer types
        self.embedding = nn.Embedding(self.vocab_size, self.embedd_dim)

        # Activation function
        self.negative_slope = negative_slope
        if activation_func.lower() == 'leaky_relu':
            self.activation_func = nn.LeakyReLU(self.negative_slope)
        elif activation_func.lower() == 'relu':
            self.activation_func = nn.ReLU()

        # Batch norm layers
        self.bn1 = nn.BatchNorm1d(self.hidden_dim1)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim2)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim3)

        # Linear layers without batch norm in WGAN-GP
        self.proj1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.proj2 = nn.Linear(self.hidden_dim1 + self.embedd_dim * self.vocab_size, self.hidden_dim2)
        self.proj3 = nn.Linear(self.hidden_dim2 + self.embedd_dim * self.vocab_size, self.hidden_dim3)
        # Output (unconstrained, i.e no TanH/Sigmoid)
        self.proj_output = nn.Linear(self.hidden_dim3 + self.embedd_dim * self.vocab_size, self.output_dim)

    def forward(self, x: torch.tensor, y: torch.tensor):
        """ Main function to generate from input noise vector.
        ----
        Parameters:
            x (torch.tensor): input noise vector
            y (torch.tensor): input categorical conditions to embed
        Returns:
            (torch.tensor): generated data
        """
        y = y.argmax(dim=1) if y.dim() > 1 else y.long()
        y = self.embedding(y)  # Embedding for tissue types/cancer types
        # Concatenate all variables (expression data and conditions)
        x = torch.cat((x, y.flatten(start_dim=1)), 1)
        x = self.activation_func(self.bn1(self.proj1(x))) 
        x = torch.cat((x, y.flatten(start_dim=1)), 1) # residual connexion with conditioning variable
        x = self.activation_func(self.bn2(self.proj2(x)))
        x = torch.cat((x, y.flatten(start_dim=1)), 1) # residual connexion with conditioning variable
        x = self.activation_func(self.bn3(self.proj3(x)))
        x = torch.cat((x, y.flatten(start_dim=1)), 1) # residual connexion with conditioning variable
        # Output (unconstrained)
        x = self.proj_output(x)

        return x
    
    def gen_from_embedding(self, x: torch.tensor, y: torch.tensor):
        """ Main function to generate from input noise vector.
        ----
        Parameters:
            x (torch.tensor): input noise vector
            y (torch.tensor): input embedding of categorical condition
        Returns:
            (torch.tensor): generated data
        """
        # Concatenate all variables (expression data and conditions)
        x = torch.cat((x, y.flatten(start_dim=1)), 1)
        x = self.activation_func(self.bn1(self.proj1(x))) 
        x = torch.cat((x, y.flatten(start_dim=1)), 1) # residual connexion with conditioning variable
        x = self.activation_func(self.bn2(self.proj2(x)))
        x = torch.cat((x, y.flatten(start_dim=1)), 1) # residual connexion with conditioning variable
        x = self.activation_func(self.bn3(self.proj3(x)))
        x = torch.cat((x, y.flatten(start_dim=1)), 1) # residual connexion with conditioning variable
        # Output (unconstrained)
        x = self.proj_output(x)

        return x


class Discriminator(nn.Module):
    """ Critic class
    """

    def __init__(self, x_dim: int,
                 embedd_dim: int,
                 hidden_dim1: int,
                 hidden_dim2: int,
                 output_dim: int,
                 vocab_size: int = None,
                 activation_func: str = 'relu',
                 negative_slope: float = 0.1,
                 spectral_norm:bool=False):
        super().__init__()
        """Parameters:
            x_dim (int): dimension of data (nb of genes).
            embedd_dim (int): dimension of categorical embedded variables.
            numerical_dim (int): dimension of numerical variables.
            hidden_dim1 (int): dimension of 1st hidden layer.
            hidden_dim2 (int): dimension of 2nd hidden layer.
            output_dim (int): output dimension.
            vocab_size (int): vocabulary size for cancer/tissue embeddings.
            activation_func (str): activation function between hidden layers (either 'relu' or 'leaky_relu'). Default 'relu'.
            negative_slope (float): slope of leaky relu activation. Default 0.1.
        """
        # Layers params
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        # Dimensions
        self.vocab_size = vocab_size
        self.embedd_dim = embedd_dim
        # We concatenate conditional covariates with gene expression variables
        self.input_dim = x_dim  + self.embedd_dim * self.vocab_size
        # Embedding layers for tissue type/cancer types
        self.embedding = nn.Embedding(self.vocab_size, self.embedd_dim)

        # Activation function
        self.negative_slope = negative_slope
        if activation_func.lower() == 'leaky_relu':
            self.activation_func = nn.LeakyReLU(self.negative_slope)
        elif activation_func.lower() == 'relu':
            self.activation_func = nn.ReLU()

        # Linear layers
        self.proj1 = apply_SN(nn.Linear(self.input_dim, self.hidden_dim1), spectral_norm)
        self.proj2 = apply_SN(nn.Linear(self.hidden_dim1 + self.embedd_dim * self.vocab_size, self.hidden_dim2), spectral_norm)
        self.proj_output = apply_SN(nn.Linear(self.hidden_dim2 + self.embedd_dim * self.vocab_size, self.output_dim), spectral_norm)

    def forward(self, x: torch.tensor, y: torch.tensor):
        """ Main function to discriminate input data.
        ----
        Parameters:
            x (torch.tensor): input data.
            y (torch.tensor): input conditional categorical data.
        Returns:
            (torch.tensor): predicted score (unconstrained)
        """
        y = y.argmax(dim=1) if y.dim() > 1 else y.long()
        y = self.embedding(y)  # Embedding for tissue types/cancer types
        # Concatenate all variables (expression data, numerical covariates and
        # flattened embeddings)
        x = torch.cat((x, y.flatten(start_dim=1)), 1) 
        x = self.activation_func(self.proj1(x))
        x = torch.cat((x, y.flatten(start_dim=1)), 1) # residual connexion with conditioning variable
        x = self.activation_func(self.proj2(x))
        x = torch.cat((x, y.flatten(start_dim=1)), 1) # residual connexion with conditioning variable
        x = self.proj_output(x)
        return x  # Linear output score for the Critic


class WGAN(object):
    """
    Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP) class.
    """

    def __init__(self, config: dict):
        """ Parameters:
            config (dict): model architecture dictionary.
        """
        # Set architecture
        self.latent_dim = config['latent_dim']
        self.x_dim = config['x_dim']
        self.embedded_dim = config['embedded_dim']
        self.hidden_dim1_g = config['hidden_dim1_g']
        self.hidden_dim2_g = config['hidden_dim2_g']
        self.hidden_dim3_g = config['hidden_dim3_g']
        self.hidden_dim1_d = config['hidden_dim1_d']
        self.hidden_dim2_d = config['hidden_dim2_d']
        self.output_dim = config['output_dim']
        self.vocab_size = config['vocab_size']
        self.epochs_checkpoints = "/home/daniilf/rna-diffusion/checkpoints"
        self.nb_nn = config['nb_nn']

        #  use gpu if available
        self.device = config['device']
        if 'device_frechet' in config.keys():
            self.device_frechet = config['device_frechet']

        self.dataset = config['dataset']

        # Set generator and critic models with init arguments
        self.G = Generator(
            self.latent_dim,
            self.embedded_dim,
            self.hidden_dim1_g,
            self.hidden_dim2_g,
            output_dim=self.x_dim,
            vocab_size=self.vocab_size,
            hidden_dim3=self.hidden_dim3_g,
            activation_func=config['activation'],
            negative_slope=config['negative_slope']).to(
            self.device)

        self.D = Discriminator(
            self.x_dim,
            self.embedded_dim,
            self.hidden_dim1_d,
            self.hidden_dim2_d,
            self.output_dim,
            vocab_size=self.vocab_size,
            activation_func=config['activation'],
            negative_slope=config['negative_slope'],
            spectral_norm=config['sn']).to(
            self.device)

        self.checkpoint_prefix = None
        self.G_path = None
        self.D_path = None

        # Callbacks objects
        self.LossTracker = TrackLoss()

    def gradient_penalty(
            self,
            real_data: torch.tensor,
            fake_data: torch.tensor,
            categorical_vars: torch.tensor = None):
        """
        Compute gradient penalty.
        ----
        Parameters:
            real_data (torch.tensor): real data
            fake_data (torch.tensor): generated data
            categorical_vars (torch.tensor): conditional categorical variables
        Returns:
            gp (torch.tensor): gradient penalty i.e mean squared gradient norm on interpolations (||Grad[D(x_inter)]||2 - 1)^2
        """
        # Fixed batch size
        BATCH_SIZE = real_data.size(0)

        # Sample alpha from uniform distribution
        alpha = torch.rand(
            BATCH_SIZE,
            1,
            requires_grad=True,
            device=real_data.device)

        # Interpolation between real data and fake data.
        interpolation = torch.mul(alpha, real_data) + \
            torch.mul((1 - alpha), fake_data)

        # Discriminator forward pass
        disc_outputs = self.D(interpolation, categorical_vars)
        grad_outputs = torch.ones_like(
            disc_outputs,
            requires_grad=False,
            device=real_data.device)

        # Retrieve gradients
        gradients = autograd.grad(
            outputs=disc_outputs,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True)[0]

        # Compute gradient penalty
        gradients = gradients.view(BATCH_SIZE, -1)
        grad_norm = gradients.norm(2, dim=1)

        return torch.mean((grad_norm - 1) ** 2)

    def train_disc(
            self,
            x: torch.tensor,
            z: torch.tensor,
            categorical_vars: torch.tensor,
            lambda_penalty: int = 10,
            prob_success: int = 0,
            norm_scale: float = 0.5):
        """
        Train critic.
        ----
        Parameters:
            x (torch.tensor): input data.
            z (torch.tensor): sampled latent noise variables.
            categorical_vars (torch.tensor): conditional categorical covariates.
            lambda_penalty (int): weight for the gradient penalty. Default 10.
            prob_success (int): no augmentation if prob_success = 0 since the binomial returns only zeros. Default 0.
            norm_scale (int): parameter for random normal sampling for augmentations. Default 0.5.

        Returns:
         disc_loss (torch.tensor): critic wasserstein loss with gradient penalty.
        """
        BATCH_SIZE = z.shape[0]

        # Reset gradients back to 0
        self.D_optimizer.zero_grad()

        # We train only the critic
        for p in self.D.parameters():
            p.requires_grad = True

        # Avoid gradient computations on the generator
        for p in self.G.parameters():
            p.requires_grad = False

        with autocast():

            # Generator forward pass with concatenated expression data and
            # numerical covariates with categorical covariates
            gen_outputs = self.G(z, categorical_vars)
            NB_GENES = gen_outputs.shape[1]

            # Perform random augmentations for stability
            augmentations = torch.distributions.binomial.Binomial(total_count=1, probs=prob_success).sample(torch.tensor([BATCH_SIZE])).to(gen_outputs.device)
            gen_outputs = gen_outputs + augmentations[:, None] * torch.normal(0, norm_scale, size=(NB_GENES,), device=gen_outputs.device)
            #print("training BATCH size", x.shape)
            #print("BATCH size", BATCH_SIZE)
            x = x + augmentations[:,None] * torch.normal(0, norm_scale, size=(BATCH_SIZE,NB_GENES), device=x.device)

            # Forward pass on discriminator with concatenated variables
            disc_outputs = self.D(gen_outputs, categorical_vars)
            disc_real = self.D(x, categorical_vars)

            # Compute adversarial loss with GP
            real_loss, fake_loss = discriminator_loss(disc_real, disc_outputs)
            d_loss = real_loss + fake_loss
            gp = self.gradient_penalty(
                x,
                gen_outputs,
                categorical_vars=categorical_vars)

            # Full loss
            disc_loss = d_loss + lambda_penalty * gp

        # disc_loss.backward()
        self.D_scaler.scale(disc_loss).backward()

        # No need for gradient clipping here since we use GP
        # Update parameters
        # self.D_optimizer.step()
        self.D_scaler.step(self.D_optimizer)
        # Updates the scale for next iteration.
        self.D_scaler.update()

        # Track loss components
        self.LossTracker({"disc_loss_gp": disc_loss.detach().item(),
                        "d_loss": d_loss.detach().item(),
                        "gp": gp.detach().item(),
                        "real_loss": real_loss.detach().item(),
                        "fake_loss": fake_loss.detach().item()})

        return disc_loss

    def train_gen(
            self,
            z: torch.tensor,
            categorical_vars: torch.tensor,
            prob_success: int = 0,
            norm_scale: int = 0.5):
        """
        Train generator.
        ----
        Parameters:
            z (torch.tensor): sampled latent noise variables.
            categorical_vars (torch.tensor): conditional categorical covariates.
            numerical_vars (torch.tensor): conditional numerical covariates.
            prob_success (int): no augmentation if prob_success = 0 since the binomial returns only zeros. Default 0.
            norm_scale (int): parameter for random normal sampling for augmentations. Default 0.5.

        Returns:
            gen_loss (tensor): generator wasserstein loss.
        """

        BATCH_SIZE = z.shape[0]
        self.G.train()  # Train mode

        # Reset gradients back to 0
        self.G_optimizer.zero_grad()

        # We train only the generator
        for p in self.G.parameters():
            p.requires_grad = True

        # Avoid gradient computations on the critic
        for p in self.D.parameters():
            p.requires_grad = False

        with autocast():
            # Generator forward pass with concatenated variables
            gen_outputs = self.G(z, categorical_vars)
            NB_GENES = gen_outputs.shape[1]

            # Perform random augmentations for stability
            augmentations = torch.distributions.binomial.Binomial(total_count=1, probs=prob_success).sample(torch.tensor([BATCH_SIZE])).to(gen_outputs.device)
            gen_outputs = gen_outputs + augmentations[:, None] * torch.normal(0, norm_scale, size=(NB_GENES,), device=gen_outputs.device)

            # Forward pass on discriminator
            disc_outputs = self.D(gen_outputs, categorical_vars)

            # Compute losses
            gen_loss = generator_loss(disc_outputs)

        # Backpropagate
        self.G_scaler.scale(gen_loss).backward()
        # gen_loss.backward()

        # Track loss components
        self.LossTracker({"g_loss_batch": gen_loss.detach().item()})

        # No need for gradient clipping here since we use GP
        # Update parameters
        # self.G_optimizer.step()
        self.G_scaler.step(self.G_optimizer)
        # Updates the scale for next iteration.
        self.G_scaler.update()

        return gen_loss

    def train(self, TrainDataLoader,
              ValDataLoader,
              z_dim: int,
              epochs: int,
              iters_critic: int = 5,
              lambda_penalty: int = 10,
              step: int = 5,
              verbose: bool = True,
              checkpoint_dir: str = './checkpoints',
              log_dir: str = './logs/',
              fig_dir: str = './figures',
              prob_success: int = 0,
              norm_scale: float = 0.5,
              optimizer: str = 'rms_prop',
              lr_g: float = 5e-4,
              lr_d: float = 5e-4,
              config: dict = None,
              hyperparameters_search: bool = False):
        """
        Main train function to train full model.
        ----
        Parameters:
            TrainDataLoader (pytorch loader): train data loader with expression data, covariates and labels.
            ValDataLoader (pytorch loader): validation data loader.
            z_dim (int): latent noise dimension.
            epochs (int): number of training epochs.
            iters_critic (int): number of iteration per batch for critic. Default 5.
            lambda_penalty (int): penalty on gradient in loss. Default 10.
            step (int): epoch frequency to compute evaluation metrics. Default 5.
            verbose (bool): print training callbacks (default True).
            checkpoint_dir (str): path directory where to save model weights.
            log_dir (str): path where to save logs.
            fig_dir (str): path where to save figures.
            prob_success (int): probability of success in Bernouilli function to apply random noise on some variables. Default 0.
            norm_scale (float): Bernouilli function variance to apply random noise on some variables. Default 0.5.
            optimizer (str): either 'rms_prop' or 'adam'. Default 'rms_prop'.
            lr_g (float): generator learning rate. Default 5e-4.
            lr_d (float): discriminator learning rate. Default 5e-4.
            config (dict): dictionnary of model configuration.
            hyperparameters_search (bool): whether training is performed for a search and weights should be saved in same path every run. Default False.
        """

        # Log training duration
        self.t_begin = t.time()
        self.epoch_times = []

        # Init optimizers and directories
        self.init_train(log_dir,
                        checkpoint_dir,
                        fig_dir,
                        optimizer=optimizer,
                        lr_g=lr_g,
                        lr_d=lr_d,
                        epochs=epochs,
                        )

        #--Scaler for Mixed Precision--
        self.G_scaler = GradScaler()
        self.D_scaler = GradScaler()

        # Write configuration and architecture in log
        write_config(file_path=self.path_log, config=config)

        # Init training metrics history
        all_epochs_val_score = []
        all_epochs_prec_recall_train = []
        all_epochs_dens_cov_train = []
        all_aats_train = []
        all_frechet_train = []

        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            self.start_epoch_time = t.time()

            # Init epoch losses
            epoch_disc_loss = 0
            epoch_gen_loss = 0

            # Loop over batches
            for batch, labels in TrainDataLoader:
                # print("shape",batch.shape)
                # print("shape again",labels.shape)
                # To device
                batch = batch.to(self.device)
                batch_categorical = labels.to(self.device)

                ################ Train critic ################
                disc_loss_iters = []
                for iter in range(iters_critic):
                    # Get random latent variables z
                    batch_z = torch.normal(0,1, size=(batch.shape[0], z_dim), device=self.device)

                    # Train critic and return loss
                    disc_loss = self.train_disc(
                        batch,
                        batch_z,
                        batch_categorical,
                        lambda_penalty=lambda_penalty,
                        prob_success=prob_success,
                        norm_scale=norm_scale)

                    disc_loss_iters.append(disc_loss.detach().item())

                # Track loss components
                self.LossTracker({"disc_loss_batch": disc_loss_iters[-1]})

                # Store last iter loss of critic model as epoch loss
                epoch_disc_loss += disc_loss_iters[-1]

                ################ Train generator ################
                # Sample random latent variables z
                batch_z = torch.normal(0, 1, size=(batch.shape[0], z_dim), device=self.device)
                gen_loss = self.train_gen(
                    batch_z,
                    batch_categorical,
                    prob_success=prob_success,
                    norm_scale=norm_scale)
                epoch_gen_loss += gen_loss.detach().item()

            # Store all losses
            self.LossTracker({"disc_loss_epoch": epoch_disc_loss /
                              len(TrainDataLoader), "g_loss_epoch": epoch_gen_loss /
                              len(TrainDataLoader)})

            self.end_epoch_time = t.time()
            self.epoch_time = self.end_epoch_time - self.start_epoch_time

            ########## Epochs checkpoints ####################
            if epoch % step == 0:

                x_real, x_gen = self.generate(TrainDataLoader, return_labels=False)
                x_real, x_gen = x_real.numpy(), x_gen.numpy()

                # Train metrics
                all_epochs_val_score, all_epochs_prec_recall_train, all_epochs_dens_cov_train, all_aats_train, all_frechet_train = epoch_checkpoint_train(x_real, x_gen,
                                                                                                                self.nb_nn,
                                                                                                                list_val_score = all_epochs_val_score,
                                                                                                                list_prec_recall_train=all_epochs_prec_recall_train,
                                                                                                                list_dens_cov_train=all_epochs_dens_cov_train,
                                                                                                                list_aats_train=all_aats_train,
                                                                                                                list_frechet_train=all_frechet_train,
                                                                                                                dataset=self.dataset,
                                                                                                                device=self.device_frechet
                                                                                                                )

                # Epoch duration
                self.epoch_times.append(self.epoch_time)

                # Write logs
                watch_dict = {'epoch': epoch,
                              'gen_loss': epoch_gen_loss,
                              'disc_loss': epoch_disc_loss,
                              'val_score': all_epochs_val_score[-1],
                              'precision_train': round(all_epochs_prec_recall_train[-1][0], 3),
                              'recall_train': round(all_epochs_prec_recall_train[-1][1], 3),
                              'density_train': round(all_epochs_dens_cov_train[-1][0], 3),
                              'coverage_train': round(all_epochs_dens_cov_train[-1][1], 3),
                              'AAts_train': round(all_aats_train[-1], 3),
                              'FD_train': round(all_frechet_train[-1], 3),
                              'nb_samples': len(TrainDataLoader)}

                write_log(file_path=self.path_log, metrics_dict=watch_dict)

                if verbose:
                    print_func(watch_dict)

        ############### End of training ##############################

        # Print training time
        self.time_sec = print_training_time(self.t_begin)

        # Save last weigths for G and D
        save_weights(
            self.G,
            self.D,
            self.G_path,
            self.D_path,
            hyperparameters_search)

        # Save history
        save_history(all_epochs_val_score, self.history_val_score_path)
        save_history(all_epochs_prec_recall_train, self.history_prec_recall_train_path)
        save_history(all_epochs_dens_cov_train, self.history_dens_cov_train_path)
        save_history(all_aats_train, self.history_aats_train_path)
        save_history(all_frechet_train, self.history_frechet_train_path)
        save_history(self.epoch_times, self.epoch_times_path)

    def init_train(
            self,
            log_dir: str,
            checkpoint_dir: str,
            fig_dir: str,
            optimizer: str = 'rms_prop',
            lr_g: float = 5e-4,
            lr_d: float = 5e-4,
            epochs: int = None):
        """
        Training initialization: init directories, callbacks and PCA.
        ----
        Parameters:
            log_dir (str): path where to save logs.
            checkpoint_dir (str): path where to save model weights.
            fig_dir (str): path where to store figures.
            optimizer (str): either 'rms_prop' or 'adam'. Default 'rms_prop'.
            lr_g (float): generator learning rate. Default 5e-4.
            lr_d (float): discriminator learning rate. Default 5e-4.
            epochs (int): number of training epochs.
            trainloader (pytorch loader): train data loader with expression data, covariates and labels.
            nb_principal_components (int): dimension of principal components reduction for analysis. Default 2000.
        """
        # Optimizers
        if optimizer.lower() == 'rms_prop':
            # Use the RMSProp version of gradient descent with small learning
            # rate and no momentum (e.g. 0.00005).
            self.G_optimizer = torch.optim.RMSprop(
                self.G.parameters(), lr=lr_g)
            self.D_optimizer = torch.optim.RMSprop(
                self.D.parameters(), lr=lr_d)
        elif optimizer.lower() == 'adam':
            self.G_optimizer = torch.optim.Adam(
                self.G.parameters(), lr=lr_g, betas=(.9, .99))
            self.D_optimizer = torch.optim.Adam(
                self.D.parameters(), lr=lr_d, betas=(.9, .99))

        # Set up logs and checkpoints
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = log_dir + '/'+ current_time

        # Make dir if it does not exist
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
            print("Directory '%s' created" % self.log_dir)

        # training history path
        self.history_gen_path = self.log_dir + '/train_gen_loss.npy'
        self.history_disc_path = self.log_dir + '/train_disc_loss.npy'
        self.history_val_score_path = self.log_dir + '/val_score.npy'
        self.history_prec_recall_train_path = self.log_dir + '/precision_recall_train.npy'
        self.history_dens_cov_train_path = self.log_dir + '/density_coverage_train.npy'
        self.history_aats_train_path = self.log_dir + '/aats_train.npy'
        self.history_frechet_train_path = self.log_dir+'/frechet.npy'
        self.epoch_times_path = self.log_dir + '/epoch_durations.npy'

        # Init log path
        self.path_log = self.log_dir + '/logs.txt'

        # Define checpoints path
        self.checkpoint_dir = os.path.join(checkpoint_dir, current_time)
        # Make dir if it does not exist
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
            print("Directory '%s' created" % self.checkpoint_dir)

        # Create figures folder
        self.fig_dir = os.path.join(fig_dir, current_time)
        # Make dir if it does not exist
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)
        if not os.path.exists(self.fig_dir):
            os.mkdir(self.fig_dir)
            print("Directory '%s' created" % self.fig_dir)

        # Initialize the paths for models
        self.D_path = self.checkpoint_dir + '/_disc.pt'
        self.G_path = self.checkpoint_dir + '/_gen.pt'

        # Initialize the tracking object for loss components
        self.LossTracker = TrackLoss(
            path=self.log_dir +
            '/train_history.npy',
            nb_epochs=epochs)

    def generate(self, DataLoader, return_labels:bool=False):
        """
        Returns real and generated data.
        """
        x_gen = []
        x_real = []
        y = []
        self.G.eval()  # Evaluation mode

        with torch.no_grad():
            for batch, labels in DataLoader:
                # Conditioning variable
                labels = labels.to(self.device)

                # Get random latent variables z
                batch_z = torch.normal(0,1, size=(batch.shape[0], self.G.latent_dim), device=self.device)

                with autocast():
                    gen_outputs = self.G(batch_z, labels)
                
                # Store data
                x_gen.append(gen_outputs.detach().cpu())
                x_real.append(batch)
                if return_labels:
                    y.append(labels.detach().cpu())

        # Concatenate and to array
        x_gen = torch.cat(x_gen, 0).detach()
        x_real = torch.cat(x_real, 0)

        if return_labels:
            y = torch.cat((y), axis=0)
            return x_real, x_gen, y

        elif not return_labels:
            return x_real, x_gen
        
    def load_generator(self, path:str=None, location:str="cpu"):
        """
        Loading previously trained generator model.
        ----
        Parameters:
            path (str): path where model has been stored"""
        
        assert path is not None, "Please provide a path to load the Generator from."
        try:
            self.G.load_state_dict(torch.load(path, map_location=location))
            print('Generator loaded.')
        except FileNotFoundError: # if no model saved at given path
            print(f"No previously trained weights found at given path: {path}.")
