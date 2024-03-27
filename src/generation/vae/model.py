import os
import sys
import torch.nn as nn 
import torch
import numpy as np
from src.generation.vae.utils import init_weights

class VAE(nn.Module):
    """
    """

    def __init__(self, config=None):
        super(VAE, self).__init__()

        self.device = config['device']

        list_dims_enc = config['list_dims_encoder']
        list_dims_dec = config['list_dims_decoder']

        # Dimensions
        self.vocab_size = config['vocab_size']
        self.embedd_dim = config['embedding_dim']
        self.latent_dim = config['latent_dim']

        # Embedding layers for tissue type/cancer types
        self.embedding = nn.Embedding(self.vocab_size, self.embedd_dim)

        # encoder
        layers_enc =[]
        for i in range(len(list_dims_enc)-1):
            if i < len(list_dims_enc)-2:
                if i==0:
                    layers_enc.append(nn.Linear(list_dims_enc[i]+ self.embedd_dim * self.vocab_size, list_dims_enc[i+1]))
                else:
                    layers_enc.append(nn.Linear(list_dims_enc[i], list_dims_enc[i+1]))
                layers_enc.append(nn.BatchNorm1d(list_dims_enc[i+1]))
                layers_enc.append(nn.LeakyReLU(config['negative_slope']))
                #layers_enc.append(nn.Dropout(config['dropout']))

        self.encoder = nn.Sequential(*layers_enc)
        self.encoder.apply(init_weights) # Xavier init
        
        # latent mean and variance (conditioned)
        self.mean_layer = nn.Linear(list_dims_enc[-2] + self.embedd_dim * self.vocab_size, list_dims_enc[-1])
        self.logvar_layer = nn.Linear(list_dims_enc[-2] + self.embedd_dim * self.vocab_size, list_dims_enc[-1])

        # decoder
        layers_dec =[]
        for i in range(len(list_dims_dec)-1):
            if i < len(list_dims_dec)-2:
                if i==0:
                    layers_dec.append(nn.Linear(list_dims_dec[i]+ self.embedd_dim * self.vocab_size, list_dims_dec[i+1]))
                else:
                    layers_dec.append(nn.Linear(list_dims_dec[i], list_dims_dec[i+1]))
                layers_dec.append(nn.BatchNorm1d(list_dims_dec[i+1]))
                layers_dec.append(nn.LeakyReLU(config['negative_slope']))
                #layers_dec.append(nn.Dropout(config['dropout']))
        # output of decoder unconstrained
        layers_dec.append(nn.Linear(list_dims_dec[-2], list_dims_dec[-1]))
        self.decoder = nn.Sequential(*layers_dec)
        self.decoder.apply(init_weights) # Xavier init
     
    def encode(self, x, y):
        y = self.embedding(y)  # Embedding for tissue types/cancer types
        x = torch.cat((x, y.flatten(start_dim=1)), 1) # Concatenate all variables (expression data and conditions)
        x = self.encoder(x)
        mean, logvar = self.mean_layer(torch.cat((x, y.flatten(start_dim=1)), 1)), self.logvar_layer(torch.cat((x, y.flatten(start_dim=1)), 1))
        return mean, logvar

    def reparameterization(self, mean, logvar):   
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, x, y):
        y = self.embedding(y)  # Embedding for tissue types/cancer types
        x = torch.cat((x, y.flatten(start_dim=1)), 1) # Concatenate all variables (expression data and conditions)
        return self.decoder(x)

    def forward(self, x, y):
        mean, logvar = self.encode(x, y)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z, y)
        return x_hat, mean, logvar
    

    def generate(self, DataLoader, return_labels:bool=False):
        """
        Returns real and generated data.
        """
        x_gen = []
        x_real = []
        y = []
        self.decoder.eval()  # Evaluation mode

        with torch.no_grad():
            for batch, labels in DataLoader:
                # Conditioning variable
                labels = labels.to(self.device)

                # Get random latent variables z
                batch_z = torch.randn(batch.shape[0], self.latent_dim, device=self.device)
                gen_outputs = self.decode(batch_z, labels)
                
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
        
        
    def load_decoder(self, path:str=None, location:str="cpu"):
        """
        Loading previously trained generator model.
        ----
        Parameters:
            path (str): path where model has been stored"""
        
        assert path is not None, "Please provide a path to load the Generator from."
        try:
            self.decoder.load_state_dict(torch.load(path, map_location=location))
            print('Decoder loaded.')
        except FileNotFoundError: # if no model saved at given path
            print(f"No previously trained weights found at given path: {path}.")