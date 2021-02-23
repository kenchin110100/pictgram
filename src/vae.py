# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self, in_features, second_features, latent_features):
        super(LinearVAE, self).__init__()
        
        self.latent_features = latent_features
 
        # encoder
        self.enc1 = nn.Linear(in_features=in_features, out_features=second_features)
        self.enc2 = nn.Linear(in_features=second_features, out_features=latent_features*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=latent_features, out_features=second_features)
        self.dec2 = nn.Linear(in_features=second_features, out_features=in_features)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
    
    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, self.latent_features)
        return x
    
    def cal_mu_var(self, x):
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        return mu, log_var
    
    def decode(self, z):
        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction
 
    def forward(self, x):
        # encoding
        x = self.encode(x)
        
        mu, log_var = self.cal_mu_var(x)
        
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        
        # decode
        reconstruction = self.decode(z)

        return reconstruction, mu, log_var
 