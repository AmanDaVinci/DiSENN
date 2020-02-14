import torch
import torch.nn as nn
from .encoders import ConvEncoder
from .decoders import ConvDecoder


class VaeConceptizer(nn.Module):
    """Variational Auto Encoder to generate basis concepts

    Concepts should be independently sensitive to single generative factors,
    which will lead to better interpretability and fulfill the "diversity" 
    desiderata for basis concepts in a Self-Explaining Neural Network.
    VAE can be used to learn interpretable representations of the basis concepts 
    by emphasizing the discovery of latent factors which are disentangled. 
    """

    def __init__(self, num_concepts: int):
        """Initialize Variational Auto Encoder

        Parameters
        ----------
        num_concepts : int
            number of basis concepts to learn in the latent distribution space
        """
        super().__init__()
        self.z_dim = num_concepts
        self.encoder = ConvEncoder(self.z_dim)
        self.decoder = ConvDecoder(self.z_dim)

    def forward(self, x):
        """Forward pass through the encoding, sampling and decoding step

        Parameters
        ----------
        x : torch.tensor 
            input of shape [batch_size x 64 x 64], which will be flattened

        Returns
        -------
        concept_mean : torch.tensor
            mean of the latent distribution induced by the posterior input x

        concept_logvar : torch.tensor
            log variance of the latent distribution induced by the posterior input x

        x_reconstruct : torch.tensor
            reconstruction of the input x
        """
        assert len(x.shape)==4 and x.shape[2]==64 and x.shape[3]==64,\
        "input must be of shape (batch_size x 3 x 64 x 64)"
        concept_mean, concept_logvar = self.encoder(x)
        concept_sample = self.sample(concept_mean, concept_logvar)
        x_reconstruct = self.decoder(concept_sample)
        return (concept_mean, concept_logvar, x_reconstruct.view_as(x))

    def sample(self, mean, logvar):
        """Samples from the latent distribution using reparameterization trick

        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon is drawn from a standard normal distribution
        
        Parameters
        ----------
        mean : torch.tensor
            mean of the latent distribution of shape [batch_size x z_dim]
        log_var : torch.tensor
            diagonal log variance of the latent distribution of shape [batch_size x z_dim]
        
        Returns
        -------
        z : torch.tensor
            sample latent tensor of shape [batch_size x z_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            z = mean + std * epsilon
        else:
            z = mean
        return z