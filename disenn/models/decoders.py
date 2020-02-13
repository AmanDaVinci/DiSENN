import torch
import torch.nn as nn
import numpy as np

class ConvDecoder(nn.Module):
    """ Convolutional Decoder

    Architecture
    ------------
    FC: Latent Dimension
    FC: 256 neurons
    FC: 256 neurons
    Transpose Conv: 32 channels, 4x4
    Transpose Conv: 32 channels, 4x4
    Transpose Conv: 32 channels, 4x4
    Transpose Conv: 32 channels, 4x4
    
    References
    ----------
    [1] Burgess, Christopher P., et al. "Understanding disentangling in $\beta$-VAE." (2018)
    """

    def __init__(self, z_dim: int):
        """ Instantiate Decoder for a VAE

        Parameters
        ----------
        z_dim: int
            dimension of the latent distribution to be learnt
        """
        super().__init__()
        self.z_dim = z_dim
        h_dim = 256
        h_channels = 32
        kernel_size = 4
        out_channels = 3 
        self.conv_shape = (h_channels, kernel_size, kernel_size)

        self.fc_block = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, np.product(self.conv_shape))
        )
        self.tconv_block = nn.Sequential(
            nn.ConvTranspose2d(h_channels, h_channels, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_channels, h_channels, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_channels, h_channels, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_channels, out_channels, kernel_size, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.fc_block(x)
        x = x.view(batch_size, *self.conv_shape)
        x = self.tconv_block(x)
        return torch.sigmoid(x)

