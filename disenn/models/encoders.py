import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    """ Convolutional Encoder

    Architecture
    ------------
    Conv: 32 channels, 4x4
    Conv: 32 channels, 4x4
    Conv: 32 channels, 4x4
    Conv: 32 channels, 4x4
    FC: 256 neurons
    FC: 256 neurons
    FC: Latent Dimension x 2 => (Mean, Log Variance)
    
    References
    ----------
    [1] Burgess, Christopher P., et al. "Understanding disentangling in $\beta$-VAE." (2018)
    """

    def __init__(self, z_dim: int):
        """ Instantiate Encoder for a VAE

        Parameters
        ----------
        z_dim: int
            dimension of the latent distribution to be learnt
        """
        super().__init__()
        in_channels = 3 
        h_channels = 32
        kernel_size = 4
        h_dim = 256
        self.z_dim = z_dim

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, h_channels, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_channels, h_channels, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_channels, h_channels, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_channels, h_channels, kernel_size, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_block = nn.Sequential(
            nn.Linear((h_channels * kernel_size * kernel_size), h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        # layers which generates mean and log variance
        self.mu_layer = nn.Linear(h_dim, self.z_dim)
        self.logvar_layer = nn.Linear(h_dim, self.z_dim)

    def forward(self, x):
        """Forward pass of a convolutional encoder"""
        assert len(x.shape)==4 and x.shape[2]==64 and x.shape[3]==64,\
        "input must be of shape (batch_size x 3 x 64 x 64)"
        batch_size = x.shape[0]
        x = self.conv_block(x)
        x = x.view(batch_size, -1)
        x = self.fc_block(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

