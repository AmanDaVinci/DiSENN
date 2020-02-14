import torch
import torch.nn as nn

class ConvParameterizer(nn.Module):
    """ Convolutional Parameterizer for DiSENN model

    Generates parameters as relevance scores for concepts.
    Follows the same architecture as the VAE Conceptizer's encoder module.

    Architecture
    ------------
    Conv: 32 channels, 4x4
    Conv: 32 channels, 4x4
    Conv: 32 channels, 4x4
    Conv: 32 channels, 4x4
    FC: 256 neurons
    FC: 256 neurons
    FC: num_concepts + num_classes
    """

    def __init__(self, num_concepts: int, num_classes: int):
        """ Init ConvParameterizer

        Parameters
        ----------
        num_concepts: int
            Number of concepts for which relevance scores must be generated

        num_classes: int
            Number of classes to be classifed in the downstream task
        """
        super().__init__()
        in_channels = 3 
        h_channels = 32
        kernel_size = 4
        h_dim = 256
        self.num_concepts = num_concepts
        self.num_classes = num_classes

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
        # final layer to generate the parameters for each concept and class
        self.concept_class_layer = nn.Linear(h_dim, (num_concepts + num_classes))
  
    def forward(self, x: torch.Tensor):
        """ Generates relevance scores as parameters for concepts

        Parameters
        ----------
        x: torch.Tensor
            input tensor of shape (batch_size, 3, 64, 64)

        Returns
        -------
        parameters: torch.Tensor
            relevance scores for concepts as parameters of shape (batch_size, num_concepts, num_classes)
        """
        assert len(x.shape)==4 and x.shape[2]==64 and x.shape[3]==64,\
        "input must be of shape (batch_size x 3 x 64 x 64)"
        batch_size = x.shape[0]
        x = self.conv_block(x)
        x = x.view(batch_size, -1)
        x = self.fc_block(x)
        x = self.concept_class_layer(x)
        return x.view(batch_size, self.num_concepts, self.num_classes)
