import torch
import torch.nn.functional as F


def BVAE_Loss(x, x_hat, z_mean, z_logvar):
    """ Calculate Beta-VAE loss as in [1]

    Parameters
    ----------
    x : torch.tensor
        input data to the Beta-VAE

    x_hat : torch.tensor
        input data reconstructed by the Beta-VAE

    z_mean : torch.tensor
        mean of the latent distribution of shape
        (batch_size, latent_dim)

    z_logvar : torch.tensor
        diagonal log variance of the latent distribution of shape
        (batch_size, latent_dim)

    Returns
    -------
    loss : torch.tensor
        loss as a rank-0 tensor calculated as:
        reconstruction_loss + beta * KL_divergence_loss

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """
    # recon_loss = F.binary_cross_entropy(x_hat, x.detach(), reduction="mean")
    recon_loss = F.mse_loss(x_hat, x.detach(), reduction="mean")
    kl_loss = kl_div(z_mean, z_logvar)
    return recon_loss, kl_loss


def kl_div(mean, logvar):
    """Computes KL Divergence between a given normal distribution
    and a standard normal distribution

    Parameters
    ----------
    mean : torch.tensor
        mean of the normal distribution of shape (batch_size x latent_dim)

    logvar : torch.tensor
        diagonal log variance of the normal distribution of shape (batch_size x latent_dim)

    Returns
    -------
    loss : torch.tensor
        KL Divergence loss computed in a closed form solution

    References
    ----------
    [1] Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014

    """
    # see Appendix B from [1]:
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return loss
