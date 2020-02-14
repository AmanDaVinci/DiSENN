import torch
import torch.nn.functional as F

def celeba_robustness_loss(x, aggregates, concepts, relevances):
    """Computes Robustness Loss for CelebA dataset
    
    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format

    Parameters
    ----------
    x            : torch.tensor
                 Input as (batch_size x num_features)
    aggregates   : torch.tensor
                 Aggregates from SENN as (batch_size x num_classes)
    concepts     : torch.tensor
                 Concepts from Conceptizer as (batch_size x num_concepts)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)
   
    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as frobenius norm of (batch_size x num_classes x num_features)
    """
    batch_size = x.size(0)
    num_channels = x.size(1)
    num_concepts = concepts.size(1)
    num_classes = aggregates.size(1)

    # Jacobian of aggregates wrt x
    jacobians = []
    for i in range(num_classes):
        grad_tensor = torch.zeros(batch_size, num_classes).to(x.device)
        grad_tensor[:, i] = 1.
        j_yx = torch.autograd.grad(outputs=aggregates, inputs=x, \
                                   grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
        # bs x 3 x 64 x 64 -> bs x (4096*3) x 1
        jacobians.append(j_yx.view(batch_size, -1).unsqueeze(-1))
    # bs x num_features x num_classes (bs x (4096*3) x 2)
    J_yx = torch.cat(jacobians, dim=2)

    # Jacobian of concepts wrt x
    jacobians = []
    for i in range(num_concepts):
        grad_tensor = torch.zeros(batch_size, num_concepts).to(x.device)
        grad_tensor[:, i] = 1.
        j_hx = torch.autograd.grad(outputs=concepts, inputs=x, \
                                   grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
        # bs x 3 x 64 x 64 -> bs x (4096*3) x 1
        jacobians.append(j_hx.view(batch_size, -1).unsqueeze(-1))
    # bs x num_features x num_concepts (bs x (4096*3) x 2)
    J_hx = torch.cat(jacobians, dim=2)

    # bs x num_features x num_classes
    robustness_loss = J_yx - torch.bmm(J_hx, relevances)

    return robustness_loss.norm(p='fro')


def BVAE_loss(x, x_hat, z_mean, z_logvar):
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
