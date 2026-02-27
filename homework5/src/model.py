import torch
from torch import nn

class Autoencoder(nn.Module):
    """
    TODO: Implement encoder and decoder using linear layers and ReLU activation.
    """
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        # TODO: Define encoder as a small MLP: input_dim -> hidden_dim -> latent_dim
        # TODO: Use nn.Linear + nn.ReLU (optionally nn.Dropout between layers)
        # TODO: Define decoder as a mirrored MLP: latent_dim -> hidden_dim -> input_dim
        # TODO: Keep output linear for reconstruction (loss handles scale)
        pass

    def forward(self, x):
        # TODO: Encode x to latent representation z
        # TODO: Decode z to reconstruction x_hat
        # TODO: Return (x_hat, z)
        pass

class VariationalAutoencoder(nn.Module):
    """
    TODO: Implement:
      - Encoder with layers to compute mu and logvar (log variance)
      - Decoder network
      - Reparameterization trick in reparameterize()

    Hint: Encoder: input -> hidden -> (mu, logvar)
          Decoder: latent -> hidden -> output
    """
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        # TODO: Define encoder: input_dim -> hidden_dim (Linear + ReLU)
        # TODO: From encoder hidden, create two heads:
        #       - mu layer (hidden_dim -> latent_dim)
        #       - logvar layer (hidden_dim -> latent_dim)
        # TODO: Define decoder: latent_dim -> hidden_dim -> input_dim
        pass

    def reparameterize(self, mu, logvar):
        # TODO: Compute std = exp(0.5 * logvar)
        # TODO: Sample eps ~ N(0, I) with torch.randn_like(std)
        # TODO: Return z = mu + eps * std
        pass

    def forward(self, x):
        # TODO: Encode x to get mu and logvar
        # TODO: Sample z with reparameterize(mu, logvar)
        # TODO: Decode z to reconstruction x_hat
        # TODO: Return (x_hat, mu, logvar) for VAE loss
        pass
