import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.distributions as dist

def plot_embedding(data, labels, filename, method="pca", random_state=0):
    """
    data: (N, D) numpy array
    labels: (N,) array-like (ints)
    """
    data = np.asarray(data)
    labels = np.asarray(labels).reshape(-1)

    if method == "pca":
        reduced = PCA(n_components=2, random_state=random_state).fit_transform(data)
    elif method == "tsne":
        # Optional: pre-reduce to 50 dims for speed/stability
        data_50 = PCA(n_components=min(50, data.shape[1]), random_state=random_state).fit_transform(data)
        reduced = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(data_50)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    plt.figure(figsize=(5, 4))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=3)
    plt.title(f"{method.upper()} projection")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def add_gaussian_noise(x, std=0.1, clamp_nonneg=True):
    x_noisy = x + torch.randn_like(x) * std
    if clamp_nonneg:
        x_noisy = torch.clamp(x_noisy, min=0.0)
    return x_noisy

def add_negative_binomial_noise(x, r=10.0, p=0.5, clamp_nonneg=True):
    """
    Adds i.i.d. NB noise counts to x. This assumes x is in count space.
    """
    nb = dist.NegativeBinomial(total_count=torch.tensor(r, device=x.device), probs=torch.tensor(p, device=x.device))
    noise = nb.sample(x.shape).float()
    x_noisy = x + noise
    if clamp_nonneg:
        x_noisy = torch.clamp(x_noisy, min=0.0)
    return x_noisy
