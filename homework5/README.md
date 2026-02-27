# Homework 5: Autoencoders and VAEs for Single-Cell Expression

## Overview

In this homework, you will work with **single-cell gene expression data** to explore key ideas behind **Autoencoders (AEs)** and **Variational Autoencoders (VAEs)**. You will:

- Preprocess and explore single-cell RNA-seq count data
- Train AEs and VAEs to learn low-dimensional representations
- Visualize both raw and reconstructed embeddings using PCA and t-SNE
- Reflect on how biological variation is captured in latent space

---

## Dataset

You are provided with a small single-cell RNA-seq dataset (`counts.npy` and `labels.txt`) for experimentation.

- **counts.npy**: Raw gene expression counts (cells × genes)
- **labels.txt**: Cell type labels (used for evaluation and visualization)


Later in the homework, you will be asked to **select and explore your own single-cell dataset** from a public source (e.g., [GEO](https://www.ncbi.nlm.nih.gov/geo/), [CellxGene](https://cellxgene.cziscience.com/), or [Open Problems](https://openproblems.bio)).

---

## Installation

1. **Clone** this repo:
   ```bash
   git clone git@github.com:brown-csci1851/stencil.git
   cd stencil/homework5
   ```
   If you already cloned it, update and move into the homework folder:
   ```bash
   cd stencil
   git pull
   cd homework5
   ```
2. Create virtual environment:
    ```bash
    python -m venv .hw5
    ```
3. Install dependencies:
    ```bash
    source .hw5/bin/activate (Linux/MacOS) or .\.hw5\Scripts\activate (Windows)
    pip install -r requirements.txt
    ```
---

## Tasks

You will complete the following:

- [ ] Train AEs with latent sizes **5**, **10**, **50**, and **100** on the provided dataset
- [ ] Compare their reconstruction **Mean Squared Error (MSE)** on a held-out validation set
- [ ] Train a VAE and compare it to the AE (using the same latent sizes)
- [ ] Generate PCA and t-SNE projections (visualization) of:
  - Raw gene expression data
  - Reconstructed data
  - Latent embeddings
- [ ] Analyze how latent dimension size affects **clustering structure in the latent space**
- [ ] Convert your AE into a **denoising autoencoder**:
  - Add **Gaussian noise**
  - Add **negative binomial noise**
  - Evaluate and compare latent space quality across models using PCA/t-SNE and at least one quantitative metric (e.g., silhouette score)
- [ ] Apply the same analysis (preprocessing, training, visualization, and evaluation) to the dataset of your choice
- [ ] Compare trends observed on the provided dataset versus your own dataset
- [ ] Perform **clustering** (e.g., K-Means, DBSCAN) in the latent space
- [ ] [Optional] Use **feature-attribution tools** (e.g., SHAP or DeepLIFT) to relate latent representations back to input genes

---

## Final Reflection

You will then write a **2–3 page reflection** that includes **figures** and **interpretation** of your results. Your write-up should clearly reference the plots, tables, and metrics you generated.

- [ ] How does the performance of an AE compare to that of a VAE?
- [ ] How does latent dimension size affect reconstruction error and clustering structure?
- [ ] How do silhouette scores vary across models and latent dimensions?
- [ ] What do the PCA and t-SNE visualizations reveal about the structure of the learned representations?
- [ ] How does adding Gaussian noise versus negative binomial noise affect the learned latent space?
- [ ] What differences do you observe in robustness or clustering behavior?
- [ ] What patterns or clusters emerge in the latent space?
      Do these correspond to known biological groupings (e.g., cell types or conditions), and where do they fail?
- [ ] What aspects of the data or model behavior surprised you?
- [ ] How did the model help—or fail—to reveal meaningful biological structure in the data?

---

## Expected Skills

By the end of this homework, you should be able to:

- Preprocess and visualize high-dimensional gene expression data
- Understand how AEs and VAEs learn compressed representations
- Evaluate reconstruction quality and latent space structure
- Apply dimensionality reduction techniques for biological data exploration
