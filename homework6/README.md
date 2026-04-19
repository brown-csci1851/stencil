# Homework 6: Introduction to GNNs and R-GCNs

## Overview

In this homework, you will work with biomedical graph data to explore fundamental ideas behind Graph Neural Networks (GNNs) and Relational Graph Convolutional Networks (R-GCNs). You will:

- Load and process graph-structured datasets
- Construct meaningful node and edge representations
- Train GNN and R-GCN models for link prediction
- Evaluate model performance
- Visualize learned node embeddings and graph structure

---

## Dataset

You are provided with two small biomedical graph datasets for experimentation.

- **Drug-disease interactions**: nodes are drugs and diseases, and edges describe relations such as `treats`, `palliates`, or `neither`
- **Drug-drug interactions**: nodes are drugs, and edges describe pharmacological interaction types

These datasets are provided as split CSV files for training and testing (use part of your training set for validation). In this homework, you will convert those CSV files into PyTorch Geometric `Data` objects and use them for graph-based link prediction.

---

## Installation

1. **Clone** this repo:
   ```bash
   git clone git@github.com:brown-csci1851/stencil.git
   cd stencil/homework6
   ```
   If you already cloned it, update and move into the homework folder:
   ```bash
   cd stencil
   git pull
   cd homework6
   ```
2. Create virtual environment:
    ```bash
    python -m venv .hw6
    ```
3. Install dependencies:
    ```bash
    source .hw6/bin/activate (Linux/MacOS) or .\.hw6\Scripts\activate (Windows)
    pip install -r requirements.txt
    ```

After creating and activating the virtual environment, select it as the Jupyter kernel in `src/playground.ipynb` to run the notebook using the same installed dependencies.

---

## Tasks

You will complete the following:

- [ ] Implement your own CSV-to-graph loading pipeline:
  - Parse the split CSV files
  - Build shared node and relation ID mappings
  - Construct PyTorch Geometric `Data` objects
- [ ] Preserve lookup metadata during loading so you can interpret embeddings later:
  - Store a mapping from node ID back to the original drug or disease identifier
  - Store a mapping from relation ID back to the original interaction label
- [ ] Load and explore both graphs (drug-disease and drug-drug)
- [ ] Visualize node degree and edge-type distributions
- [ ] Train a simple **GCN** model for link prediction (for both datasets)
- [ ] Train an **R-GCN** model for link prediction (for both datasets)
- [ ] Compare **GCN** and **R-GCN** performance on the provided data
- [ ] Experiment with different node feature types:
  - Random features
  - One-hot features
  - [Optional] Pretrained features derived from the entity description
    (for example, embeddings using the drug's or disease's definition provided with the data)
- [ ] Compare results across multiple hyperparameter settings:
  - Embedding dimension
  - Number of GNN layers
  - Node feature type (random, one-hot, description-based features)
- [ ] Generate PCA and t-SNE projections of learned node embeddings
- [ ] Interpret whether drugs, diseases, or relation types cluster together or separate cleanly, or behave like outliers

---

## Final Reflection

You will then write a **2-3 page reflection** that includes **figures** and **interpretation** of your results. Your write-up should clearly reference the plots, tables, and metrics you generated.

- [ ] How does the performance of a **GCN** compare to that of an **R-GCN**?
- [ ] How much do relation types help link prediction, and when do they matter most?
- [ ] How do node feature choices affect results: `random` vs `one-hot`, and `pretrained` if you implemented description-based features as an extension?
- [ ] What do the PCA and t-SNE visualizations reveal about the structure of the learned node embeddings?
- [ ] Do drugs and diseases form meaningful clusters or separations in embedding space, and where do those patterns fail?
- [ ] Which biomedical entities or relation types seem to drive the most interesting clusters, outliers, or separations?
- [ ] How stable are your conclusions across different hyperparameter settings?
- [ ] What aspects of the graph structure or model behavior surprised you?
- [ ] What are the strengths and limitations of GNNs for biomedical data, especially when datasets are small, sparse, or synthetic?

---

## Expected Skills

By the end of this homework, you should be able to:

- Represent structured biomedical data as graphs
- Understand and apply GNN and R-GCN layers
- Handle multi-relation graphs for link prediction
- Evaluate graph models with appropriate metrics
- Reflect on the role of graph structure and node features in biomedical learning
