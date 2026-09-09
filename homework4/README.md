# Homework 4: CNNs vs Fully Connected Networks (FCNs)

## Overview

In this homework, you will explore the differences between Convolutional Neural Networks (CNNs)
and Fully Connected Networks (FCNs), also known as Multilayer Perceptrons or MLPs,
using an RNA sequence classification dataset.
You will:

- Load and process RNA sequence data from CSV files.
- Understand the architectural differences between CNNs and FCNs.
- Implement and train both CNN and FCN models for multiclass classification.
- Visualize intermediate activations and output predictions.
- Compare how each architecture handles local sequence patterns and generalizes to unseen data.

---

## Datasets

You will work with one primary dataset:
1. RNA Family Classification (Multiclass Classification)
- **Sequences**: Non-coding RNA nucleotide sequences
- **Labels**: RNA family labels such as `riboswitch`, `tRNA`, `miRNA`, and `5S_rRNA`
- **Task**: Predict the RNA family from the nucleotide sequence

The dataset will be loaded using the provided `HW4DataLoader` class.
It is split into separate CSV files for training, validation, and testing.

```text
data/
├── train.csv
├── val.csv
└── test.csv
```

Each CSV contains:

```text
sequence,sequence_length,label,label_idx
```

The model input is the RNA sequence. The target is `label_idx`.

---

## Installation

Install dependencies using pip:

1. **Clone** this repo (first time only):
   ```bash
   git clone git@github.com:brown-csci1851/stencil.git
   cd stencil/homework4
   ```
   If you already cloned it, update and move into the homework folder:
   ```bash
   cd stencil
   git pull
   cd homework4
   ```
2. Create virtual environment:
    ```bash
    python -m venv .hw4
    ```
3. Install dependencies:
    ```bash
    source .hw4/bin/activate (Linux/MacOS) or .\.hw4\Scripts\activate
    pip install -r requirements.txt
    ```

---

## Preprocessing and Augmentation

You will experiment with sequence preprocessing and encoding techniques:

- Converting RNA strings into model-ready tensors
- Padding or truncating sequences to a fixed length
- One-hot encoding nucleotide bases
- Handling ambiguous bases: real sequence databases may use letters such as `N`, `R`, `Y`, and `K` when the exact nucleotide is unknown or could be one of several bases. For this homework, map any non-`A/C/G/U` character to `N`.
- Comparing flattened sequence representations with 1D convolutional representations

These help prepare data for CNN or FCN processing and influence how models generalize.

---

## Tasks

You will complete the following:

### Data + Exploration
- [ ] Load the dataset and confirm split sizes (train/val/test)
- [ ] Inspect example RNA sequences from each class
- [ ] Plot class distributions
- [ ] Plot or summarize sequence length distributions
- [ ] Implement sequence preprocessing:
  - [ ] normalize nucleotide characters
  - [ ] pad or truncate sequences to a fixed length
  - [ ] one-hot encode sequences

### A) FCN baseline
- [ ] Build an FCN that classifies RNA sequences after flattening the encoded input
- [ ] Track input/output tensor shapes carefully
- [ ] Train and evaluate it on the dataset

### B) CNN classifier
- [ ] Build a 1D CNN that preserves local sequence structure
- [ ] Print intermediate tensor shapes through the forward pass
- [ ] Train and evaluate it on the same splits as the FCN classifier

### Evaluation + Analysis

For both models (FCN and CNN), report:
- [ ] Accuracy on val/test
- [ ] Macro-averaged F1 score on val/test
- [ ] Confusion matrix on test
- [ ] Training curves (loss + metric)

Visualization + interpretation:
- [ ] Visualize CNN intermediate activations or learned filters for a few examples
- [ ] Compare how FCN vs CNN responds to sequence length and class imbalance
- [ ] Explain what local sequence information is lost when flattening an encoded sequence

### Extension
- [ ] Try one additional modeling or preprocessing choice of your own
  (for example, a different fixed sequence length, k-mer features, dropout, batch normalization, or class-weighted loss)

---

## Final Reflection

You will then write a **2-3 page reflection** that includes **figures** and **interpretation** of your results. Your write-up should clearly reference the plots, tables, and metrics you generated (not just final numbers).

- [ ] How did performance differ between the FCN and CNN?
- [ ] Why might 1D CNNs work well for biological sequence classification?
- [ ] What local sequence information was preserved or lost in each architecture?
- [ ] What did you learn from visualizing intermediate activations or learned filters?
- [ ] Which modeling or preprocessing choice had the biggest effect on performance?

---

## Expected Skills

By the end of this homework, you should be able to:

- Implement both FCN and 1D CNN models for sequence classification.
- Convert variable-length biological sequences into fixed-size model inputs.
- Evaluate multiclass classification models using appropriate metrics.
- Visualize learned sequence features and interpret model behavior.
- Understand how convolutional architectures encode local patterns in biological sequences.
- Understand the limitations of flattening sequence inputs for dense networks.
