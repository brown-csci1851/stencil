# Homework 1: Introduction to Regression and Classification

## Overview
In this homework, you will work with two datasets to explore foundational models in machine learning: linear regression and classification. 

You will:

- Load and inspect biomedical datasets.
- Perform classification (heart disease).
- Use classification models to learn important features.
- Perform regression (biological aging prediction).
- Learn foundations of feature selection and model evaluation.  

---

## Datasets

You will work with two datasets:
1. Heart Disease Dataset (Classification)
- **Goal**: Predict presence/absence of heart disease.
- **Features**: Demographics, cholesterol, ECG results, etc.

2. Biological Aging Dataset (Regression)
- **Goal**: Predict biological age using molecular and physiological features.
- **Source**: High-dimensional gene expression dataset (GSE139307).
- **Features**: A hundred genomic and biological markers, already preprocessed and normalized. The data provided can be further separated into four tissue types: blood WBC, blood leukocyte, saliva, and kidney.

---

## Installation

Install dependencies using pip:

1. **Clone** this repo:
   ```bash
   git clone git@github.com:brown-csci1851/stencil.git
   cd stencil/homework1
   ```
2. Create virtual environment:
    ```bash
    python -m venv .hw1
    ```
3. Install dependencies:
    ```bash
    source .hw1/bin/activate (Linux/MacOS) or .\.hw1\Scripts\activate (Windows-PowerShell)
    pip install -r requirements.txt
    ```

After creating and activating the virtual environment, select it as the Jupyter kernel in `src/playground.ipynb` to run the notebook using the same installed dependencies.

---

## Tasks

You will complete the TODOs in `src/model.py` and `src/playground.ipynb` to accomplish the following tasks:

- [ ] Load both datasets using your HW1DataLoader.  (The code is provided for you!)

### Heart Disease Dataset
- [ ] Visualize distributions of numeric features, class balance (heart disease), outliers and missing values.
- [ ] Train a logistic regression model on the heart disease dataset with and without feature transformation.
- [ ] Evaluate with accuracy, precision, recall, F1-score using K-fold cross-validation.

### Biological Aging Dataset
- [ ] Train a linear regression model on the biological aging dataset split by tissue type.
- [ ] Evaluate with MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), $R^2$ score.
- [ ] Evaluate generalizability of tissue-specific models to other tissue-types.
- [ ] Train more models with limited feature selection selecting maximally variable features and evaluate.
- [ ] Visualize predicted vs actual age plots.

---

## Common Machine Learning Practices

Throughout this assignment, make sure to follow these standard machine learning best practices:

* Never train your model on the full dataset — always split the data into training and validation sets.
* Always examine summary statistics of your data (for regression tasks, we want feature values to be similarly scaled).
* Always check for missing values and handle them appropriately.

---

## Final Reflection

You will then write a **2-3 page reflection** that includes **figures** and **interpretation** of your results. Your write-up should clearly reference the plots and metrics you generated (not just final numbers).

Answer the following questions (with figures/screenshots where relevant):
* What are the main characteristics of the heart disease dataset and its features?
* What do the feature distributions look like, how balanced are the classes (heart disease present/absent), and are there any outliers or missing values?
* What does your confusion matrix show, and what does your ROC curve indicate about performance?
* What were your cross-validation results (summarize in a table or plot)?
* What were your accuracy, precision, recall, F1-score, and ROC-AUC results?
* Which features were most important and least important for classification, and why?
* What are the main limitations of the dataset and the logistic regression model?
* How did feature scaling or transformations change your results (ROC-AUC and/or decision boundary)?

* What are the main characteristics of the biological aging dataset and the specific tissues present in the dataset?
* How well did tissue-specific models perform (MAE, RMSE, \(R^2\))?
* How well did models generalize across tissues (summarize in a heatmap or table: train tissue \(\rightarrow\) test tissue)?
* How closely did predicted ages match true ages (include predicted vs. actual scatter plots, at least one per tissue)?
* How did performance change when using limited feature sets (top variance 100 features) compared to full-feature models?
* What are at other feature selection strategies you could apply beyond selecting top-variance features?
* What are the main limitations of the dataset and regression models (data quality, assumptions, generalizability)?
* Which tissues generalized poorly, and why might that be?
---

## Expected Skills

By the end of this homework, you should be able to:

* Perform classification and regression using linear and logistic models.
* Explore feature selection and model evaluation in biological contexts.
* Visualize and interpret simple model predictions.

