# Credit Card Fraud Detection

This project focuses on detecting credit card fraud using machine learning techniques. It utilizes a dataset from Kaggle ([https://www.kaggle.com/code/drapraks/credit-card-fraud-detection](https://www.kaggle.com/code/drapraks/credit-card-fraud-detection)) to build a predictive model that can classify transactions as either fraudulent or legitimate.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Libraries Used](#libraries-used)
- [Results](#results)
- [Installation](#installation)

## Introduction

Credit card fraud is a significant problem that affects individuals and financial institutions alike. This project aims to develop an effective fraud detection system using machine learning. By analyzing various transaction features, the model can identify patterns indicative of fraudulent activity, helping to minimize losses and enhance security.

## Dataset

The dataset used in this project is obtained from Kaggle:

-   **Source:** [https://www.kaggle.com/code/drapraks/credit-card-fraud-detection](https://www.kaggle.com/code/drapraks/credit-card-fraud-detection)

It contains credit card transactions with features such as time, amount, and anonymized features (V1 to V28), and a class label indicating whether a transaction is fraudulent (1) or legitimate (0). Due to confidentiality, the original features are transformed using PCA.

## Methodology

The project follows these key steps:

1.  **Data Loading and Exploration:**
    * Loading the dataset using pandas.
    * Exploring the dataset's structure, missing values, and class distribution.
2.  **Data Preprocessing:**
    * Scaling the 'Amount' and 'Time' features to ensure they are on a similar scale as other features.
    * Splitting the data into training and testing sets to evaluate the model's performance.
3.  **Model Training:**
    * Training a Logistic Regression model on the training data.
4.  **Model Evaluation:**
    * Evaluating the model's performance using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC curve.
    * Visualizing the results to better understand the model's effectiveness.
5.  **Handling Class Imbalance:**
    * Addressing the class imbalance issue (where fraudulent transactions are far fewer than legitimate ones) using the Synthetic Minority Over-sampling Technique (SMOTE).
    * Retraining and evaluating the model on the balanced dataset to improve fraud detection.

## Libraries Used

-   pandas
-   numpy
-   sklearn (scikit-learn)
-   matplotlib
-   seaborn
-   imblearn (for SMOTE)

## Results

The Logistic Regression model's performance is evaluated using various metrics. The key findings include:

-   **Baseline Model:** The initial model showed good accuracy but poor performance in identifying fraudulent transactions due to class imbalance.
-   **SMOTE Implementation:** After applying SMOTE to balance the classes, the model's ability to detect fraudulent transactions significantly improved, with higher recall and F1-score for the positive class.
-   **AUC-ROC Curve:** The AUC-ROC curve visualizes the trade-off between true positive rate and false positive rate, providing further insight into the model's classification performance.

## Installation

To run this project, ensure you have Python 3.x installed. You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
