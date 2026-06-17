# Bankruptcy Prediction

A machine learning project that predicts company bankruptcy from financial indicators, built in Python. The project covers the full pipeline from data cleaning and exploratory analysis through to model training, evaluation, and selection on imbalanced data.

## Problem and motivation

Predicting whether a company is at risk of bankruptcy is a classic and high-value problem in finance. Lenders, investors, and analysts use these signals to manage risk and make better decisions. This project builds a classification model that estimates bankruptcy risk from a set of financial indicators, and focuses on handling the class imbalance that is typical of real-world bankruptcy data, where bankruptcies are rare compared to healthy companies.

## Data
- **Source:** Dataset provided from University
- **Size:** 96 columns and 6665 rows
- **Target:** A binary label indicating whether the company went bankrupt (bankruptcy?)
- **Note:** The dataset is imbalanced, with far fewer bankrupt companies than healthy ones, which shaped the modelling and evaluation choices below.


## Approach
1. Data cleaning: Handled missing values, checked data types, and removed or corrected anomalies.
2. Exploratory data analysis: Explored distributions, correlations between financial indicators, and the degree of class imbalance, using Matplotlib and Seaborn.
3. Feature preparation: Scaled features and selected features for training.
4. Handling imbalance: Using imblearn two datasets were created one with oversampling and one with undersampling 
5. Modelling: Trained Random Forest classifiers
6. Evaluation: Compared models using metrics suited to imbalanced data, focusing on precision, recall, and the precision-recall trade-off rather than accuracy alone.

## Tools
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn. Developed in Google Colab.

## Results
- Key metrics: 19 of the 95 features were selected for this model
- Interpretation: These features were selected from the correlation found during EDA.


## What I learned and would do next
This project deepened my understanding of why accuracy is misleading on imbalanced data, and how to choose metrics that reflect the real cost of missing a bankruptcy. Next, I would explore a specific improvement, in this case used imblearn to verify the training difference when using balanced data for oversampling and undersampling and how this selection affects false negatives and false positives.

## Repository structure
- notebook.ipynb - the full analysis and modelling
- README.md - this file
