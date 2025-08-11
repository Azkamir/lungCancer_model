# Lung Cancer Prediction

This repository contains code and data for predicting lung cancer levels using machine learning techniques. 
The project involves data preprocessing, feature selection, class imbalance handling, and model training with ensemble classifiers.


## Project Overview

The goal is to classify lung cancer (two different datasets) based on clinical features from the dataset. 
Several models including Logistic Regression, Random Forest, and SVM are used in a voting ensemble to improve classification accuracy.

## Dataset
- Description: Clinical dataset with features related to lung cancer patients.
- Target Variable: `Level` (categorical: Low, Medium, High), 'Lung Cancer': (Yes,No)
- Data preprocessing includes encoding categorical variables and handling missing values.

## Dependencies

The code requires the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imblearn

Software Versions:
Python: 3.9
scikit-learn: 1.2.1
numpy: 1.26.4

Key Steps in the Pipeline
Data Cleaning: Dropping irrelevant columns (Patient Id) and encoding target labels.

Visualization: Distribution plots, heatmaps, pairplots, and boxplots for exploratory data analysis.

Feature Selection: SelectKBest with ANOVA F-value to pick top 9 features.

Handling Imbalance: SMOTE can be integrated (code snippet included for oversampling).

Modeling: Ensemble VotingClassifier combining Logistic Regression, Random Forest, and SVM.

Evaluation: Confusion matrix, classification report, accuracy score.



