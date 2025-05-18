# Titanic-ML-Project

This project applies various machine learning models to predict passenger survival on the Titanic dataset.

## Overview

The Titanic dataset contains real data of passengers from the 1912 disaster. The goal is to classify whether a passenger survived based on features like age, class, fare, and sex.

## Models Used

- Logistic Regression  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Naive Bayes  
- Artificial Neural Network (ANN)  
- Linear Regression (as classifier)

## File Structure
Titanic-ML-Project/
├── train_all_models.py            # Python script to train all models
├── model_accuracy_chart.png       # Accuracy bar chart with table
└── Data/
├── original_data/
│   └── titanic.csv
├── preprocessed_data/
│   ├── X.csv
│   ├── X_test.csv
│   ├── Y.csv
│   └── Y_test.csv
└── Results/
├── predictions_LogisticRegression_model.csv
├── predictions_SVM_model.csv
├── predictions_KNN_model.csv
├── predictions_DecisionTree_model.csv
├── predictions_RandomForest_model.csv
├── predictions_NaiveBayes_model.csv
├── predictions_ANN_model.csv
├── predictions_LinearRegression_model.csv
└── Y_test.csv

## How to Run

1. Run train_all_models.py to train all models and generate predictions.
2. Or open your_code.ipynb in Google Colab for full step-by-step execution.

## Future Work
 • Hyperparameter Tuning: Implement grid search or randomized search to optimize model parameters for better accuracy.
 • Deep Learning: Explore neural network architectures to compare their performance with traditional ML models.
 • Model Deployment: Develop a web-based application to allow real-time survival predictions based on user input.
 • Additional Features: Incorporate external data or engineer new features to enhance predictive power.

## Data Source

https://www.kaggle.com/datasets/yasserh/titanic-dataset

## Author

Ebtsam Althaqafi
