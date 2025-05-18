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

## Evaluation Metrics

Model performance was evaluated primarily using accuracy. Additional metrics such as precision, recall, and F1-score can be calculated from the saved predictions.

A bar chart was created to visually compare the accuracy of all models.

## Data Preprocessing

- Missing values in the 'Age' column were filled with the mean age.  
- Categorical variables such as 'Sex' and 'Embarked' were encoded.  
- Irrelevant columns like 'PassengerId', 'Name', 'Ticket', and 'Cabin' were dropped.  
- Data was split into training and testing sets with an 80-20 ratio.

## Future Work
 • Hyperparameter Tuning: Implement grid search or randomized search to optimize model parameters for better accuracy.
 • Deep Learning: Explore neural network architectures to compare their performance with traditional ML models.
 • Model Deployment: Develop a web-based application to allow real-time survival predictions based on user input.
 • Additional Features: Incorporate external data or engineer new features to enhance predictive power.

## Data Source

https://www.kaggle.com/datasets/yasserh/titanic-dataset

## Author

Ebtsam Althaqafi
