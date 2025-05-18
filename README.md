
# Titanic Machine Learning Project

## 🧾 Overview
This project applies supervised machine learning techniques to predict survival on the Titanic dataset. It demonstrates the full ML pipeline including data preprocessing, model training, evaluation, and visualization. Multiple models were tested to find the best-performing approach.

## 📊 Dataset
- **Name:** Titanic - Machine Learning from Disaster  
- **Source:** Kaggle (https://www.kaggle.com/datasets/yasserh/titanic-dataset)  
- **Samples:** 891  
- **Features:** Categorical and numerical  
- **Target:** Survived (0 = No, 1 = Yes)

---

## 🔧 Key Steps in the Project

### 📌 Data Preprocessing
- Removed irrelevant features (`Name`, `Ticket`, `Cabin`, `PassengerId`)
- Handled missing values (`Age`, `Fare`, `Embarked`)
- Encoded categorical variables (`Sex`, `Embarked`)
- Standardized numerical features using `StandardScaler`

### 🤖 Modeling
- Trained multiple machine learning models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Naive Bayes
  - Artificial Neural Network (ANN)
  - Linear Regression (for comparison)

### 📈 Model Evaluation
- Used accuracy as the primary evaluation metric.
- Evaluated each model’s performance on a test set (20% split).
- Tracked predictions and stored them in the `Results/` folder.

### 🧠 Model Interpretability
- Analyzed which features influenced model outcomes.
- Observed how categorical and numerical variables contributed to survival prediction.

### 📊 Visualization
- Used **Seaborn** and **Matplotlib** to:
  - Plot countplots, histograms, and heatmaps during EDA
  - Create a bar chart to compare model accuracies
  - Annotate bars with accuracy values and provide a summary table

### 💾 Model Saving
- Saved prediction results as `.csv` files in the `Data/Results/` folder.
- Included test labels `Y_test.csv` for performance checking.

---

## 🚀 Features and Results
- End-to-end ML pipeline implemented in Python
- Preprocessing and model results clearly separated
- **Best model:** Decision Tree (Accuracy: 83.24%)
- All model results stored in `Results` folder

---

## 🧪 Multiple Models
Each model was trained and tested separately. Results were compared using accuracy scores to identify the top performer.

## 🧬 Ensemble Learning
Random Forest, as an ensemble method, was included and showed strong performance.

## 🔍 Model Interpretability
Simple models like Decision Tree and Logistic Regression helped explain which features had the highest impact on predictions.

## 📊 Evaluation Metrics
- **Primary metric:** Accuracy
- Results visualized using comparative bar plots
- Predictions saved and matched with true values for verification

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ebtsamalthaqafi/Titanic-ML-Project.git
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   *(or manually install: pandas, numpy, matplotlib, seaborn, scikit-learn)*

---

## ▶️ Usage
Run the notebook in [Google Colab](https://colab.research.google.com/) or any Jupyter environment:
- Main notebook: `your_code.ipynb`
- Outputs saved in `/Data` subfolders

---

## 🔮 Future Work
- Improve accuracy using hyperparameter tuning
- Try cross-validation instead of a single train/test split
- Explore deep learning models with more data
- Add ROC curves and confusion matrix for better evaluation

---

## ✅ Conclusion
This project showcases a complete ML workflow using the Titanic dataset.  
It demonstrates how data preprocessing, careful model selection, and result visualization can lead to strong predictive performance using classic ML algorithms.
