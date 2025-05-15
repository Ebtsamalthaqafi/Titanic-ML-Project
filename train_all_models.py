
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os

# Load and preprocess data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
os.makedirs("Data/Results", exist_ok=True)

# 1. Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))
pd.DataFrame(lr_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_LogisticRegression_model.csv", index=False)

# 2. SVM
svm = SVC()
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))
pd.DataFrame(svm_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_SVM_model.csv", index=False)

# 3. KNN
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn_preds = knn.predict(X_test_scaled)
print("KNN Accuracy:", accuracy_score(y_test, knn_preds))
pd.DataFrame(knn_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_KNN_model.csv", index=False)

# 4. Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_preds))
pd.DataFrame(tree_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_DecisionTree_model.csv", index=False)

# 5. Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
pd.DataFrame(rf_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_RandomForest_model.csv", index=False)

# 6. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_preds = nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_preds))
pd.DataFrame(nb_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_NaiveBayes_model.csv", index=False)

# 7. ANN
ann = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
ann.fit(X_train_scaled, y_train)
ann_preds = ann.predict(X_test_scaled)
print("ANN Accuracy:", accuracy_score(y_test, ann_preds))
pd.DataFrame(ann_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_ANN_model.csv", index=False)

# 8. Linear Regression
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)
linreg_preds = linreg.predict(X_test_scaled)
linreg_binary = (linreg_preds > 0.5).astype(int)
print("Linear Regression Accuracy:", accuracy_score(y_test, linreg_binary))
pd.DataFrame(linreg_binary, columns=["Predicted"]).to_csv("Data/Results/predictions_LinearRegression_model.csv", index=False)
