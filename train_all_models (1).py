
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
import matplotlib.pyplot as plt
import seaborn as sns
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

# Dictionary to collect model accuracies
accuracies = {}

# 1. Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)
acc = accuracy_score(y_test, lr_preds)
accuracies["Logistic Regression"] = acc
pd.DataFrame(lr_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_LogisticRegression_model.csv", index=False)

# 2. SVM
svm = SVC()
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)
acc = accuracy_score(y_test, svm_preds)
accuracies["SVM"] = acc
pd.DataFrame(svm_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_SVM_model.csv", index=False)

# 3. KNN
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn_preds = knn.predict(X_test_scaled)
acc = accuracy_score(y_test, knn_preds)
accuracies["KNN"] = acc
pd.DataFrame(knn_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_KNN_model.csv", index=False)

# 4. Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)
acc = accuracy_score(y_test, tree_preds)
accuracies["Decision Tree"] = acc
pd.DataFrame(tree_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_DecisionTree_model.csv", index=False)

# 5. Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
acc = accuracy_score(y_test, rf_preds)
accuracies["Random Forest"] = acc
pd.DataFrame(rf_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_RandomForest_model.csv", index=False)

# 6. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_preds = nb.predict(X_test)
acc = accuracy_score(y_test, nb_preds)
accuracies["Naive Bayes"] = acc
pd.DataFrame(nb_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_NaiveBayes_model.csv", index=False)

# 7. ANN
ann = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
ann.fit(X_train_scaled, y_train)
ann_preds = ann.predict(X_test_scaled)
acc = accuracy_score(y_test, ann_preds)
accuracies["ANN"] = acc
pd.DataFrame(ann_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_ANN_model.csv", index=False)

# 8. Linear Regression
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)
linreg_preds = linreg.predict(X_test_scaled)
linreg_binary = (linreg_preds > 0.5).astype(int)
acc = accuracy_score(y_test, linreg_binary)
accuracies["Linear Regression"] = acc
pd.DataFrame(linreg_binary, columns=["Predicted"]).to_csv("Data/Results/predictions_LinearRegression_model.csv", index=False)

# Plotting the accuracy chart with table
df_acc = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
df_acc = df_acc.sort_values(by="Accuracy", ascending=False)
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
barplot = sns.barplot(x="Model", y="Accuracy", data=df_acc, palette="Set2")

plt.title("Model Accuracy Comparison", fontsize=16)
plt.xlabel("Model")
plt.ylabel("Accuracy")

# Annotate bars
for i, (model, acc) in enumerate(zip(df_acc["Model"], df_acc["Accuracy"])):
    barplot.text(i, acc + 0.002, f"{acc:.4f}", ha='center', color='black')

# Add accuracy table
plt.table(cellText=df_acc.values,
          colLabels=df_acc.columns,
          cellLoc='center',
          loc='bottom',
          bbox=[0.0, -0.4, 1, 0.3])

plt.subplots_adjust(bottom=0.35)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_accuracy_chart.png", dpi=300)
plt.show()
