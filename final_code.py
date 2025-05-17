 from google.colab import files
uploaded = files.upload()

import pandas as pd

df = pd.read_csv("titanic.csv")
df.head()
# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load the dataset (تأكد من رفع الملف إلى Colab أولاً)
df = pd.read_csv("/content/titanic.csv")

# Step 3: Preview the data
print("First 5 rows:")
print(df.head())

print("
Info:")
print(df.info())

# Step 4: Drop irrelevant columns
df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Step 5: Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Step 6: Encode categorical features
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Step 7: Split features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 9: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 10: Show shapes
print("
Shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
import os
os.makedirs("Data/Results", exist_ok=True)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# تدريب النموذج
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# التوقعات
lr_preds = lr_model.predict(X_test_scaled)

# حساب الدقة
lr_acc = accuracy_score(y_test, lr_preds)
print("Logistic Regression Accuracy:", lr_acc)

# حفظ التوقعات
pd.DataFrame(lr_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_LogisticRegression_model.csv", index=False)
.., [19/11/46 07:18 م]
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# تدريب النموذج
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)  # بدون تحجيم لأن الشجرة ما تحتاجه

# التوقعات
tree_preds = tree_model.predict(X_test)

# حساب الدقة
tree_acc = accuracy_score(y_test, tree_preds)
print("Decision Tree Accuracy:", tree_acc)

# حفظ التوقعات
pd.DataFrame(tree_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_DecisionTree_model.csv", index=False)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# تدريب النموذج
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)  # ما يحتاج تحجيم

# التوقعات
rf_preds = rf_model.predict(X_test)

# حساب الدقة
rf_acc = accuracy_score(y_test, rf_preds)
print("Random Forest Accuracy:", rf_acc)

# حفظ التوقعات
pd.DataFrame(rf_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_RandomForest_model.csv", index=False)


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# تدريب النموذج
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)  # يحتاج تحجيم

# التوقعات
svm_preds = svm_model.predict(X_test_scaled)

# حساب الدقة
svm_acc = accuracy_score(y_test, svm_preds)
print("SVM Accuracy:", svm_acc)

# حفظ التوقعات
pd.DataFrame(svm_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_SVM_model.csv", index=False)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# تدريب النموذج
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)  # يحتاج تحجيم

# التوقعات
knn_preds = knn_model.predict(X_test_scaled)

# حساب الدقة
knn_acc = accuracy_score(y_test, knn_preds)
print("KNN Accuracy:", knn_acc)

# حفظ التوقعات
pd.DataFrame(knn_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_KNN_model.csv", index=False)


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

# تدريب النموذج
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)  # ما يحتاج تحجيم

# التوقعات
nb_preds = nb_model.predict(X_test)

# حساب الدقة
nb_acc = accuracy_score(y_test, nb_preds)
print("Naive Bayes Accuracy:", nb_acc)

# حفظ التوقعات
pd.DataFrame(nb_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_NaiveBayes_model.csv", index=False)


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# تدريب النموذج
ann_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
ann_model.fit(X_train_scaled, y_train)  # يحتاج تحجيم

# التوقعات
ann_preds = ann_model.predict(X_test_scaled)

# حساب الدقة
ann_acc = accuracy_score(y_test, ann_preds)
print("ANN Accuracy:", ann_acc)

# حفظ التوقعات
pd.DataFrame(ann_preds, columns=["Predicted"]).to_csv("Data/Results/predictions_ANN_model.csv", index=False)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# تدريب النموذج
linreg_model = LinearRegression()
linreg_model.fit(X_train_scaled, y_train)  # يحتاج تحجيم

# التوقعات (راح تكون أرقام بين 0 و 1)
linreg_preds = linreg_model.predict(X_test_scaled)

# نحول القيم إلى 0 و 1 (تصنيف)
linreg_binary = (linreg_preds > 0.5).astype(int)

# حساب الدقة
linreg_acc = accuracy_score(y_test, linreg_binary)
print("Linear Regression Accuracy:", linreg_acc)

# حفظ التوقعات
pd.DataFrame(linreg_binary, columns=["Predicted"]).to_csv("Data/Results/predictions_LinearRegression_model.csv", index=False)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# جمع الدقة يدوياً أو من المتغيرات اللي حسبناها
accuracies = {
    "SVM": svm_acc,
    "KNN": knn_acc,
    "Random Forest": rf_acc,
    "Decision Tree": tree_acc,
    "Linear Regression": linreg_acc,
    "Logistic Regression": lr_acc,
    "ANN": ann_acc,
    "Naive Bayes": nb_acc
}

# تحويلها إلى DataFrame
df_acc = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
df_acc = df_acc.sort_values(by="Accuracy", ascending=False)

# رسم الرسم البياني
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
barplot = sns.barplot(x="Model", y="Accuracy", data=df_acc, palette="Set2")

plt.title("Model Accuracy Comparison", fontsize=16)
plt.xlabel("Model")
plt.ylabel("Accuracy")

# عرض القيم فوق الأعمدة
for i, (model, acc) in enumerate(zip(df_acc["Model"], df_acc["Accuracy"])):
    barplot.text(i, acc + 0.002, f"{acc:.4f}", ha='center', color='black')

# إضافة جدول أسفل الشكل
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


from google.colab import files
files.download("model_accuracy_chart.png")


import os
import shutil
import pandas as pd

# 1. إنشاء المجلدات المطلوبة
os.makedirs("Data/original_data", exist_ok=True)
os.makedirs("Data/preprocessed_data", exist_ok=True)
os.makedirs("Data/Results", exist_ok=True)

# 2. حفظ البيانات الأصلية
shutil.copy("Titanic-Dataset.csv", "Data/original_data/Titanic-Dataset.csv")

# 3. حفظ بيانات المعالجة المسبقة
pd.DataFrame(X_train, columns=X.columns).to_csv("Data/preprocessed_data/X.csv", index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv("Data/preprocessed_data/X_test.csv", index=False)
pd.DataFrame(y_train, columns=["Survived"]).to_csv("Data/preprocessed_data/Y.csv", index=False)
pd.DataFrame(y_test, columns=["Survived"]).to_csv("Data/preprocessed_data/Y_test.csv", index=False)

# 4. حفظ نسخة من y_test داخل مجلد Results 
pd.DataFrame(y_test, columns=["Survived"]).to_csv("Data/Results/Y_test.csv", index=False)

print("✅ All files saved and organized correctly.")


import shutil
from google.colab import files

# ضغط مجلد Data بالكامل في ملف ZIP
shutil.make_archive("Titanic_Project_Files", 'zip', "Data")

# تحميل الملف المضغوط
files.download("Titanic_Project_Files.zip") 