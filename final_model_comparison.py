
# ابدأي من هنا بنسخ كل الكود اللي كتبتيه سابقاً
# import المكتبات
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

# تحميل البيانات
df = pd.read_csv("/content/Data/original_data/titanic.csv")

# معالجة البيانات
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = df[features]
Y = df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# تدريب النماذج وحساب الدقة

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, Y_train)
svm_predictions = svm_model.predict(X_test)
svm_acc = accuracy_score(Y_test, svm_predictions)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, Y_train)
dt_predictions = dt_model.predict(X_test)
dt_acc = accuracy_score(Y_test, dt_predictions)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)
rf_predictions = rf_model.predict(X_test)
rf_acc = accuracy_score(Y_test, rf_predictions)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)
knn_predictions = knn_model.predict(X_test)
knn_acc = accuracy_score(Y_test, knn_predictions)

ann_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, Y_train, epochs=50, batch_size=16, verbose=0)
ann_predictions = (ann_model.predict(X_test) > 0.5).astype(int).flatten()
ann_acc = accuracy_score(Y_test, ann_predictions)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, Y_train)
lr_predictions = lr_model.predict(X_test)
lr_acc = accuracy_score(Y_test, lr_predictions)

nb_model = GaussianNB()
nb_model.fit(X_train, Y_train)
nb_predictions = nb_model.predict(X_test)
nb_acc = accuracy_score(Y_test, nb_predictions)

# جدول المقارنة
models = ['SVM', 'Decision Tree', 'Random Forest', 'KNN', 'ANN', 'Logistic Regression', 'Naive Bayes']
accuracies = [svm_acc, dt_acc, rf_acc, knn_acc, ann_acc, lr_acc, nb_acc]
comparison_df = pd.DataFrame({'Model': models, 'Accuracy': accuracies})
print(comparison_df)

# الرسم البياني
plt.figure(figsize=(10, 6))
plt.bar(comparison_df['Model'], comparison_df['Accuracy'], color='skyblue')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.savefig("/content/Data/Results/model_accuracy_plot.png")
plt.show()
