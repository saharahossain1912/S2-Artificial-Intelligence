import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("Heart-Disease/heart.csv")

X = df[["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)
log_pred = model1.predict(X_test)

print("LOGISTIC REGRESSION RESULTS")
print(f"Accuracy : {accuracy_score(y_test, log_pred):.4f}")
print(f"Precision: {precision_score(y_test, log_pred):.4f}")
print(f"Recall   : {recall_score(y_test, log_pred):.4f}")
print(f"F1-score : {f1_score(y_test, log_pred):.4f}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)

print()
print("KNN CLASSIFICATION RESULTS")
print(f"Accuracy : {accuracy_score(y_test, knn_pred):.4f}")
print(f"Precision: {precision_score(y_test, knn_pred):.4f}")
print(f"Recall   : {recall_score(y_test, knn_pred):.4f}")
print(f"F1-score : {f1_score(y_test, knn_pred):.4f}")

log_accuracy = accuracy_score(y_test, log_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)

log_f1 = f1_score(y_test, log_pred)
knn_f1 = f1_score(y_test, knn_pred)

print()
print("MODEL COMPARISON")
if knn_accuracy > log_accuracy:
    print("Based on accuracy, KNN performed better.")
else:
    print("Based on accuracy, Logistic Regression performed better.")
if knn_f1 > log_f1:
    print("Based on F1-score, KNN performed better.")
else:
    print("Based on F1-score, Logistic Regression performed better.")
