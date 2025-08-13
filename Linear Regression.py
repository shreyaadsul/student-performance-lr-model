# 📌 Student Performance Prediction using Logistic Regression

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Step 2: Load Dataset
# If you have your own CSV file, replace with: pd.read_csv("your_file.csv")
data = pd.DataFrame({
    "Hours_Study": [5, 10, 15, 7, 9, 12, 3, 8, 16, 14, 2, 11, 4, 13, 6],
    "Attendance": [75, 88, 95, 80, 82, 90, 60, 85, 98, 96, 50, 89, 65, 94, 70],
    "Prev_Score": [60, 78, 85, 70, 72, 88, 50, 74, 92, 90, 45, 86, 55, 89, 65],
    "Sleep_Hours": [6, 7, 8, 6, 7, 8, 5, 6, 8, 7, 4, 8, 5, 7, 6],
    "Pass": [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0]  # 0 = Fail, 1 = Pass
})

print("First 5 rows of dataset:")
print(data.head())

# Step 3: Split Features & Target
X = data.drop("Pass", axis=1)
y = data["Pass"]

# Step 4: Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Create & Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Step 8: Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Step 9: Predict on New Data
# Step 9: Predict on New Data (Fixed)
new_student = pd.DataFrame([[10, 85, 78, 7]], columns=["Hours_Study", "Attendance", "Prev_Score", "Sleep_Hours"])
new_student_scaled = scaler.transform(new_student)
prediction = model.predict(new_student_scaled)
probability = model.predict_proba(new_student_scaled)[0][1]

print(f"\nNew Student Prediction: {'Pass' if prediction[0] == 1 else 'Fail'}")
print(f"Pass Probability: {probability*100:.2f}%")

