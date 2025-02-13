import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import IPython
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ✅ Ensure IPython is installed for SHAP visualization
IPython.get_ipython()

# 📂 Load dataset
data_path = '/content/drive/MyDrive/dataset/finance/fraud detection/train_transaction.csv'
df = pd.read_csv(data_path)

# Debugging: Check dataset structure
print("Dataset Info:")
print(df.info())

# 🎯 Define target and features
target_col = 'isFraud'
X = df.drop(columns=[target_col])
y = df[target_col]

# 🔄 Convert categorical columns to numerical (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# 🏋️‍♂️ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Ensure training & test sets have the same features
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 🌲 Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🎯 Evaluate model
y_pred = model.predict(X_test)
print("\n🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# 💾 Save model using joblib
model_path = '/content/drive/MyDrive/fraud_detection_model.joblib'
joblib.dump(model, model_path)
print(f"\n✅ Model saved at: {model_path}")

# 🔍 Explainability using SHAP
explainer = shap.Explainer(model, X_train)  # Use training data
shap_values = explainer(X_test)

# ✅ Ensure test features match training features
X_test = X_test[X_train.columns]

# 💾 Save SHAP values & test samples
shap_values_path = '/content/drive/MyDrive/shap_values.npy'
np.save(shap_values_path, shap_values.values)
X_test.to_csv('/content/drive/MyDrive/shap_test_samples.csv', index=False)
print(f"\n✅ SHAP values saved at: {shap_values_path}")

# 📊 Plot SHAP summary
plt.figure(figsize=(10, 6))
shap.summary_plot(np.array(shap_values.values), X_test)
plt.show()
