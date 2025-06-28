import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # Fixed import
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# 1️⃣ Reading and preprocessing data
df = pd.read_csv("medical_records.csv")
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# 2️⃣ Split the data
X = df.drop('disease', axis=1)
y = df['disease']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

# 3️⃣ Balance data and train models
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)  # Fixed variable names

# Train Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_resampled, y_resampled)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)

# 4️⃣ Evaluate models
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_pred))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# 5️⃣ Save trained models
joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
print("\nModels saved successfully!")

