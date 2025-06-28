# create_data.py - Generate sample medical data
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 1000

print("Creating sample medical data...")

# Generate realistic medical data
ages = np.random.normal(50, 15, n_samples).astype(int)
ages = np.clip(ages, 18, 90)  # Keep ages between 18-90

# Generate correlated features (more realistic)
data = []
for i in range(n_samples):
    age = ages[i]
    sex = np.random.choice(['male', 'female'])
    
    # Blood pressure tends to increase with age
    bp_base = 90 + (age - 18) * 0.8
    blood_pressure = int(np.random.normal(bp_base, 15))
    blood_pressure = np.clip(blood_pressure, 80, 200)
    
    # Cholesterol also tends to increase with age
    chol_base = 150 + (age - 18) * 1.2
    cholesterol = int(np.random.normal(chol_base, 30))
    cholesterol = np.clip(cholesterol, 100, 400)
    
    # Sugar levels
    sugar_base = 90 + (age - 18) * 0.5
    sugar = int(np.random.normal(sugar_base, 20))
    sugar = np.clip(sugar, 50, 300)
    
    # Disease probability increases with risk factors
    risk_score = 0
    if age > 60: risk_score += 0.3
    if blood_pressure > 140: risk_score += 0.2
    if cholesterol > 240: risk_score += 0.2
    if sugar > 140: risk_score += 0.2
    if sex == 'male': risk_score += 0.1
    
    # Add some randomness
    risk_score += np.random.normal(0, 0.2)
    disease = 1 if risk_score > 0.5 else 0
    
    data.append({
        'age': age,
        'sex': sex,
        'blood_pressure': blood_pressure,
        'cholesterol': cholesterol,
        'sugar': sugar,
        'disease': disease
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('medical_records.csv', index=False)

print(f"✅ Sample data created successfully!")
print(f"📊 Dataset shape: {df.shape}")
print(f"📈 Disease distribution:")
print(df['disease'].value_counts())
print(f"📋 First few rows:")
print(df.head())
print("\nData saved as: medical_records.csv")
print("Now you can run: python3 train_models.py")
