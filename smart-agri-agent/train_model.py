import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
data = pd.read_csv("data/crop.csv")

X = data.drop("label", axis=1)
y = data["label"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/crop_model.pkl", "wb"))

print("✅ Model trained and saved!")