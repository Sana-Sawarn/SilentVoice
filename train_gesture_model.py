import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load collected gesture data
df = pd.read_csv('gesture_data.csv')

# Split into features and labels
X = df.drop('label', axis=1)
y = df['label']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open("gesture_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Gesture model trained and saved as gesture_classifier.pkl")
