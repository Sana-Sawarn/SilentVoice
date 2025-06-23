# create_dummy_model.py

import pickle
from dummy_model import DummyASLModel

model = DummyASLModel()

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Dummy model saved successfully.")
