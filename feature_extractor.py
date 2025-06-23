import numpy as np

def extract_features(landmarks):
    features = []
    for lm in landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features)
