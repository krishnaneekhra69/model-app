import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample data (N, P, K, pH, EC)
X = np.array([
    [90, 40, 40, 6.5, 1.2],
    [20, 10, 10, 4.5, 0.8],
    [70, 50, 60, 6.8, 1.5],
    [10, 20, 30, 5.0, 0.7]
])
y = ['Fertile', 'Infertile', 'Fertile', 'Infertile']

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open('model.pkl', 'wb'))
print("✅ Model saved successfully!")
