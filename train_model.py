from processor import DataProcessor
from sklearn.ensemble import RandomForestRegressor
import joblib  # This replaces tensorflow's save function
import numpy as np

# 1. Prepare Data
print("Loading Data...")
processor = DataProcessor('data/PJME_hourly.csv')
data, scaler = processor.load_and_process()

# Use 20,000 rows (Random Forest handles this easily)
data = data[:20000] 

SEQ_LENGTH = 24  # Look back 24 hours
X, y = processor.create_sequences(data, SEQ_LENGTH)

# Flatten y to make sure it's the right shape (N,) instead of (N, 1)
y = y.ravel()

# Split Train/Test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 2. Build Model (Random Forest)
print("Training Random Forest... (This is stable on Mac)")
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# 3. Save the model
joblib.dump(model, 'grid_ai_model.pkl') # Note the .pkl extension
print("✅ Success! Model saved as grid_ai_model.pkl")