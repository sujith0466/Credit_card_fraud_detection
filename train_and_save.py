from src.data_preprocessing import preprocess_data
from src.model_training import train_random_forest
from src.evaluate_model import evaluate_model
import joblib
import os

# Make sure models directory exists
os.makedirs('models', exist_ok=True)

# Step 1: Preprocess
X_train, X_test, y_train, y_test, scaler = preprocess_data('data/creditcard.csv')

# Step 2: Train
model = train_random_forest(X_train, y_train, 'models/random_forest_model.pkl')

# Step 3: Save scaler
joblib.dump(scaler, 'models/scaler.pkl')

# Step 4: Evaluate
evaluate_model(model, X_test, y_test)
