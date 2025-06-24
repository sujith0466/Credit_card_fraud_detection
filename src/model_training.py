from sklearn.ensemble import RandomForestClassifier
import joblib

def train_random_forest(X_train, y_train, save_path):
    model = RandomForestClassifier(n_estimators=50, random_state=42)  # 👈 Reduced trees
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model
