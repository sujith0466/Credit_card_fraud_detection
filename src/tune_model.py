from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_random_forest(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(rf, param_dist, n_iter=10, scoring='f1', cv=3, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best Params: {search.best_params_}")
    return search.best_estimator_
