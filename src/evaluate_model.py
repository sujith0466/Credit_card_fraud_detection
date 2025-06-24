from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(" Classification Report:\n")
    print(classification_report(y_test, y_pred))

    print(f" ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print(f" PR-AUC Score: {average_precision_score(y_test, y_proba):.4f}")
