import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(filepath='data/creditcard.csv'):
    df = pd.read_csv(filepath)
    df = df.sample(n=50000, random_state=42)
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scale 'Amount' and 'Time'
    scaler = StandardScaler()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

    # SMOTE to balance data
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test, scaler
