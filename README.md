# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent transactions using real-world credit card data. This project includes full EDA, preprocessing, model training, evaluation, and a deployment-ready app.

---
## 📁 Dataset

The dataset used in this project is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

Download it manually and place it in the `/data` folder as:



## 🚀 Project Overview

- **Goal**: Detect credit card frauds using ML models like Random Forest.
- **Techniques**: Data balancing (SMOTE), scaling, hyperparameter tuning.
- **Deployment**: Flask-based web app in `/app/app.py`.

---

## 🗂️ Folder Structure
credit-card-fraud-detection/
│
├── data/
│ └── creditcard.csv # Original dataset
│
├── notebooks/
│ └── fraud_detection.ipynb # EDA + model training
│
├── models/
│ └── best_model.pkl # Saved model
│
├── app/
│ ├── app.py # Streamlit or Flask app
│ ├── utils.py # Helper functions
│
├── requirements.txt
└── README.md

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Imbalanced**: Only ~0.17% are fraud

## ⚙️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

