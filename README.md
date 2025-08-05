Developed a fraud detection system to identify suspicious transactions in financial datasets using supervised machine learning techniques. The system leverages data preprocessing, feature engineering, class imbalance handling (via SMOTE), and a robust Random Forest Classifier to achieve accurate fraud predictions.

📊 Key Highlights:

📁 Dataset: "Fraud.csv" – real-world simulated financial transaction data

🧹 Data Cleaning & Feature Engineering:

Removed ID fields (nameOrig, nameDest)

Encoded transaction types using LabelEncoder

Created balance error features (errorBalanceOrig, errorBalanceDest)

⚖️ Class Imbalance Handling: Applied SMOTE to upsample fraud cases

🧠 Model: Trained a Random Forest Classifier with 100 estimators

📈 Evaluation:

Metrics: Confusion Matrix, Classification Report, ROC AUC Score

Visualized feature importance to interpret model decisions

📌 Outcomes:

Detected fraudulent transactions with high recall and precision

Successfully handled extreme class imbalance using SMOTE

Gained practical experience in end-to-end ML pipeline for anomaly detection
