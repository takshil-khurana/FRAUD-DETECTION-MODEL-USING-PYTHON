# Fraud Detection Project
# Dataset: Fraud.csv
# Author: Takshil Khurana (or User's name)

# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# --- 2. Load the Dataset ---
df = pd.read_csv('Fraud.csv')

# --- 3. Initial Exploration ---
print(df.info())
print(df.describe())
print(df['type'].value_counts())
print(df['isFraud'].value_counts())

# --- 4. Data Cleaning ---
# Drop columns with IDs
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# Encode 'type'
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Add balance difference features
df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
df['errorBalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']

# Check and remove any potential NaNs or infs
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# --- 5. Feature Selection ---
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# --- 6. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# --- 7. Handle Class Imbalance ---
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# --- 8. Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_res, y_res)

# --- 9. Evaluate Model ---
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# --- 10. Feature Importance ---
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
feat_imp.plot(kind='bar')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()


