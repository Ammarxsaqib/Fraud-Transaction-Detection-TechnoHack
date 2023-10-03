import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
import warnings

# Suppress FutureWarnings related to is_categorical_dtype
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the data
data = pd.read_csv('tested.csv')

# Data Preprocessing
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

X = data.drop('Class', axis=1)
y = data['Class']

# Handling Class Imbalance
oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model (Random Forest with Simplified Hyperparameters)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)

# Model Evaluation
y_pred = rf_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Feature Importance Plot
feature_importances = rf_model.feature_importances_
features = X.columns
sorted_idx = np.argsort(feature_importances)

plt.figure(figsize=(16, 8))
plt.subplot(2, 3, 1)
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')

# ROC Curve
y_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.subplot(2, 3, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.subplot(2, 3, 3)
plt.plot(recall, precision, color='darkorange', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# Histograms of Features
plt.subplot(2, 3, 4)
sns.histplot(data, x='V1', hue='Class', bins=50, kde=True, common_norm=False)
plt.title('Histogram of V1 by Class')

plt.subplot(2, 3, 5)
sns.histplot(data, x='V17', hue='Class', bins=50, kde=True, common_norm=False)
plt.title('Histogram of V17 by Class')

# Box Plots of Features
plt.subplot(2, 3, 6)
sns.boxplot(x='Class', y='V14', data=data)
plt.title('Box Plot of V14 by Class')

plt.tight_layout()
plt.show()
