import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('/Users/small_yellow/Documents/ECS 170/Final Project/mushrooms.csv')

# Replace categorical values with numerical values
replace_dict = {
    'cap-shape': {'b': 0, 'c': 1, 'x': 2, 'f': 3, 'k': 4, 's': 5},
    'cap-surface': {'f': 0, 'g': 1, 'y': 2, 's': 3},
    'cap-color': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9},
    'bruises': {'t': 0, 'f': 1},
    'odor': {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8},
    'gill-attachment': {'a': 0, 'd': 1, 'f': 2, 'n': 3},
    'gill-spacing': {'c': 0, 'w': 1, 'd': 2},
    'gill-size': {'b': 0, 'n': 1},
    'gill-color': {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'g': 4, 'r': 5, 'o': 6, 'p': 7, 'u': 8, 'e': 9, 'w': 10, 'y': 11},
    'stalk-shape': {'e': 0, 't': 1},
    'stalk-root': {'b': 0, 'c': 1, 'u': 2, 'e': 3, 'z': 4, 'r': 5, '?': 6},
    'stalk-surface-above-ring': {'f': 0, 'y': 1, 'k': 2, 's': 3},
    'stalk-surface-below-ring': {'f': 0, 'y': 1, 'k': 2, 's': 3},
    'stalk-color-above-ring': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
    'stalk-color-below-ring': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
    'veil-type': {'p': 0, 'u': 1},
    'veil-color': {'n': 0, 'o': 1, 'w': 2, 'y': 3},
    'ring-number': {'n': 0, 'o': 1, 't': 2},
    'ring-type': {'c': 0, 'e': 1, 'f': 2, 'l': 3, 'n': 4, 'p': 5, 's': 6, 'z': 7},
    'spore-print-color': {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'r': 4, 'o': 5, 'u': 6, 'w': 7, 'y': 8},
    'population': {'a': 0, 'c': 1, 'n': 2, 's': 3, 'v': 4, 'y': 5},
    'habitat': {'g': 0, 'l': 1, 'm': 2, 'p': 3, 'u': 4, 'w': 5, 'd': 6},
    'class': {'p': 0, 'e': 1}
}

data.replace(replace_dict, inplace=True)

# Define X and y with all the relevant features
X = data.drop(columns='class')
y = data['class']

# Build and fit the Categorical Naive Bayes model
model = CategoricalNB()
model.fit(X, y)

# Predict with the model
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Show accuracy
report = classification_report(y, y_pred, output_dict=True)
print(classification_report(y, y_pred))

# Convert classification report to DataFrame
report_df = pd.DataFrame(report).transpose()

# Plot classification report metrics
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Classification Report Metrics', fontsize=16)

# Precision
sns.barplot(x=report_df.index[:-3], y=report_df['precision'][:-3], ax=ax[0, 0], palette='viridis')
ax[0, 0].set_title('Precision')
ax[0, 0].set_ylabel('Score')
ax[0, 0].set_xticklabels(ax[0, 0].get_xticklabels(), rotation=45)

# Recall
sns.barplot(x=report_df.index[:-3], y=report_df['recall'][:-3], ax=ax[0, 1], palette='viridis')
ax[0, 1].set_title('Recall')
ax[0, 1].set_ylabel('Score')
ax[0, 1].set_xticklabels(ax[0, 1].get_xticklabels(), rotation=45)

# F1-Score
sns.barplot(x=report_df.index[:-3], y=report_df['f1-score'][:-3], ax=ax[1, 0], palette='viridis')
ax[1, 0].set_title('F1-Score')
ax[1, 0].set_ylabel('Score')
ax[1, 0].set_xticklabels(ax[1, 0].get_xticklabels(), rotation=45)

# Support
sns.barplot(x=report_df.index[:-3], y=report_df['support'][:-3], ax=ax[1, 1], palette='viridis')
ax[1, 1].set_title('Support')
ax[1, 1].set_ylabel('Count')
ax[1, 1].set_xticklabels(ax[1, 1].get_xticklabels(), rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Confusion Matrix
mat = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(mat)

# Visualize the confusion matrix with a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=['Poisonous', 'Edible'], yticklabels=['Poisonous', 'Edible'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Calculate and print Mean Squared Error
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse}")
