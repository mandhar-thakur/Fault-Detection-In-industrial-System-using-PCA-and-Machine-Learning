import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

data = pd.read_csv('/content/Loan_defaultn.csv')

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Convert categorical features to numerical representations
data['HasMortgage'] = data['HasMortgage'].map({1: 'Yes', 0: 'No'})
data['HasDependents'] = data['HasDependents'].map({1: 'Yes', 0: 'No'})
data['HasCoSigner'] = data['HasCoSigner'].map({1: 'Yes', 0: 'No'})

# Get dummy variables for categorical features
HasMortgage_dummies = pd.get_dummies(data['HasMortgage'], prefix='HasMortgage')
HasDependents_dummies = pd.get_dummies(data['HasDependents'], prefix='HasDependents')
HasCoSigner_dummies = pd.get_dummies(data['HasCoSigner'], prefix='HasCoSigner')

# Combine numerical and dummy features
X = pd.concat([data[['CreditScore', 'DTIRatio', 'Income', 'LoanAmount']], HasMortgage_dummies, HasDependents_dummies, HasCoSigner_dummies], axis=1)
y = data['Default']

pca = PCA(n_components=0.95)
principalComponents = pca.fit_transform(X)

# Separate out the response variable from the features
y = principalComponents[:, -1]
y = (y > y.mean()).astype(int)
# Split the features into predictors and the response variable
X_train, X_test, y_train, y_test = train_test_split(principalComponents[:, :-1], y, test_size=0.2, random_state=0)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Find the IDs of defaulters

# Make predictions
y_pred = classifier.predict(principalComponents[:, :-1])

# Get the loan IDs
loan_ids = data['LoanID']

# Print the predicted loan IDs
print('Predicted loan IDs:', loan_ids[y_pred == 1])

# Calculate the total number of predicted defaulters
predicted_defaulters = len(loan_ids[y_pred == 1])
print('Total number of predicted defaulters:', predicted_defaulters)