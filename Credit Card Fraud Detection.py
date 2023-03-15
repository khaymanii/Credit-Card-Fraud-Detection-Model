
# Importing Libraries

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Data Collection and Preprocessing

credit_card_data = pd.read_csv('creditcard.csv')
credit_card_data.head()
credit_card_data.tail()
credit_card_data.info()
credit_card_data.isnull().sum()

# Distribution of legit and fraudulent transaction

credit_card_data['Class'].value_counts()


# seperating the data for analysis

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


print(legit.shape)
print(fraud.shape)


# Statistical measures of the data

legit.Amount.describe()

fraud.Amount.describe()

# Compare the values for both transactions

credit_card_data.groupby('Class').mean()


# Under_sampling to balance the class or labels

legit_sample = legit.sample(n = 492)


# Concatenating 2 dataframes

new_dataset = pd.concat([legit_sample, fraud], axis = 0)


new_dataset.head()
new_dataset.tail()
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()


# splitting into Features and Target

X = new_dataset.drop(columns='Class', axis = 1)
y = new_dataset['Class']

print(X)
print(y)


# Splitting into Train and Test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)


# Model Training

model = LogisticRegression()

# Training the model with training data

model.fit(X_train, y_train)


# Evaluation of Model based on accuracy of training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)

print("Accuracy on Training data : ", training_data_accuracy)


# Evaluation of Model based on accuracy of test data

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)


print("Accuracy on Test data : ", test_data_accuracy)

