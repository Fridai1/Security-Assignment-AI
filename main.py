import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('SSL Renegotiation_dataset-002.csv') 
labels = pd.read_csv('SSL Renegotiation_labels.csv')
y = labels.iloc[:,1:2]



X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size = 0.2, random_state = 0)

regressor = LogisticRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)




score = regressor.score(y_test, y_pred)


