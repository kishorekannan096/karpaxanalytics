#Importing the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Mall_Customers_Details.csv')
x = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Classification using LogisticRegression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=0)
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)

#Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)
print(cm,"\n",acc_score)

#Saving model as a pickle
import pickle
pickle.dump(logreg, open("logreg_model.sav", 'wb'))
pickle.dump(sc, open("scaled_model.sav", 'wb'))