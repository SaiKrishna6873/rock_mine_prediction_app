#Logistic Regression

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv('sonar_data.csv', header = None)
a = sonar_data.head()
b = sonar_data.shape
c = sonar_data.describe()
vc = sonar_data[60].value_counts()
print(vc)
mean = sonar_data.groupby(60).mean()
#print(mean)
# separating data and Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
#print(X)
#print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)
print(X.shape, X_train.shape, X_test.shape)
print(X_train)
print(Y_train)
model = LogisticRegression()
#training the Logistic Regression model with training data
model.fit(X_train, Y_train)
#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) 
print('Accuracy on training data : ', training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) 
print('Accuracy on test data :', test_data_accuracy)

import pickle
pickle_out = open("logistic_reg.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()

index = int(input('Enter a index number to pick the set of test data:'))
X_test_np = np.asarray(X_test)
X_test_1 = X_test_np.reshape(1, -1)
print(X_test_1)
'''
prediction = model.predict(X_test_1)
print(prediction)
if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a Mine')
'''