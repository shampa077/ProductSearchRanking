# -*- coding: utf-8 -*-
"""
Created on Tue May 16 22:49:24 2017

@author: shampa
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('sales_real_201706222346.csv')
X = dataset.iloc[:, 1:5].values
y1 = dataset.iloc[:, 6].values
            
y=[]
p=[]           
for i in range(len(y1)):
    if y1[i]=='callcentre\r':
        y.append(0)
    else:
        y.append(1)
    p.append(float(X[i,3].replace(",","")))
    
X=pd.DataFrame(X)
X[3]=p
y=pd.DataFrame(y)        

#==============================================================================
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# imputer = imputer.fit(X[:, 3])
# X[:, 3] = imputer.transform(X[:, 3])
#==============================================================================


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

X[1] = labelencoder_X.fit_transform(X[1])
X[2] = labelencoder_X.fit_transform(X[2])

onehotencoder = OneHotEncoder(categorical_features = [1,2])
X = onehotencoder.fit_transform(X).toarray()


#==============================================================================

# Splitting the dataset into the Training set trand Test set
#X_train_r,X_test_r, y_train_r, y_test_r = train_test_split(k,m, test_size = 1/3, random_state = 0)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_test=  sc_y.fit_transform(y_test)
#y_train = sc_y.fit_transform(y_train)

k=pd.DataFrame(X_test)
m=pd.DataFrame(y_test)


# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression

# instantiate model
logreg = LogisticRegression()

# fit model
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

k=pd.DataFrame(y_test)
m=pd.DataFrame(y_pred)

from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

np.count_nonzero(y_pred)
np.count_nonzero(y_test)
## Fitting Simple Linear Regression to the Training set
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)
#
## Predicting the Test set results
#y_pred = regressor.predict(X_test)

y_pred_prob_1 = logreg.predict_proba(X_test)[:, 1]

y_pred_prob_0 = logreg.predict_proba(X_test)[:, 0]

real=y_test.values

valueMCount=[]
valueNMCount=[]
for i in range(0,len(y_test)):
    if y_pred[i]==real[i]:
        valueMCount.append(int(i))
    else:
        valueNMCount.append(int(i))
print("Match",len(valueMCount))
print("Not Match",len(valueNMCount))
plt.cla() 
plt.scatter(valueNMCount, valueNMCount, color = 'red',label='Match')  
plt.scatter(valueMCount, valueMCount, color = 'blue',label='No Match')
#plt.plot(value, real, color = 'red')
plt.title('Real vs Prediction')
plt.xlabel('Index')
plt.ylabel('Sell Type')
legend = plt.legend(loc='upper left', shadow=True)
plt.show()