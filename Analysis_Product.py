# -*- coding: utf-8 -*-
"""
Created on Tue May 16 22:49:24 2017

@author: shampa
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Importing the dataset
dataset = pd.read_csv('sales_info_rank_sale_201706231134.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
            

X=pd.DataFrame(X)
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
X[0] = labelencoder_X.fit_transform(X[0])
X[1] = labelencoder_X.fit_transform(X[1])

onehotencoder = OneHotEncoder(categorical_features = [0,1])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]
X1=X
X = X[:, 0:15]
X=pd.DataFrame(X)
X['15'] = pd.Series(X1[:,16], index=X.index)
X['16'] = pd.Series(X1[:,17], index=X.index)
X['17'] = pd.Series(X1[:,18], index=X.index)

#backward elimination
#X=X[:,[0,6,14,17,18]]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


value=[]
valueDumy=[]
valueSale=[]
for i in range(0,len(y_pred)):
    value.append(int(i))
    if y_pred[i]<0:
        valueSale.append(0)
    else:
        valueSale.append(math.ceil(y_pred[i]))
    valueDumy.append(0)
# Visualising the Test set results
plt.cla()
plt.plot(value, y_test, color = 'blue',label='original sell')
#plt.plot(value, valueDumy, color = 'black',label='original sell')
plt.plot(value, y_pred, color = 'red',label='predicted sell')
plt.plot(value, valueSale, color = 'black',label='corrected predicted sell')
plt.title('Sell Predict')
plt.xlabel('Index')
plt.ylabel('Sold Amount')
legend = plt.legend(loc='upper left', shadow=True)
plt.show()

#optimal model using backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((len(X),1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=X[:,[0,6,14,17,18]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

#The explained_variance_score computes the explained variance regression score.
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error,median_absolute_error,r2_score
explained_variance_score(y_test, y_pred)  
explained_variance_score(y_test, valueSale)
mean_absolute_error(y_test, y_pred)
mean_squared_error(y_test, y_pred)
median_absolute_error(y_test, y_pred)
r2_score(y_test, y_pred)  
