#Multiple Linear Regression


# Data Preprocessing 

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset 
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values 


#Missing Data
'''from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")

imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])'''


#Ecncoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

#Avoiding the dummy variable trap
X = X[:, 1:]


#Splitting dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


#Fitting the multiple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#Predicting the test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))


#Building the optimal model using backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) 
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()