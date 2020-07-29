#Decision Tree Regression

#Regression Template

# Data Preprocessing 

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset 
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values 

#Missing Data
'''from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")

imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])'''

#Ecncoding categorical data
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)'''

#Splitting dataset into test set and training set
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''


#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


#Fitting Regression model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


#Predicting Regression result 
y_pred = regressor.predict([[6.5]])


#Visualising Regression Model
plt.scatter(X, y, color="red")
plt.plot(X, regressor.predict(X), color="blue")
plt.title("Truth or Bluff (Decision Tree Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Visualising Regression Model for higher resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Truth or Bluff (Decision Tree Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
