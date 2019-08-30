# Importing the libraries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 

# Importing the dataset
dataset = pd.read_csv('student_scores1.csv') 
#moving column 1 to x
X = dataset.iloc[:,:-1].values  
#moving column 2 to y
y = dataset.iloc[:, 1].values  


#dataset.shape 
 
#dataset.keys()

# Taking care of missing data
from sklearn.preprocessing import Imputer
#replacing missing with mean
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:1])
#transfering imputed values back to X
X[:,0:1] = imputer.transform(X[:,0:1])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

from sklearn.linear_model import LinearRegression  
#creating regressor object of linear regression
regressor = LinearRegression()  
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)


from sklearn import metrics  
#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

#plt.scatter(X_train, y_train, color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#plt.title('Hours vs Percentage')  
#plt.xlabel('Hours Studied')  
#plt.ylabel('Percentage Score')  
#plt.show()


plt.scatter(X_test, y_test, color = 'red')
#plt.plot(X_test, y_test, color = 'blue')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()