#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Logistic regression on the titanic dataset
titanic = pd.read_csv('titanic_train.csv')




# Well, how many records are there in the data frame anyway?


#Taking care of missing values¶



#Dropping irrelevant features
titanic_data = titanic.drop(['PassengerId','Name','Ticket','Cabin','Fare'], axis=1)


#Taking care of missing values
def age_approx(cols):
    Age = cols[0]          #local vars
    Pclass = cols[1]
    
    if pd.isnull(Age):     #if age not null pass some random num to each of pclass
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
#parameters passed to age_approx is Age and Pclass
titanic_data['Age'] = titanic_data[['Age', 'Pclass']].apply(age_approx, axis=1)

#Converting categorical variables to a dummy indicators
#one-hot encoding we convert male-female to 1 or zero
#droping the first colm result in encoded val in single colm
gender = pd.get_dummies(titanic_data['Sex'],drop_first=True)

#one-hot encoding we convert C,Q,S to binary
#droping the first colm result in 2 encoded val colms
embark_location = pd.get_dummies(titanic_data['Embarked'],drop_first=True)

#drop sex and embark since we made its better part
titanic_data.drop(['Sex', 'Embarked'],axis=1,inplace=True)

#concat the newly formed colms
titanic_dmy = pd.concat([titanic_data,gender,embark_location],axis=1)


#Pclass is not , We can drop
titanic_dmy.drop(['Pclass'],axis=1,inplace=True)


#Checking that your dataset size is sufficient
titanic_dmy.info()

X = titanic_dmy.iloc[:,1:7].values
y = titanic_dmy.iloc[:,0].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

# Predicting the Test set results
y_pred = LogReg.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

#from sklearn.metrics import classification_report
#print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score
print("Accuracy = ",end=" ")
print(accuracy_score(y_test,y_pred)*100)

