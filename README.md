# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import dataset and get data info
2.check for null values
3.Map values for position column
4.Split the dataset into train and test set
5.Import decision tree regressor and fit it for data
6.Calculate MSE,R2 and y predict.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: AFSAR JUMAIL S
RegisterNumber:  212222240004
*/
import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
l0=LabelEncoder()

data["Position"]=l0.fit_transform(data['Position'])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
data.head()

![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343395/b030d9e9-a396-4d54-85cc-efa718148e24)


data.info()

![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343395/7b07824d-0c69-4db6-ad4b-5394eaf91b43)


isnull() and sum()

![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343395/3a650ed8-dc64-4062-8e60-1095154a6e8d)


data.head() for salary

![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343395/99230934-2671-4209-ab15-3a0943f44441)


MSE Value

![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343395/379af3a9-b2a1-4462-b90e-cc04e8eab518)


r2 value

![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343395/5b933b35-0116-4896-bc6b-7e2b65db4092)


data prediction

![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343395/aba6ea26-4de5-4efc-a896-a20453aadb47)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
