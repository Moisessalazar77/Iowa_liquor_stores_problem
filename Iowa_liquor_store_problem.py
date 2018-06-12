# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:15:15 2018

@author: moisessalazar77
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn import linear_model
import sklearn.metrics as metrics

dataset=pd.read_csv('Iowa_Liquor_sales_sample_10pct.csv')



dataset_with_target=dataset.copy()
dataset_with_target.dropna(inplace=True)


dataset_with_target['Zip Code'] = dataset_with_target['Zip Code'].replace('712-2',7122)  
dataset_with_target['Zip Code'] = dataset_with_target['Zip Code'].astype('int32')
dataset_with_target['Sale (Dollars)'] = dataset_with_target['Sale (Dollars)'].replace({'\$':''}, regex=True)
dataset_with_target['Sale (Dollars)'] = dataset_with_target['Sale (Dollars)'].astype('float32')
dataset_with_target['State Bottle Cost'] = dataset_with_target['State Bottle Cost'].replace({'\$':''}, regex=True)
dataset_with_target['State Bottle Cost'] = dataset_with_target['State Bottle Cost'].astype('float32')
dataset_with_target['State Bottle Retail'] = dataset_with_target['State Bottle Retail'].replace({'\$':''}, regex=True)
dataset_with_target['State Bottle Retail'] = dataset_with_target['State Bottle Retail'].astype('float32')
dataset_with_target['City']=dataset_with_target['City'].str.lower()
dataset_with_target['Category Name']=dataset_with_target[ 'Category Name'].str.lower()
dataset_with_target['County']=dataset_with_target['County'].str.lower()
dataset_with_target['Item Description']=dataset_with_target['Item Description'].str.lower()

dataset_with_target['Profit in %']=(dataset_with_target['State Bottle Retail']-dataset_with_target['State Bottle Cost'])*(dataset_with_target['State Bottle Retail']**(-1))*100
dataset_with_target['Volume Sold (Mililiters)']=dataset_with_target['Volume Sold (Liters)']*1000
c=dataset_with_target['Volume Sold (Mililiters)'].min()
d=dataset_with_target['Volume Sold (Mililiters)'].max()
dataset_with_target['Volume Sold (Mililiters)'] = dataset_with_target['Volume Sold (Mililiters)'].apply(lambda x: (x-c)/(d-c))
a=dataset_with_target['Sale (Dollars)'].min()
b=dataset_with_target['Sale (Dollars)'].max()
dataset_with_target['Sale (Dollars)'] = dataset_with_target['Sale (Dollars)'].apply(lambda x: (x-a)/(b-a))

print(dataset_with_target['Sale (Dollars)'].describe())
print(dataset_with_target['State Bottle Cost'].describe())
print(dataset_with_target['State Bottle Retail'].describe())
print(dataset_with_target['Profit in %'].describe())


y0=dataset_with_target.iloc[:,15].values

X1=dataset_with_target.iloc[:,[1,5,7,8,10,13,11,14,]].values


labelencoder_X=LabelEncoder()
X1[:,1]=labelencoder_X.fit_transform(X1[:,1])
X1[:,2]=labelencoder_X.fit_transform(X1[:,2])
X1[:,4]=labelencoder_X.fit_transform(X1[:,4])



X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y0, test_size = 0.2, random_state = 0)
regressor1=LinearRegression()
regressor1.fit(X_train1,y_train1)


#Predicting the test set results
y_pred1=regressor1.predict(X_test1)

X_opt3=np.append(arr=np.ones((269258,1)).astype(int),values=X1,axis=1).astype(float)
regressor_OLS2=sm.OLS(y0,X_opt3).fit()
regressor_OLS2.summary()

reg1 = reg = linear_model.Ridge(alpha=0.00000005, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=True, random_state=0, solver='auto', tol=0.000000000001)
        
reg1.fit(X_train1,y_train1)
y_pred2=reg1.predict(X_test1)

metrics.r2_score(y_pred = y_pred2,
                 y_true = y_test1)
