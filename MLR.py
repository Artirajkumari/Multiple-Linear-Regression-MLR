# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:09:56 2025

@author: ArtiR
"""

""" Regression == 1.Linear== Simple linear regression 2> Multi linear regression ,  
Gradient descent/sstocastic gradient descend/batch gra descend(No parcticle requoired)


2. Non linear ==>
polynomial regression
support vector regression 
decissions tree regression 
random forest regressions 
lesso regularizations/ lesso regressor/L1 regularizations 
 """

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


## import the data 
df= pd.read_csv(r"D:\All data\50_Startups.csv")
x= df.iloc[: , :-1]
y= df["Profit"]

x= pd.get_dummies(x, dtype=int)
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(x,y, test_size=0.2,random_state= 0)


from sklearn.linear_model import LinearRegression
slr= LinearRegression()
slr.fit(x_train,y_train)
y_pred= slr.predict(x_test)


slr.score(x_train,y_train)
slr.score(x_test,y_test)


""" Stats model formula  api== it is used to call all the formulas  == """

# import the models
import statsmodels.formula.api as sm
## append the constant columns // y= mx1+mx2+...+C
x= np.append(arr = np.ones((50,1)).astype(int), values= x, axis = 1)

""" once we build mlr model we will generate regression ols table hv to check
r2> adjusted r2
 does these values are range between 0-1
 significate have as 0.05
  p value is >0.05 ..> reject the null hypothesis(stats)// elmiate the feature /atributes
   keep doing the steps till check the conditions p value <= 0.05
   atribute is left that would be answer (business need to invest money of those attributes"""
import statsmodels.api as sm

x_opt=x[:,[0,1,2,3,4,5]]

slr_OLS = sm.OLS(endog=y,exog=x_opt).fit()
slr_OLS.summary()
## 0.990>0.05 so we hv eleminate this feature
import statsmodels.api as sm

x_opt=x[:,[0,1,2,3,5]]

slr_OLS1 = sm.OLS(endog=y,exog=x_opt).fit()
slr_OLS1.summary()
## 0.943>0.05 so we hv eleminate this feature

import statsmodels.api as sm

x_opt=x[:,[0,1,2,3]]

slr_OLS1 = sm.OLS(endog=y,exog=x_opt).fit()
slr_OLS1.summary()

## 0.608>0.05 so we hv eleminate this feature

import statsmodels.api as sm

x_opt=x[:,[0,1,2]]

slr_OLS1 = sm.OLS(endog=y,exog=x_opt).fit()
slr_OLS1.summary()
## 0.123>0.05 so we hv eleminate this feature

import statsmodels.api as sm

x_opt=x[:,[0,1]]

slr_OLS1 = sm.OLS(endog=y,exog=x_opt).fit()
slr_OLS1.summary()

## So R&D department is best for investing and get the profit 







### if p value > s.i(significance value==0.05) ..> reject the null hypothesis(eliminate the feature)


## Regression , tree algorithm (decission tree & random forest) does not required feature scaling.