# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 19:32:00 2018

@author: Pranjal
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


dfs = pd.read_excel('ACO Problem Statement.xlsx', sheet_name='Data')


dfs['adv_pay_amt'] = dfs['adv_pay_amt'].fillna(0)

dfs = dfs.drop(['aco_num', 'aco_name', 'adv_pay'], axis=1)

dfs = dfs[np.isfinite(dfs['per_capita_exp_total_py'])]

# Removing columns which has more then 60% of NaN value
Remove_NaN = True
Removed_feature = []
if Remove_NaN:
    Total_feature = dfs.axes[1]
    print("Removing Null Values...")
    TN = dfs.shape[0]
    for feature in Total_feature:
        val = pd.isnull(dfs[feature]).sum()
        avg = (val / TN) * 100
        if avg > 60:
            Removed_feature.append(feature)
            dfs = dfs.drop(feature, axis=1)
    print("Shape of new dfs ", dfs.shape)
    print(len(Removed_feature),"Feature removed are: ", str(Removed_feature))
    
# Encoding categorical data
labelencoder = LabelEncoder()
dfs['aco_state'] = labelencoder.fit_transform(dfs['aco_state'])

y = dfs['per_capita_exp_total_py']
X = dfs.drop(['per_capita_exp_total_py'], axis=1)

# Imputer
from sklearn.preprocessing import Imputer
imputer1 = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer1.fit(X)
X = imputer.transform(X)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.1)

params_fixed = {
    'objective': 'reg:linear',
    'silent': 0,
    'n_estimators' :1000,
     'max_depth': 6,
     'learning_rate': 0.01
}

xgb = XGBRegressor(**params_fixed)
xgb.fit(Xtrain, ytrain)

predicted = xgb.predict(Xtest)

from sklearn import metrics

print("Training Accuracy :" + str((metrics.r2_score(ytrain, xgb.predict(Xtrain)))*100))
print("Test Accuracy :" + str((metrics.r2_score(ytest, xgb.predict(Xtest)))*100))









