 
In This Document I am going to explain how I passive towards the solution innovaccer problem statement.  
you can read the problem statement and the data-set from ACO Problem Statement.xlsx file 
## Explanation

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,  Imputer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
```

imported all the required libraries, which are:  
> pandas  
> numpy  
> sklearn  
> xgboost  

```python
dfs = pd.read_excel('ACO Problem Statement.xlsx', sheet_name='Data')
dfs['adv_pay_amt'] = dfs['adv_pay_amt'].fillna(0)
dfs = dfs.drop(['aco_num', 'aco_name', 'adv_pay'], axis=1)
dfs = dfs[np.isfinite(dfs['per_capita_exp_total_py'])]
```
Reading .xlsx file to pandas dataframs  
Filling missing values of 'adv_pay_amt' column  
droping redundant features  
removing rows with NaN values in 'per_capita_exp_total_py' column  
```python
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
```
Removing Features which have more then 60% of missing values.

```python
# Encoding categorical data
labelencoder = LabelEncoder()
dfs['aco_state'] = labelencoder.fit_transform(dfs['aco_state'])
```
Encoding categorical data in 'aco_state' column

```python
# defining features and target
y = dfs['per_capita_exp_total_py']
X = dfs.drop(['per_capita_exp_total_py'], axis=1)
```
defining features and target

```python
# Imputer
imputer1 = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer1.fit(X)
X = imputer.transform(X)
```
Imputing missing values from features
```python
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.1)
```
Spilling the data-set in training and test with 1/10  test size

```python
params_fixed = {
    'objective': 'reg:linear',
    'silent': 0,
    'n_estimators' :1000,
     'max_depth': 6,
     'learning_rate': 0.01
}

xgb = XGBRegressor(**params_fixed)
```

Defining XGBoost Regresses model and its parameters.

```python
xgb.fit(Xtrain, ytrain)
predicted = xgb.predict(Xtest)
```
Training the model on train set and predicting on test set

```python
print("Training Accuracy (R^2 score):" + str((metrics.r2_score(ytrain, xgb.predict(Xtrain)))*100))
print("Test Accuracy (R^2 score):" + str((metrics.r2_score(ytest, xgb.predict(Xtest)))*100))
```
Printing the Train and Test Accuracy (R^2 score)

## Model results 
### Training Accuracy (R^2 score): 99.92146242417436
### Test Accuracy (R^2 score): 96.89975735051324
