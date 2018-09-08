import numpy as np
import pandas as pd
import lightgbm as lgbm
import csv
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import xgboost as xgb
from skopt import gp_minimize
from sklearn import linear_model
from yellowbrick.regressor import ResidualsPlot
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv('train.csv')

submit = pd.read_csv('test.csv')
x_submit = submit.iloc[:,1:]
ids = submit.iloc[:,0]


x = train.iloc[:,2:]
y = train.iloc[:,1].values


def remove_zero_columns(x, submit):
    cols = []
    for col in x.columns:
        if col != 'ID' and col != 'target':
            if x[col].std() == 0: 
                cols.append(col)
    x.drop(cols, axis=1, inplace=True)
    submit.drop(cols, axis=1, inplace=True)
    return x, submit


x, x_submit = remove_zero_columns(x, x_submit)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


#svr = SVR(C=1.0, epsilon=0.2)
#svr.fit(x, y) 
#svr_preds = svr.predict(x_submit)
#print(np.sqrt(mean_squared_error(y_test, preds)))


rfr = RandomForestRegressor()
rfr.fit(x, y) 
rfr_preds = rfr.predict(x_submit)


abr = AdaBoostRegressor()
abr.fit(x, y)
abr_preds = abr.predict(x_submit)

br = BaggingRegressor()
br.fit(x, y)
br_preds = br.predict(x_submit)

etr = ExtraTreesRegressor()
etr.fit(x, y)
etr_preds = etr.predict(x_submit)

gbr = GradientBoostingRegressor()
gbr.fit(x, y)
gbr_preds = gbr.predict(x_submit)


avg_preds = pd.DataFrame(np.mean(np.array([rfr_preds, abr_preds, br_preds, etr_preds, gbr_preds]),axis=0))
avg_preds.insert(0, "ID", ids.values)
avg_preds.columns = ["ID","target"]
avg_preds.to_csv("submit.csv", index = False)



