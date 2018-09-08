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
from sklearn import preprocessing
from sklearn.decomposition import PCA
from random import sample
from math import ceil
from sklearn.model_selection import KFold





train = pd.read_csv('train.csv')
submit = pd.read_csv('test.csv')


x = train.iloc[:,2:]
x_submit = submit.iloc[:,1:]
y = np.log1p(train.iloc[:,1].values)
ids = submit.iloc[:,0]


def compress_columns(df, number_of_chunks=100):
    agg = pd.DataFrame()
    chunk=np.floor(df.shape[1] / number_of_chunks)
    indices = np.array_split(list(range(df.shape[1])), number_of_chunks)
    for c in range(number_of_chunks):
        agg[c] = np.sum(df.iloc[:,indices[c]], axis=1)
    return agg

x = compress_columns(x)
x_submit = compress_columns(x_submit)

#appended_x.to_pickle("./appended_x.pkl")
#pd.DataFrame(appended_y).to_pickle("./appended_y.pkl")


model_preds = []
kf = KFold(n_splits=4)
for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x.loc[train_index], x.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lgtrain = lgbm.Dataset(x_train, label=y_train)
    lgval = lgbm.Dataset(x_test, label=y_test)
    params = {
        "num_threads": 8,
        "verbosity": -1,
        "zero_as_missing": "true",
        "boosting":'gbdt',
        "objective" : "regression",
        "metric" : "rmse",
        "seed": 42,
        "learning_rate" : 0.005,
        "num_leaves": 29,
        "max_depth" : 24,
        "bagging_fraction": 0.4,
        "bagging_freq": 1,
        "feature_fraction": 0.68,
        "lambda_l1": 10,
    }
    evals_result = {}
    model_lgb = lgbm.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)
    model_preds.append(np.expm1(model_lgb.predict(x_submit)))





lg_preds = pd.DataFrame(np.array(model_preds).mean(axis=0))
lg_preds.insert(0, "ID", ids.values)
lg_preds.columns = ["ID","target"]

lg_preds.to_csv("submit.csv", index = False)



