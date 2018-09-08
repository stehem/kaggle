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


def remove_zero_columns(x, submit):
    cols = []
    for col in x.columns:
        if col != 'ID' and col != 'target':
            if x[col].std() == 0: 
                cols.append(col)
    x.drop(cols, axis=1, inplace=True)
    submit.drop(cols, axis=1, inplace=True)
    return x, submit



train = pd.read_csv('train.csv')
submit = pd.read_csv('test.csv')


x = train.iloc[:,2:]
x_submit = submit.iloc[:,1:]
y = np.log1p(train.iloc[:,1].values)
ids = submit.iloc[:,0]


READ_FROM_FILE=True
if READ_FROM_FILE:
    consolidated = pd.read_pickle("./consolidated.pkl")
    consolidated.drop("ID", axis=1, inplace=True)
else:
    grouped = train.groupby('target')
    consolidated = pd.DataFrame(columns=train.columns[1:,])
    print(len(grouped))
    i = 0
    for name,group in grouped:
        if i%50 == 0:
            print("XXXX")
            print(i)
            print("XXXX")
        consolidated = consolidated.append(group.mean(), ignore_index=True)
        i=i+1
    consolidated.drop("ID", axis=1, inplace=True)
    pd.to_pickle("./consolidated.pkl")


CHANGE_RATE=0.2
def noisify(value):
    mu = value
    sig = mu*CHANGE_RATE
    return np.random.normal(mu, sig)


def noisify_consolidated():
    consolidated_noisy = consolidated.copy()
    consolidated_noisy.drop("target", axis=1, inplace=True)
    cols = consolidated.columns
    for index, row in consolidated.iterrows():
        non_nul_cols = [col for col in cols if row[col] != 0]
        #get 30% to modify
        modify_rate = 0.9
        modify_qty = ceil(len(non_nul_cols)*modify_rate)
        cols2modify = sample(non_nul_cols, modify_qty)
        #add noise 
        for col in cols2modify:
            consolidated_noisy.loc[index, col] = noisify(row[col])
        #
        zero_rate = 0.5
        zero_qty = ceil(len(non_nul_cols)*zero_rate)
        cols2zero = sample(non_nul_cols, zero_qty)
        for col in cols2zero:
            consolidated_noisy.loc[index, col] = 0.0
    return consolidated_noisy


NOISE_ROUNDS = 1
appended_x = x.copy()
appended_y = y.copy()
for r in range(NOISE_ROUNDS):
    noisy = noisify_consolidated()
    appended_x = appended_x.append(noisy)
    appended_y = np.concatenate((appended_y, np.log1p(noisify(consolidated.target))))

#appended_x.to_pickle("./appended_x.pkl")
#pd.DataFrame(appended_y).to_pickle("./appended_y.pkl")

x_train, x_test, y_train, y_test = train_test_split(appended_x, appended_y, test_size = 0.2, random_state = 0)


lgtrain = lgbm.Dataset(x_train, label=y_train)
lgval = lgbm.Dataset(x_test, label=y_test)

#[0.00542047893814942, 29, 24, 0.39949465609514856, 1, 0.67943500, 10]
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


lg_preds = pd.DataFrame(np.expm1(model_lgb.predict(x_submit)))
lg_preds.insert(0, "ID", ids.values)
lg_preds.columns = ["ID","target"]

lg_preds.to_csv("submit.csv", index = False)



