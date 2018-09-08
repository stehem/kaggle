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


train = pd.read_csv('train.csv')
submit = pd.read_csv('test.csv')


x = train.iloc[:,2:]
x_submit = submit.iloc[:,1:]
y = np.log1p(train.iloc[:,1].values)
ids = submit.iloc[:,0]



#appended_x.to_pickle("./appended_x.pkl")
#pd.DataFrame(appended_y).to_pickle("./appended_y.pkl")


params = {
        "num_threads": 8,
        "verbosity": -1,
        "zero_as_missing": "true",
        "boosting":'gbdt',
        "objective" : "regression",
        "metric" : "rmse",
        "seed": 42,
        "learning_rate" : 0.005,
}


model_preds = []
columns = x.columns
number_of_models = 10
number_of_columns = 500
for i in range(number_of_models):
    model_columns = sample([col for col in x.columns], number_of_columns)
    x_ = x[model_columns]
    #
    x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size = 0.2, random_state = 0)
    lgtrain = lgbm.Dataset(x_train, label=y_train)
    lgval = lgbm.Dataset(x_test, label=y_test)
    #
    evals_result = {}
    model_lgb = lgbm.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)
    #
    model_preds.append(np.expm1(model_lgb.predict(x_submit)))






#[0.00542047893814942, 29, 24, 0.39949465609514856, 1, 0.67943500, 10]


lg_preds = pd.DataFrame(np.array(model_preds).mean(axis=0))
lg_preds.insert(0, "ID", ids.values)
lg_preds.columns = ["ID","target"]

lg_preds.to_csv("submit.csv", index = False)



