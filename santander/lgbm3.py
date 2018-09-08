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


x, x_submit = remove_zero_columns(x, x_submit)

#x = preprocessing.scale(x)


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


sample_preds = []
SAMPLE_SIZE=1500
for _ in range(10):
    x_sample = x.sample(SAMPLE_SIZE)
    x_sample_indices = x_sample.index
    y_sample = np.take(y, indices)
    x_train_sample, x_test_sample, y_train_sample, y_test_sample = train_test_split(x_sample, y_sample, test_size = 0.2, random_state = 0)
    #
    lgtrain = lgbm.Dataset(x_train_sample, label=y_train_sample)
    lgval = lgbm.Dataset(x_test_sample, label=y_test_sample)
    evals_result = {}
    sample_model = lgbm.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)
    #
    sample_preds.append(np.expm1(sample_model.predict(x_submit)))



lg_preds = pd.DataFrame(np.array(sample_preds).mean(axis=0))
lg_preds.insert(0, "ID", ids.values)
lg_preds.columns = ["ID","target"]
lg_preds.to_csv("submit.csv", index = False)



#seems to overfit, 1.76 submit
def filter_columns(threshold, x):
    counts = {}
    for i in range(len(x.columns)):
        col = x.iloc[:,i]
        count = col[col != 0].count()
        col_name = x.columns[i]
        counts[col_name] = count
    sorted_by_value = sorted(counts.items(), key= lambda kv: kv[1])
    above_threshold = [kv[0] for kv in sorted_by_value if kv[1] > threshold]
    oo
    filtered = x.filter(items=above_threshold)
    return filtered


#1.36 valid 1.7 submit :(
def filter_rows(threshold, x):
    row_indices = []
    counts = []
    for i in range(len(x)):
        row = x.iloc[i,:]
        count = len([x for x in row if x != 0])
        counts.append(count)
        if count > threshold:
            row_indices.append(i)
    return row_indices, counts


