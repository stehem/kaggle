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

x = preprocessing.scale(x)
pca = PCA(n_components=1500)
pca.fit_transform(x)

x_submit = preprocessing.scale(x_submit)
pca.fit_transform(x_submit)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)


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

np.dot(x_submit.iloc[0:1,].values, x.iloc[0:1,].values.T)
np.dot(x_submit.iloc[0:1,].values, x.values.T)
    
#
x = consolidated.iloc[:,1:]
#x.drop("target", axis=1, inplace=True)
y = np.log1p(consolidated.iloc[:,0].values)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)

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


########
#bayesian optimization
def find_hyper_params_lgb(values):
    params = {
        "num_threads": 8,
        "verbosity": -1,
        "zero_as_missing": "true",
        "boosting":'gbdt',
        "objective" : "regression",
        "metric" : "rmse",
        "seed": 42,
        "learning_rate" : 0.0054,
        "num_leaves": 76,
        "max_depth" : 55,
        "bagging_fraction": 0.4,
        "bagging_freq": 1,
        "feature_fraction": 0.68,
        "lambda_l1": 10
	}
    lgtrain = lgbm.Dataset(x_train, label=y_train)
    lgval = lgbm.Dataset(x_test, label=y_test)
    evals_result = {}
    model = lgbm.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)
    return model.best_score['valid_0']['rmse']

res_gp = gp_minimize(find_hyper_params_lgb, space, n_calls=20, random_state=0,n_random_starts=10)
res_gp.fun
res_gp.x




