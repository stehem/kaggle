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

train = pd.read_csv('train.csv')

submit = pd.read_csv('test.csv')
x_submit = submit.iloc[:,1:]
ids = submit.iloc[:,0]


x = train.iloc[:,2:]
y = np.log1p(train.iloc[:,1].values)


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
        "learning_rate" : 0.0020,
        "num_leaves": 25,
        "max_depth" : 20,
        "bagging_fraction": 0.2,
        "bagging_freq": 1,
        "feature_fraction": 0.8,
        "lambda_l1": 7,
}


evals_result = {}
model = lgbm.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)



lg_preds = pd.DataFrame(np.expm1(model.predict(x_submit)))
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


#### xgb
params = {'objective': 'reg:linear', 
          'eval_metric': 'rmse',
          'eta': 0.001,
          'max_depth': 10, 
          'nthread': 8, 
          'subsample': 0.6, 
          'colsample_bytree': 0.6,
          'alpha':0.001,
          'random_state': 42, 
          'silent': True}
    
tr_data = xgb.DMatrix(x_train, y_train)
va_data = xgb.DMatrix(x_test, y_test)
    
watchlist = [(tr_data, 'train'), (va_data, 'valid')]
    
model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 100, verbose_eval=100)
    

dtest = xgb.DMatrix(x_submit)
xg_preds = pd.DataFrame(np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit)))
xg_preds.insert(0, "ID", ids.values)
xg_preds.columns = ["ID","target"]
xg_preds.to_csv("submit.csv", index = False)
    


#### avg preds 1.57 :(
avg_preds = pd.DataFrame(np.mean(np.array([xg_preds.values.flatten(),lg_preds.values.flatten()]), axis =0))
avg_preds.insert(0, "ID", ids.values)
avg_preds.columns = ["ID","target"]
avg_preds.to_csv("submit.csv", index = False)




####
#get predictions for the val sets
lg_preds_test = np.expm1(model.predict(x_test))
xg_preds_test = np.expm1(model_xgb.predict(xgb.DMatrix(x_test), ntree_limit=model_xgb.best_ntree_limit))

features = np.array([lg_preds_test, xg_preds_test]).T
lr = linear_model.LinearRegression()
lr.fit(features, y_test)
lr = linear_model.LinearRegression()
lr.fit(features, y_test)
m_preds = pd.DataFrame(lr.predict(np.array([lg_preds.values.flatten(), xg_preds.values.flatten()]).T))
m_preds.insert(0, "ID", ids.values)
m_preds.columns = ["ID","target"]
m_preds.to_csv("submit.csv", index = False)



















space  = [
    Real(0.005, 0.01,  name='learning_rate'),
    Integer(20, 30, name='num_leaves'),
    Integer(15, 25, name='max_depth'),
    Real(0.1, 0.4, name='bagging_fraction'),
    Integer(1, 5, name='bagging_freq'),
    Real(0.5, 1, name='feature_fraction'),
    Integer(5, 10, name='lambda_l1')
]
#[0.00542047893814942, 29, 24, 0.39949465609514856, 1, 0.67943500

########
#bayesian optimization
def find_hyper_params(values):
    params = {
        "num_threads": 8,
        "verbosity": -1,
        "zero_as_missing": "true",
        "boosting":'gbdt',
        "objective" : "regression",
        "metric" : "rmse",
        "seed": 42,
        "learning_rate" : values[0],
        "num_leaves": values[1],
        "max_depth" : values[2],
        "bagging_fraction": values[3],
        "bagging_freq": values[4],
        "feature_fraction": values[5],
        "lambda_l1": values[6]
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

res_gp = gp_minimize(find_hyper_params, space, n_calls=20, random_state=0,n_random_starts=10)
