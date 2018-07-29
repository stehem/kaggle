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

#### xgb
params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.006,
        'max_depth': 20,
        'min_child_weight': 14,
        'subsample': 0.55,
        'colsample_bytree': 0.45,
        'n_jobs': -1,
        'random_state': 456,
	'silent': True
}

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
lg_preds_test = np.expm1(model_lgb.predict(x_test))
xg_preds_test = np.expm1(model_xgb.predict(xgb.DMatrix(x_test), ntree_limit=model_xgb.best_ntree_limit))

features = np.array([lg_preds_test, xg_preds_test]).T
y_test = np.expm1(y_test)

#train a meta model that get predictions from lgb xgb predictions
clf = RandomForestRegressor(random_state=0)
clf.fit(features, y_test)

#visualizer = ResidualsPlot(clf, hist=False)
#visualizer.fit(features, y_test)
#visualizer.poof()

#visualizer.feature_importances_


#now predict based on lgb/xgb predictions on submit data


lg_preds = np.expm1(model_lgb.predict(x_submit))
xg_preds = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))

m_preds = pd.DataFrame(clf.predict(np.array([lg_preds, xg_preds]).T))
m_preds.insert(0, "ID", ids.values)
m_preds.columns = ["ID","target"]
m_preds.to_csv("submit.csv", index = False)



#####
# train lg on 40% keep 10
# train xg on 40% keep 10

x_lg = x.iloc[0:2229,]
x_xg = x.iloc[2230:,]
y_lg = y.iloc[0:2229,]
y_xg = y.iloc[2230:,]

x_train_lg, x_test_lg, y_train_lg, y_test_lg = train_test_split(x_lg, y_lg, test_size = 0.10, random_state = 0)
x_train_xg, x_test_xg, y_train_xg, y_test_xg = train_test_split(x_xg, y_xg, test_size = 0.10, random_state = 0)














space  = [
    Integer(40, 100, name='num_leaves'),
    Integer(30, 60, name='max_depth'),
]
#[0.00542047893814942, 29, 24, 0.39949465609514856, 1, 0.67943500, 10]
#[40, 30 , 0.4, 10]
#[76, 55]

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













#[0.005883199373371072, 20, 14, 0.5519512436469104, 0.4515230593477767]


space  = [
    Real(0.005, 0.05, name='learning_rate'),
    Integer(5, 20, name='max_depth'),
    Integer(10, 30, name='min_child_weight'),
    Real(0.5, 0.9, name='subsample'),
    Real(0.4, 0.9, name='colsample_bytree'),
]
########
#bayesian optimization
def find_hyper_params_xgb(values):
    params = {
        'learning_rate': values[0]
        'max_depth': values[1],
        'min_child_weight': values[2],
        'subsample': values[3],
        'colsample_bytree': values[4],
        'objective': 'reg:linear',
        'n_jobs': -1,
        'random_state': 456,
	'silent': True
    }
    tr_data = xgb.DMatrix(x_train, y_train)
    va_data = xgb.DMatrix(x_test, y_test)
    watchlist = [(tr_data, 'train'), (va_data, 'valid')]
    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 100, verbose_eval=100)
    return model_xgb.best_score

res_gp = gp_minimize(find_hyper_params_xgb, space, n_calls=20, random_state=0,n_random_starts=10)
res_gp.fun
res_gp.x
