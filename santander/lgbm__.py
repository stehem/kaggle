import numpy as np
import pandas as pd
import lightgbm as lgbm
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


train = pd.read_csv('train.csv')
#test = pd.read_csv('test.csv')

x = train.iloc[:,2:]
y = np.log1p(train.iloc[:,1].values)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


lgtrain = lgbm.Dataset(x_train, label=y_train)
lgval = lgbm.Dataset(x_test, label=y_test)

params = {
        "num_threads": 4,
        "verbosity": -1,
        "zero_as_missing": "true",
        "boosting_type":'gbdt',
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



submit = pd.read_csv('test.csv')
x_submit = submit.iloc[:,1:]
ids = submit.iloc[:,0]

preds = pd.DataFrame(np.expm1(model.predict(x_submit)))

preds.insert(0, "ID", ids.values)
preds.columns = ["ID","target"]

preds.to_csv("submit.csv", index = False)
