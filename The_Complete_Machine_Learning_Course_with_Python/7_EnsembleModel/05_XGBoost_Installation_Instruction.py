import xgboost as xgb
import numpy as np

"""
# XGBoost

Installation Guide
# MacOS
Just follow the following link

https://anaconda.org/conda-forge/xgboost
"""


data = np.random.rand(100, 10) # 100 entities, each contains 10 features
label = np.random.randint(2, size=100) # binary target
dtrain = xgb.DMatrix(data, label=label)

dtest = dtrain

param = {'bst:max_depth': 2, 'bst:eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)

