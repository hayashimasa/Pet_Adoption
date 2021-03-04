from collections import OrderedDict
from pprint import pprint

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

from petdata import PetDataset
from metric import quadratic_weighted_kappa


data_train = PetDataset(data_type='train', shuffle=False, small=True)
data_test = PetDataset(data_type='test', indicies=data_train.indicies, oe=data_train.oe)

data_tab_train = pd.concat([data_train.X['cat'], data_train.X['cont']], axis=1)
data_tab_test = pd.concat([data_test.X['cat'], data_test.X['cont']], axis=1)

train_data = lgb.Dataset(data_tab_train, data_train.y)
test_data = lgb.Dataset(data_tab_test, data_test.y, reference=train_data)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting training...')
# train
gbm = lgb.train(
    params,
    train_data,
    num_boost_round=20,
    valid_sets=test_data,
    early_stopping_rounds=5
)

print('Saving model...')
# save model to file
gbm.save_model('models/lgb_model.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(data_tab_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(data_test.y, y_pred) ** 0.5)
print('The mse of prediction is:', mean_squared_error(data_test.y, y_pred))
print('The quadratic weighted kappa of prediction is:', quadratic_weighted_kappa(y_pred, data_test.y))

