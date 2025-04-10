# import warnings
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import dirichlet
from scipy.stats import spearmanr

gbm_hyperparams = {
  'task': 'train',
  'boosting_type': 'gbdt',
  'objective': 'regression',
  'metric': ['l1','l2'],
  "num_iterations": 1000, 
  'seed': 42,
  'learning_rate': 1e-2,
  "verbosity": -1,
}

def subsampling(X, p=0.1):
  w = np.zeros(X.shape[0])
  for idx in range(X.shape[0]):
    w[idx] = dirichlet.pdf(X[idx] / X[idx].sum())
  w_rank = np.argsort(-w)

def train_gbm(X_train, y_train, X_val, y_val, hyperparams=gbm_hyperparams):
  gbm = lgb.LGBMRegressor(**hyperparams)
  reg = gbm.fit(X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='l2', callbacks=[
    lgb.early_stopping(stopping_rounds=3, verbose=False),
  ])
  return reg

if __name__ == "__main__":
  import warnings
  warnings.filterwarnings("ignore", category=UserWarning)

  train_mixture_1m = pd.read_csv('train_mixture_1m.csv', index_col='index')
  train_pile_loss_1m = pd.read_csv('train_pile_loss_1m.csv', index_col='index')
  test_mixture_1B = pd.read_csv('test_mixture_1B.csv', index_col='index')
  test_pile_loss_1B = pd.read_csv('test_pile_loss_1B.csv', index_col='index')
  metrics = train_pile_loss_1m.columns.tolist()

  X_train = train_mixture_1m.values
  X_val = test_mixture_1B.values

  rs = []
  for metric in metrics:
    y_train = train_pile_loss_1m[metric].values
    y_val = test_pile_loss_1B[metric].values
    reg = train_gbm(X_train, y_train, X_val, y_val)
    r, _ = spearmanr(reg.predict(X_val), y_val)
    rs.append(r)
    print(metric, "rank corr: {}".format(np.round(r, 4)))
