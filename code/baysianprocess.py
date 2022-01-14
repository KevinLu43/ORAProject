import csv
from timeit import default_timer as timer
import lightgbm as lgb
from hyperopt import STATUS_OK
import numpy as np
from hyperopt import hp
# 建立非連續行的uniform distribution
num_leaves = {'num_leaves': hp.uniform('num_leaves', 30, 150, 1)}

# 建立評估次數
MAX_EVALS = 200
# 建立kfold 意思是要將資料分成幾個group
N_FOLDS = 10


def objective(params, n_folds=N_FOLDS):
    global ITERATION
    ITERATION += 1
    subsample = params['boosting_type'].get('subsample', 1.0)
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])
    # 準備計算跑model的時間
    start = timer()
    # 交叉驗證 early_stopping_rounds是指超過100次都沒有改善的話就會停止模型
    cv_results = lgb.cv(params, train_set,
                        num_boost_round=10000,
                        nfold=n_folds,
                        early_stopping_rounds=100,
                        metrics='auc',
                        seed=50)
    # 總運行時間
    run_time = timer() - start
    best_score = np.max(cv_results['auc-mean'])
    # 因為auc score是越大越好,所以我們必須取出最小的loss
    loss = 1 - best_score
    # estimators(樹的數量)
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)
    # 將所獲的資料寫入csv裡
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators,
            'train_time': run_time, 'status': STATUS_OK}