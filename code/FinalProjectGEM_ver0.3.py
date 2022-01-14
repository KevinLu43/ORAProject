# utils
import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import RandomizedSearchCV

# model
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


def MAPE(y_true, y_pred):
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]
    num = len(y_pred)
    sums = 0
    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp
    mape = sums * (100 / num)
    return mape


def SMAPE(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error
    Calculate the mape."""

    y = [x for x in y_true]
    y_pred = [y_pred[i] for i in range(len(y_true))]
    num = len(y_pred)
    sums = 0
    for i in range(num):
        tmp = abs(y_pred[i] - y[i]) / (abs(y[i]) + abs(y_pred[i])) * 2
        sums += tmp
    smape = sums * (100 / num)
    return smape


class Data_split():
    def __init__(self, data):
        self.data = data
        # 將Y切出並轉換Scale
        self.data_y = self.data["RUL"]/3600

        # 將X切出
        self.data_x = self.data.drop(["RUL"], axis=1)

        # X的Scale轉換(Min-Max)
        self.data_scale = minmax_scale(self.data_x)

        # Train, Test split
        testsize = 0.2
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_x,
                                                                                self.data_y,
                                                                                test_size=testsize,
                                                                                random_state=9
                                                                                )
        # Transform to Dataframe
        self.x_train_df = pd.DataFrame(self.x_train)
        self.x_test_df = pd.DataFrame(self.x_test)
        self.y_test_df = pd.DataFrame(self.y_test)
        self.y_train_df = pd.DataFrame(self.y_train)

    def rearrange_random(self):
        # Reset index
        self.y_test_df = self.y_test_df.reset_index(drop=True)
        self.y_train_df = self.y_train_df.reset_index(drop=True)
        self.x_train_df = self.x_train_df.reset_index(drop=True)
        self.x_test_df = self.x_test_df.reset_index(drop=True)
        self.x_train_r = np.array(self.x_train_df)
        self.y_train_r = np.array(self.y_train_df)
        self.x_test_r = np.array(self.x_test_df)
        self.y_test_r = np.array(self.y_test_df)

        return self.x_train_r, self.x_test_r, self.y_train_r, self.y_test_r

    def rearrange_timeseries(self):
        # Rearrage data by time series
        self.all_train = pd.concat([self.x_train_df, self.y_train_df], axis=1)
        self.all_test = pd.concat([self.x_test_df, self.y_test_df], axis=1)
        self.all_train = self.all_train.sort_values(by="RUL", ascending=False)
        self.all_train = self.all_train.reset_index(drop=True)
        self.all_test = self.all_test.sort_values(by="RUL", ascending=False)
        self.all_test = self.all_test.reset_index(drop=True)

        # 重新分割訓練集
        x_train_t = self.all_train.drop(["RUL"], axis=1)
        y_train_t = self.all_train["RUL"]

        # 重新分割測試集
        x_test_t = self.all_test.drop(["RUL"], axis=1)
        y_test_t = self.all_test["RUL"]
        self.x_train_t = np.array(x_train_t)
        self.y_train_t = np.array(y_train_t)
        self.x_test_t = np.array(x_test_t)
        self.y_test_t = np.array(y_test_t)

        return self.x_train_t, self.x_test_t, self.y_train_t, self.y_test_t


if __name__ == "__main__":

    # Read Data
    mypath = os.getcwd()
    filepath = os.path.join(mypath, "data")
    filelist = os.listdir(filepath)
    data_df = {}
    for file in filelist:
        path = os.path.join(filepath, file)
        data_df.setdefault(file, pd.read_csv(path))

    data_df_keys = list(data_df.keys())
    timestart_all = time.time()
    timespend_each = []
    timespend_opt_each = []
    timespend_p2_opt = []

    mse_svr_opt = []
    mse_rf_opt = []
    mse_gbm_opt = []
    mse_stack_p1 = []
    mse_stack_p2 = []

    mape_svr_opt = []
    mape_rf_opt = []
    mape_gbm_opt = []
    mape_stack_p1 = []
    mape_stack_p2 = []

    smape_svr_opt = []
    smape_rf_opt = []
    smape_gbm_opt = []
    smape_stack_p1 = []
    smape_stack_p2 = []

    for key in range(15):
        print(key)
        timestart_each = time.time()
        data = data_df["file1"]
        splitor = Data_split(data)
        x_train, x_test, y_train, y_test = splitor.rearrange_random()
        # Model Construct
        svr_para = {"C": np.linspace(1, 250, num=50),
                    "epsilon": np.linspace(0.001, 0.00001, num=50)
                    }
        rf_para = {"n_estimators": [i for i in range(100, 550, 50)],
                   "max_depth": [i for i in range(6, 16)]
                   }
        gbm_para = {"n_estimators": [i for i in range(100, 550, 50)],
                    "max_depth": [i for i in range(6, 16)],
                    "learning_rate": np.linspace(0.01, 0.5, num=10)
                    }

        model_svr = SVR(kernel="rbf")
        model_rf = RandomForestRegressor(random_state=9)
        model_gbm = lgb.LGBMRegressor()
        # Random Search
        gs_random_svr = RandomizedSearchCV(estimator=model_svr,
                                           param_distributions=svr_para,
                                           cv=5,
                                           n_iter=30,
                                           n_jobs=-1
                                           )
        timesvropt = time.time()
        timesvrfinish = timesvropt-timestart_each
        print("SVR OPT:", timesvrfinish)
        gs_random_rf = RandomizedSearchCV(estimator=model_rf,
                                          param_distributions=rf_para,
                                          cv=5,
                                          n_iter=20,
                                          n_jobs=-1
                                          )
        timerfopt = time.time()
        timerffinish = timesvropt-timesvropt
        print("RF OPT:", timerffinish)
        gs_random_gbm = RandomizedSearchCV(estimator=model_gbm,
                                           param_distributions=gbm_para,
                                           cv=5,
                                           n_iter=20,
                                           n_jobs=-1
                                           )
        timegbmopt = time.time()
        timegbmfinish = timegbmopt-timerfopt
        print("GBM OPT:", timegbmfinish)
        gs_random_svr.fit(x_train, y_train)
        gs_random_rf.fit(x_train, y_train)
        gs_random_gbm.fit(x_train, y_train)
        best_svr = gs_random_svr.best_params_
        best_rf = gs_random_rf.best_params_
        best_gbm = gs_random_gbm.best_params_
        timeend_opt = time.time()
        timespend_opt = timeend_opt - timestart_each
        timespend_opt_each.append(timespend_opt)
        print("Optimize spend time: ", timespend_opt)

        model_svr_opt = SVR(kernel="rbf",
                            C=best_svr["C"],
                            epsilon=best_svr["epsilon"]
                            )
        model_rf_opt = RandomForestRegressor(n_estimators=best_rf["n_estimators"],
                                             max_depth=best_rf["max_depth"],
                                             random_state=9
                                             )
        model_gbm_opt = lgb.LGBMRegressor(n_estimators=best_gbm["n_estimators"],
                                          max_depth=best_gbm["max_depth"],
                                          learning_rate=best_gbm["learning_rate"]
                                          )
        estimators = [("svr", model_svr_opt),
                      ("rfr", model_rf_opt),
                      ("lgb", model_gbm_opt)]
        model_stacking = StackingRegressor(estimators=estimators,
                                           final_estimator=linear_model.LinearRegression(),
                                           n_jobs=-1,
                                           passthrough=False)
        model_svr_opt.fit(x_train, y_train)
        model_rf_opt.fit(x_train, y_train)
        model_gbm_opt.fit(x_train, y_train)
        model_stacking.fit(x_train, y_train)

        pred_svr_opt = model_svr_opt.predict(x_test)
        pred_rf_opt = model_rf_opt.predict(x_test)
        pred_gbm_opt = model_gbm_opt.predict(x_test)
        pred_stack_p1 = model_stacking.predict(x_test)

        mse_svr_opt.append(mean_squared_error(y_test, pred_svr_opt))
        mse_rf_opt.append(mean_squared_error(y_test, pred_rf_opt))
        mse_gbm_opt.append(mean_squared_error(y_test, pred_gbm_opt))
        mse_stack_p1.append(mean_squared_error(y_test, pred_stack_p1))

        mape_svr_opt.append(MAPE(y_test, pred_svr_opt))
        mape_rf_opt.append(MAPE(y_test, pred_rf_opt))
        mape_gbm_opt.append(MAPE(y_test, pred_gbm_opt))
        mape_stack_p1.append(MAPE(y_test, pred_stack_p1))

        smape_svr_opt.append(SMAPE(y_test, pred_svr_opt))
        smape_rf_opt.append(SMAPE(y_test, pred_rf_opt))
        smape_gbm_opt.append(SMAPE(y_test, pred_gbm_opt))
        smape_stack_p1.append(SMAPE(y_test, pred_stack_p1))

        timeend_each = time.time()
        timespend_each.append(timeend_each - timestart_each)
        print("Spend time for each dataset:", (timeend_each - timestart_each))

        # Random Search Result
        rs_result_svr = pd.DataFrame(gs_random_svr.cv_results_)[["params", "rank_test_score"]]
        rs_result_rf = pd.DataFrame(gs_random_rf.cv_results_)[["params", "rank_test_score"]]
        rs_result_gbm = pd.DataFrame(gs_random_gbm.cv_results_)[["params", "rank_test_score"]]

    timeend_all = time.time()
    timespend = timeend_all - timestart_all
    print("Total Spend time: ", timespend, "s")

    print("\nMSE Avg. of Phase I:", np.mean(mse_stack_p1),
          "\nMSE Std. of Phase I:", np.std(mse_stack_p1),
          "\nMAPE Avg. of Phase I:", np.mean(mape_stack_p1),
          "\nMAPE Std. of Phase I:", np.std(mape_stack_p1),
          "\nSMAPE Avg. of Phase I:", np.mean(smape_stack_p1),
          "\nSMAPE Std. of Phase I:", np.std(smape_stack_p1))

    print("\nMSE Avg. of Phase I SVR:", np.mean(mse_svr_opt),
          "\nMSE Std. of Phase I SVR:", np.std(mse_svr_opt),
          "\nMAPE Avg. of Phase I SVR:", np.mean(mape_svr_opt),
          "\nMAPE Std. of Phase I SVR:", np.std(mape_svr_opt),
          "\nSMAPE Avg. of Phase I SVR:", np.mean(smape_svr_opt),
          "\nSMAPE Std. of Phase I SVR:", np.std(smape_svr_opt))

    print("\nMSE Avg. of Phase I RF:", np.mean(mse_rf_opt),
          "\nMSE Std. of Phase I RF:", np.std(mse_rf_opt),
          "\nMAPE Avg. of Phase I RF:", np.mean(mape_rf_opt),
          "\nMAPE Std. of Phase I RF:", np.std(mape_rf_opt),
          "\nSMAPE Avg. of Phase I RF:", np.mean(smape_rf_opt),
          "\nSMAPE Std. of Phase I RF:", np.std(smape_rf_opt))

    print("\nMSE Avg. of Phase I GBM:", np.mean(mse_gbm_opt),
          "\nMSE Std. of Phase I GBM:", np.std(mse_gbm_opt),
          "\nMAPE Avg. of Phase I GBM:", np.mean(mape_gbm_opt),
          "\nMAPE Std. of Phase I GBM:", np.std(mape_gbm_opt),
          "\nSMAPE Avg. of Phase I GBM:", np.mean(smape_gbm_opt),
          "\nSMAPE Std. of Phase I GBM:", np.std(smape_gbm_opt))