#standard libraries
import gc
import argparse

#3rd party libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RepeatedKFold, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA

#customs
from helpers.funcs import readexcel_set_df_names
from helpers.funcs import resample, resample_dask
from helpers.funcs import timeseries_train_test_split as TSSplit
from helpers.funcs import fit_model_cv
from helpers.funcs import iqr_filter_outliers
from helpers.anomalies import SimpleAnomalies

#Default Globals
fname = 'data_analytics.xlsx'
NB_Seed = 123123

parser = argparse.ArgumentParser()
parser.add_argument('--fname', action='store', dest='file_name', help='Input excel filename.', type=str, default='data_analytics.xlsx')
parser.add_argument('--seed', action='store', dest='rand_seed', help='Random seed for some of the models.', type=int, default=123123)

results = parser.parse_args()



def main():
    print("Reading train and test data from excel file: " + str(fname))
    
    tr = readexcel_set_df_names(fname, 'Train')
    ts = readexcel_set_df_names(fname, 'Test')

    print("Resampling train data to make its frequency strictly 8hours.")
    tr = tr.resample(rule='8H', base=00).mean().interpolate(method='time')
    ts = ts.resample(rule='15T').mean()

    trres = pd.DataFrame()
    for v in tr.columns.values:
        print("Using custom 2-step resampling for variable: " + str(v))
        trres[v] = resample_dask(tr[v])
    del tr;gc.collect()
    tr = trres.copy()

    print("Applying inter quartile range filtering to train dataset for anomaly detection and imputation.")
    bool_tr = iqr_filter_outliers(tr.Target, inplace=False)
    tr[bool_tr] = np.nan
    tr = tr.interpolate(method='time')
    tr.fillna(tr.mean(), inplace=True)
    
    x_tr, x_ts, y_tr, y_ts = train_test_split(tr.drop('Target',1), tr.Target, test_size=0.25, random_state = NB_Seed )
    
    scaler = StandardScaler()
    x_tr_sc = scaler.fit_transform(x_tr)
    x_ts_sc = scaler.fit_transform(x_ts)
    rkf = RepeatedKFold(n_splits=5, n_repeats=2)
    print("Fitting LassoCV using TimeSeriesSplit cross validation generator.")
    lasso = LassoCV(cv=rkf, random_state=NB_Seed, max_iter=2000, tol=0.001)
    lasso.fit(x_tr_sc, y_tr)
    coefs = pd.DataFrame(lasso.coef_, x_tr.columns)
    coefs.columns = ["coef"]
    coefs = coefs[coefs.coef!=0]
    nonzeroftrs = list(coefs.index.values)
    nonzeroftrs.append('Target')
    print("Dropping zero weight variables from train and test sets.")
    trnz = tr.loc[:,nonzeroftrs].copy()
    ts = ts.loc[:,nonzeroftrs].copy()
    del scaler, x_tr_sc, x_ts_sc, lasso, coefs, nonzeroftrs;gc.collect()

    tr_ = tr.copy()
    #naming convention
    del tr;gc.collect()
    tr = trnz.copy()
    del trnz;gc.collect()

    #Make time-features
    print("Building time-features dataframe.")
    tr_Target = tr.Target.copy()
    tr_Target_ = tr.Target.copy()
    tr_tf = tr.copy()
    ts_tf = ts.copy()
    tr_tf['Date'] = tr_tf.index.values
    tr_tf['Weekday'] = tr_tf.Date.apply(lambda x: x.weekday())
    tr_tf['Hour'] = tr_tf.Date.apply(lambda x: x.hour)
    tr_tf['Month'] = tr_tf.Date.apply(lambda x: x.month)

    ts_tf['Date'] = ts_tf.index.values
    ts_tf['Weekday'] = ts_tf.Date.apply(lambda x: x.weekday())
    ts_tf['Hour'] = ts_tf.Date.apply(lambda x: x.hour)
    ts_tf['Month'] = ts_tf.Date.apply(lambda x: x.month)

    tr_tf = tr_tf.loc[:, tr_tf.dtypes != "float64"].join(tr_Target)
    ts_tf = ts_tf.loc[:, ts_tf.dtypes != "float64"]
    tr_tf = pd.get_dummies(tr_tf, columns=["Month","Hour","Weekday"])
    ts_tf = pd.get_dummies(ts_tf, columns=["Month","Hour","Weekday"])
    tr_tf.drop('Date',1,inplace=True)
    ts_tf.drop('Date',1,inplace=True)

    #Scale rest of the dataframe
    print("Scaling train/test frames between 0-1 for the prediction phase.")
    trsc = MinMaxScaler()
    tssc = MinMaxScaler()
    tr_nf = tr.loc[:,tr.dtypes!="float64"]
    ts_nf = ts.loc[:,ts.dtypes!="float64"]
    tr_cont = tr.loc[:,tr.dtypes=="float64"].drop('Target',1).interpolate(method='time')
    tr_cont_names = tr_cont.columns.values
    tr_cont_idx = tr_cont.index
    ts_cont = ts.loc[:,ts.dtypes=="float64"]
    ts_cont_names = ts_cont.columns.values
    ts_cont_idx = ts_cont.index
    tr_cont = trsc.fit_transform(tr_cont)
    ts_cont = tssc.fit_transform(ts_cont)
    tr_cont = pd.DataFrame(tr_cont)
    ts_cont = pd.DataFrame(ts_cont)
    tr_cont.columns = tr_cont_names
    ts_cont.columns = ts_cont_names
    tr_cont.index = tr_cont_idx
    ts_cont.index = ts_cont_idx
    tr_num = tr_cont.join(tr_Target)#.join(tr_Target)
    ts_num = ts_cont.copy()
    del tr_Target;gc.collect()
    del tr_nf, ts_nf, tr_cont;gc.collect()
    del tr_cont_names, tr_cont_idx;gc.collect()
    del ts_cont, ts_cont_names, ts_cont_idx;gc.collect()

    #Split
    x_tr_n, x_val_n, y_tr_n, y_val_n = train_test_split(tr.drop('Target',1), tr.Target, test_size=0.25, random_state = NB_Seed )

    #Dimensionality reduction before splitting time-features dataframe
    print("Reducing train and prediction(test) dataframe shapes to 30.")
    trpca = PCA(n_components=30)
    tspca = PCA(n_components=30)
    tr_tf = pd.DataFrame(data=trpca.fit_transform(tr_tf),index=tr_tf.index)
    tr_tf = tr_tf.join(tr_Target_)
    ts_tf = pd.DataFrame(data=tspca.fit_transform(ts_tf),index=ts_tf.index)
    x_tr_t, x_val_t, y_tr_t, y_val_t = train_test_split(tr_tf.drop('Target',1),tr_tf.Target, test_size=0.25, random_state = NB_Seed )

    #RMSE
    def rmse(ytrue, ypred):
        return np.sqrt(mean_squared_error(ytrue, ypred))

    #Scorers
    scrl = [mean_absolute_error, mean_squared_error, rmse, r2_score]
    scrn = ['MAE', 'MSE', 'RMSE', 'R2']


    def print_metric(metrictr, metricts ,mname):
        print(mname +' train: ' + str(metrictr) + ' test: ' + str(metricts))

    #best stacked model
    ada = AdaBoostRegressor(random_state=NB_Seed)
    ext = ExtraTreesRegressor(random_state=NB_Seed)
    xgb_t = XGBRegressor(n_jobs=-1,random_state=NB_Seed,reg_lambda=0.3)
    metalr = LinearRegression(n_jobs=-1)
    stk = StackingCVRegressor(regressors=(ada,ext,xgb_t),
                                 meta_regressor=metalr,
                                 cv=rkf,
                                 use_features_in_secondary=True)

    stk.fit(x_tr_n.values, y_tr_n.values)
    #Print out score metrics for numerical stacked models
    for i, s in enumerate(scrl):
        print_metric(s(y_tr_n.values, stk.predict(x_tr_n.values)), s(y_val_n.values, stk.predict(x_val_n.values)), scrn[i])

    #best timefeatures model
    reg = SVR(kernel='linear')
    mdls = fit_model_cv(reg, x_tr_t.values, y_tr_t.values, rkf)

    #preds for accuracy calculations
    y_pred_val = np.zeros((y_val_t.shape[0],len(mdls)))
    y_pred_tr = np.zeros((y_tr_t.shape[0],len(mdls)))
    for i, m in enumerate(mdls):
        y_pred_val[:,i]= m.predict(x_val_t.values)
        y_pred_tr[:,i] = m.predict(x_tr_t.values)
    y_pred_val = y_pred_val.mean(axis=1)
    y_pred_tr = y_pred_tr.mean(axis=1)
    #Print out score metrics for numerical stacked models
    for i, s in enumerate(scrl):
        print_metric(s(y_tr_t.values, y_pred_tr), s(y_val_t.values, y_pred_val), scrn[i])

    ###Making final-actual predictions
    #From sensor variables
    y_pred_final_num = stk.predict(ts_num.values)
    #From time features
    y_pred_final_time = np.zeros((ts_tf.shape[0],len(mdls)))
    for i, m in enumerate(mdls):
        y_pred_final_time[:,i]= m.predict(ts_tf.values)
    y_pred_final_time = y_pred_final_time.mean(axis=1)

    final_pred = y_pred_final_num * 0.8 + y_pred_final_time * 0.2

    submission = pd.DataFrame({'Timestamp':ts_tf.index,
                               'Target':final_pred})

    #Save submission
    print("Trying to write file as xlsx, in absence of xlsxwriter, it will be written as csv.")
    try:
        writer = pd.ExcelWriter('submission.xlsx', engine='xlsxwriter')
        submission.to_excel(writer)
        writer.save()
    except ModuleNotFoundError:
        print("File is written as CSV.")
        submission.to_csv('submission.csv',index=False)

if __name__ == "__main__":
    fname = results.file_name
    NB_Seed = results.rand_seed
    main()
