import pandas as pd
from sklearn import tree as tree
from sklearn import ensemble as ens
from sklearn import neighbors as nghb
from sklearn import linear_model as lm
import seaborn as sns
from sklearn.model_selection import train_test_split

from pomegranate import *
# import lightgbm as lgb
import xgboost as xgb

from imputing import y_cols, index_cols, evaluate_rmse, evaluate_r_squread, RESULTS_FILE, estimators
from cleaning import txconds_list, studies_list
from c4_5 import dt_c45, prediction_dt_c45

import matplotlib.pyplot as plt
import warnings


t_test_columns = ['pre_cbcl_p1_anxdep', 'pre_cbcl_p1_wthdr', 'pre_cbcl_p1_som', 'pre_cbcl_p1_socprob',
                  'pre_cbcl_p1_thought', 'pre_cbcl_p1_attn', 'pre_cbcl_p1_rulebrk', 'pre_cbcl_p1_aggrs',
                  'pre_cbcl_p1_int',
                  'pre_cbcl_p1_ex', 'pre_cbcl_p1_totprob', 'pre_cbcl_p1_activities', 'pre_cbcl_p1_social',
                  'pre_cbcl_p1_school', 'pre_cbcl_p1_tot', 'pre_cbcl_p1_affprob', 'pre_cbcl_p1_anxprob',
                  'pre_cbcl_p1_somprob', 'pre_cbcl_p1_adhdprob', 'pre_cbcl_p1_oddprob', 'pre_cbcl_p1_condprob',
                  'pre_cbcl_p1_slugcog', 'pre_cbcl_p1_ocdprob', 'pre_cbcl_p1_ptsprob']

non_iterative_imputers = ('mean', 'median', 'knn', 'softimpute', 'em')


# for imputed data
def run_model(X, y, cv_X, model):
    res = pd.DataFrame()
    for col in y_cols:
        model.fit(X, y[col])
        res[col] = model.predict(cv_X)
        if model.__class__.__name__ == 'Lasso':
            with open(RESULTS_FILE, "a") as f:
                f.write('\nFor ' + col + ' predictive features are:\n')
            for i in range(len(model.coef_)):
                if model.coef_[i] != 0:
                    with open(RESULTS_FILE, "a") as f:
                        f.write('\n\t' + X.columns[i] + ', ' + str(model.coef_[i]))
    return res


def read_data(filename, column=None):
    data = pd.read_csv(filename)
    if column is not None:
        data = data[data[column] > 0]
    X = data.drop(y_cols, axis=1)
    X = X.drop(index_cols, axis=1)
    y = data[y_cols]
    return X, y


def model_imputed():
    regressors = [
                  lm.BayesianRidge(),
                  lm.LinearRegression(),
                  lm.Ridge(random_state=101),
                  lm.ElasticNet(random_state=101, max_iter=10000, alpha=0.3),
                  lm.Lasso(alpha=0.3, random_state=101, max_iter=10000),
                  lm.LassoLars(random_state=101, normalize=False),
                  lm.OrthogonalMatchingPursuit(normalize=False),
                  lm.ARDRegression(),
                  tree.DecisionTreeRegressor(max_features="sqrt", random_state=101),
                  ens.ExtraTreesRegressor(n_estimators=10, random_state=101),
                  nghb.KNeighborsRegressor(n_neighbors=5),
                  ens.GradientBoostingRegressor(random_state=101),
                  ens.RandomForestRegressor(max_depth=10, random_state=101),
                  ens.AdaBoostRegressor(random_state=101, base_estimator=lm.ElasticNet(alpha=0.3)),
                  ens.BaggingRegressor(base_estimator=lm.ElasticNet(alpha=0.3)),
                  ]
    for regressor in regressors:
        with open(RESULTS_FILE, "a") as f:
            f.write('\n' + regressor.__class__.__name__)
        for testing in ['cv', 'test', 'withdrawn']:
            with open(RESULTS_FILE, "a") as f:
                f.write('\n' + testing)
            for imp_est in estimators + non_iterative_imputers:  # ['ElasticNet']:
                with open(RESULTS_FILE, "a") as f:
                    f.write('\n' + imp_est)
                train_X, train_y = read_data(f'train_{imp_est}_fitted.csv')
                cv_X, cv_y = read_data(f'{testing}_{imp_est}_fitted.csv')
                missing_cv_X, missing_cv_y = read_data(f'{testing}.csv')
                y_pred = run_model(train_X, train_y, cv_X, regressor)
                for text, cv_y in {'non-missing': cv_y, 'missing': missing_cv_y}.items():

                    rmse, _, _ = evaluate_rmse(y_pred, cv_y, False)
                    r_sq = evaluate_r_squread(y_pred, cv_y, False)
                    with open(RESULTS_FILE, "a") as f:
                        f.write(f'\nRMSE evaluated on {text} data and {imp_est} is {str(rmse)}' +
                                f' and R-squared is {r_sq}')

                # txcond ablation
                sq_sum_all = {'missing': 0, 'non-missing': 0}
                cnt = {'missing': 0, 'non-missing': 0}
                rmse_all = {'missing': 0, 'non-missing': 0}
                for txcond in txconds_list:
                    with open(RESULTS_FILE, "a") as f:
                        f.write('\n' + txcond)
                    train_X, train_y = read_data(f'train_{imp_est}_fitted.csv', txcond)
                    cv_X, cv_y = read_data(f'{testing}_{imp_est}_fitted.csv', txcond)
                    missing_cv_X, missing_cv_y = read_data(f'{testing}.csv', txcond)
                    if train_X.shape[0] == 0 or cv_X.shape[0] == 0:
                        continue
                    y_pred = run_model(train_X, train_y, cv_X, regressor)
                    for text, cv_y in {'non-missing': cv_y.reset_index(drop=True),
                                       'missing': missing_cv_y.reset_index(drop=True)}.items():
                        rmse, sq_sum, n = evaluate_rmse(y_pred, cv_y, False)
                        r_sq = evaluate_r_squread(y_pred, cv_y, False)
                        with open(RESULTS_FILE, "a") as f:
                            f.write(f'\nRMSE evaluated on {text} data and {imp_est} is {str(rmse)}' +
                                    f' and R-squared is {r_sq}')
                        sq_sum_all[text] += sq_sum
                        cnt[text] += n
                for text in ['non-missing', 'missing']:
                    if cnt[text] > 0:
                        rmse_all[text] = sq_sum_all[text] / cnt[text]
                        with open(RESULTS_FILE, "a") as f:
                            f.write(f'\nRMSE evaluated on {text} data and {imp_est} is {str(rmse_all[text])}')

                # study ablation
                sq_sum_all = {'missing': 0, 'non-missing': 0}
                cnt = {'missing': 0, 'non-missing': 0}
                rmse_all = {'missing': 0, 'non-missing': 0}
                for study in studies_list:
                    with open(RESULTS_FILE, "a") as f:
                        f.write('\n' + study)
                    train_X, train_y = read_data(f'train_{imp_est}_fitted.csv', study)
                    cv_X, cv_y = read_data(f'{testing}_{imp_est}_fitted.csv', study)
                    missing_cv_X, missing_cv_y = read_data(f'{testing}.csv', study)
                    if train_X.shape[0] == 0:
                        continue
                    y_pred = run_model(train_X, train_y, cv_X, regressor)
                    for text, cv_y in {'non-missing': cv_y.reset_index(drop=True),
                                       'missing': missing_cv_y.reset_index(drop=True)}.items():
                        rmse, sq_sum, n = evaluate_rmse(y_pred, cv_y, False)
                        r_sq = evaluate_r_squread(y_pred, cv_y, False)
                        with open(RESULTS_FILE, "a") as f:
                            f.write(f'\nRMSE evaluated on {text} data and {imp_est} is {str(rmse)} and R2 is {r_sq}')
                        sq_sum_all[text] += sq_sum
                        cnt[text] += n
                for text in ['non-missing', 'missing']:
                    if cnt[text] > 0:
                        rmse_all[text] = sq_sum_all[text] / cnt[text]
                        with open(RESULTS_FILE, "a") as f:
                            f.write(f'\nRMSE evaluated on {text} data and {imp_est} is {str(rmse_all[text])}')


def bucket_columns(data):
    for col in t_test_columns:
        data.loc[data[col] < -1.6, col] = -5
        data.loc[(data[col] < -1.2) & (data[col] >= -1.6), col] = -4
        data.loc[(data[col] < -0.8) & (data[col] >= -1.2), col] = -3
        data.loc[(data[col] < -0.4) & (data[col] >= -0.8), col] = -2
        data.loc[(data[col] < 0) & (data[col] >= -0.4), col] = -1
        data.loc[(data[col] < 0.4) & (data[col] >= 0), col] = 1
        data.loc[(data[col] < 0.8) & (data[col] >= 0.4), col] = 2
        data.loc[(data[col] < 1.2) & (data[col] >= 0.8), col] = 3
        data.loc[(data[col] < 1.6) & (data[col] >= 1.2), col] = 4
        data.loc[data[col] >= 1.6, col] = 5
    return data


def model_c4_5():
    train_X, train_y = read_data('train.csv')
    cv_X, cv_y = read_data('cv.csv')
    missing_cv_X, missing_cv_y = read_data('cv.csv')
    train_X = bucket_columns(train_X)
    cv_X = bucket_columns(cv_X)
    missing_cv_X = bucket_columns(missing_cv_X)
    y_pred = pd.DataFrame()
    for text, [cv_X, cv_y] in {'non-missing': [cv_X, cv_y], 'missing': [missing_cv_X, missing_cv_y]}.items():
        for col in y_cols:
            with open(RESULTS_FILE, "a") as f:
                f.write(f'\nModel to fit column {col}')
            y = train_y[col].apply(str)
            dt_model = dt_c45(Xdata=train_X, ydata=y)
            with open(RESULTS_FILE, "a") as f:
                f.write(f'\n{dt_model}\n=========================\n\n')
            y_pred[col] = prediction_dt_c45(dt_model, cv_X)
            rmse, _, _ = evaluate_rmse(y_pred, cv_y.apply(str), False)
            r_sq = evaluate_r_squread(y_pred, cv_y.apply(str), False)
            with open(RESULTS_FILE, "a") as f:
                f.write(f'\nRMSE for c4.5 evaluated on {text} data {str(rmse)} and R-squared is {r_sq}')


def xgboost():
    train_X, train_y = read_data('train.csv')
    cv_X, cv_y = read_data('cv.csv')
    missing_cv_X, missing_cv_y = read_data('cv.csv')
    param = {'max_depth': 6, 'eta': 0.3, 'objective': 'reg:squarederror'}
    param['eval_metric'] = 'auc'
    dtrain = xgb.DMatrix(train_X, label=train_y)
    dtest = xgb.DMatrix(cv_X, label=cv_y)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_boost_round=10, evals=evallist)
    ypred = bst.predict(dtest)


# def light_gbm():
#     train_X, train_y = read_data('train.csv')
#     cv_X, cv_y = read_data('cv.csv')
#     missing_cv_X, missing_cv_y = read_data('cv.csv')
#     model = lgb.LGBMRegressor()
#     model.fit(train_X, train_y)
#     y_pred = model.predict(cv_X)
#     for text, cv_y in {'non-missing': cv_y, 'missing': missing_cv_y}.items():
#         rmse, _, _ = evaluate_rmse(y_pred, cv_y, False)
#         r_sq = evaluate_r_squread(y_pred, cv_y, False)
#         with open(RESULTS_FILE, "a") as f:
#             f.write(f'\nRMSE evaluated on {text} data {str(rmse)} and R-squared is {r_sq}')


# def naive_bayes():
#     train_X, train_y = read_data('train.csv')
#     cv_X, cv_y = read_data('cv.csv')
#     missing_cv_X, missing_cv_y = read_data('cv.csv')
#     model = NaiveBayes.from_samples(NormalDistribution, train_X, train_y)
#     y_pred = model.predict(cv_X)
#     for text, cv_y in {'non-missing': cv_y, 'missing': missing_cv_y}.items():
#         rmse, _, _ = evaluate_rmse(y_pred, cv_y, False)
#         r_sq = evaluate_r_squread(y_pred, cv_y, False)
#         with open(RESULTS_FILE, "a") as f:
#             f.write(f'\nRMSE evaluated on {text} data {str(rmse)} and R-squared is {r_sq}')


def graph_best_and_average_prediction_over_imputers(imputers):
    best_prediction_missing = [1.4321572135, 1.4222136875, 1.4225801525, 1.4228678135, 1.434121045, 1.44950567,
                               1.42950461925, 1.4324683265, 1.443401223375, 1.447903097775, 1.35058418,	1.34954217]
    best_prediction_missing_studies = [2.128897799, 2.1106236495, 2.1113269195, 2.11132796, 2.1717511783, 2.1728888185,
                                       2.2214302715, 2.133511703, 2.221944561, 2.264071755, 1.9772134653,
                                       1.9644707311]
    avg_prediction_missing = [1.5371888018061, 1.52462348339129, 1.52221951758673, 1.52567670806516, 1.53777416649433,
                              1.54528668549089, 1.53369274704816, 1.52437375487417, 1.56172123676967, 1.54044336414463,
                              1.45466083419933, 1.48923358624767]
    avg_prediction_missing_studies = [2.45401810434812, 2.42099211103333, 2.426103355342, 2.50223196398053,
                                      2.44550466691033, 2.60341684163767, 2.58734529215677, 2.54377642577023,
                                      2.80727758539683, 2.58049640235515, 2.33135873080713, 2.2756689625578]
    worst_prediction_missing = [1.813671015, 1.8487237524, 1.8279126575, 1.78199401075, 1.7999405045, 1.855284438,
                                1.8192860215, 1.75803442255, 1.8628097910141, 1.8018992405, 1.7267431433,
                                1.94094732151]
    worst_prediction_missing_studies = [3.59144876777174, 3.104843475, 3.264027018, 3.651252205, 3.1936900660435,
                                        3.43503575, 3.612905393, 3.6147903, 5.42654487, 3.775443353345, 3.27465015,
                                        3.14454455965]
    plt.plot(imputers, best_prediction_missing, label="Bayesian Ridge prediction full (best)", linestyle='dashed')
    plt.plot(imputers, avg_prediction_missing, label="Average prediction full", linestyle='dashed')
    plt.plot(imputers, worst_prediction_missing, label="Extra Trees prediction full (worst)", linestyle='dashed')

    plt.plot(imputers, best_prediction_missing_studies, label="Bayesian Ridge prediction avg study (best)")
    plt.plot(imputers, avg_prediction_missing_studies, label="Average prediction avg study")
    plt.plot(imputers, worst_prediction_missing_studies, label="Decision Tree prediction avg study (worst)")
    plt.legend(loc='upper left', prop={'size': 8})
    plt.title('Performance of the best, average, and worst predictor\n '
              'over imputation methods evaluated on cross validation')
    plt.ylabel('RMSE')
    plt.xlabel('Imputation method')
    plt.savefig('predictors_over_imputers.png')
    plt.clf()


def graph_best_and_average_prediction_over_imputers_withdrawal(imputers):
    best_prediction_missing = [1.8816214856, 1.8286929575, 1.82728683678, 1.864352319, 1.879976867, 1.86539160498,
                               1.8791478, 1.850841598, 1.846402406, 1.85463358,	1.8693573329, 1.82853386]
    best_prediction_missing_studies = [3.750877071, 3.6813532, 3.681622057, 3.689526265074, 3.741147296, 4.002379789,
                                       3.68520704311, 3.718534074655, 3.74696495, 3.800834792, 3.710225649, 3.68206404]
    avg_prediction_missing = [1.99569761687413, 1.96292015265667, 1.96739499014067, 1.97823477892667, 1.98136757543927,
                              1.98754249202267, 1.9760178196892, 1.9694912997172, 2.015292631982, 1.98594783817,
                              1.99123876764867, 1.98518098800287]
    avg_prediction_missing_studies = [4.34807835274267, 4.2370231598, 4.2412566868952, 4.232619814652, 4.3116905791,
                                      4.51152985862738, 4.36187266645933, 4.33618738621633, 4.327253180296,
                                      4.41184607989622, 4.36429353135333, 4.24219206854667]
    worst_prediction_missing = [2.3733514712, 2.43027107, 2.4762545475, 2.539911605, 2.443598676, 2.3984561, 2.3773287,
                                2.51584293, 2.5382036489, 2.4439797878, 2.496265374, 2.6373172579]
    worst_prediction_missing_studies = [6.5605458584, 6.229349287, 6.1838961387, 6.180481444, 6.024832407, 6.477099236,
                                        6.5208874, 6.5193893, 5.8167214446, 6.47213561939326, 6.57504715, 5.92325412]
    plt.plot(imputers, best_prediction_missing, label="OMP prediction full (best)", linestyle='dashed')
    plt.plot(imputers, avg_prediction_missing, label="Average prediction full", linestyle='dashed')
    plt.plot(imputers, worst_prediction_missing, label="Decision Trees prediction full (worst)", linestyle='dashed')

    plt.plot(imputers, best_prediction_missing_studies, label="Lasso prediction avg study (best)")
    plt.plot(imputers, avg_prediction_missing_studies, label="Average prediction avg study")
    plt.plot(imputers, worst_prediction_missing_studies, label="Decision Tree prediction avg study (worst)")
    plt.legend(loc='center left', bbox_to_anchor=(0., 0.0, 0.0, 1.42), prop={'size': 8})
    plt.title('Performance of the best, average, and worst predictor\n '
              'over imputation methods evaluated on withdrawn data')
    plt.ylabel('RMSE')
    plt.xlabel('Imputation method')
    plt.savefig('withdrawal predictors_over_imputers.png')
    plt.clf()


def graph_imputation_heatmap():
    data = pd.read_csv('imp_heatmap.csv')
    sns.heatmap(data.iloc[:12, 1:])
    plt.savefig('imp_heatmap.png')
    plt.clf()
    sns.heatmap(data.iloc[12:, 1:])
    plt.savefig('log_imp_heatmap.png')
    plt.clf()


def predictions_on_lesley_data():
    regressors = [
        lm.BayesianRidge(),
        lm.LinearRegression(),
        lm.Ridge(random_state=101),
        lm.ElasticNet(random_state=101, max_iter=10000, alpha=0.3),
        lm.Lasso(alpha=0.3, random_state=101, max_iter=10000),
        lm.LassoLars(random_state=101, normalize=False),
        lm.OrthogonalMatchingPursuit(normalize=False),
        lm.ARDRegression(),
        tree.DecisionTreeRegressor(max_features="sqrt", random_state=101),
        ens.ExtraTreesRegressor(n_estimators=10, random_state=101),
        nghb.KNeighborsRegressor(n_neighbors=5),
        ens.GradientBoostingRegressor(random_state=101),
        ens.RandomForestRegressor(max_depth=10, random_state=101),
        ens.AdaBoostRegressor(random_state=101, base_estimator=lm.ElasticNet(alpha=0.3)),
        ens.BaggingRegressor(base_estimator=lm.ElasticNet(alpha=0.3)),
    ]

    X, y = read_data('lesley.csv')
    X = X.dropna(axis='columns')
    y = y.dropna(axis='columns')
    train_X, cv_X, train_y, cv_y = train_test_split(X, y, test_size=0.2, random_state=42)
    train_X = train_X.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    cv_X = cv_X.reset_index(drop=True)
    cv_y = cv_y.reset_index(drop=True)

    RESULTS_FILE = 'results_lesley.txt'

    for regressor in regressors:
        with open(RESULTS_FILE, "a") as f:
            f.write('\n' + regressor.__class__.__name__)
        y_pred = run_model(train_X, train_y, cv_X, regressor)
        for text, cv_y in {'non-missing': cv_y}.items():
            rmse, _, _ = evaluate_rmse(y_pred, cv_y, False)
            r_sq = evaluate_r_squread(y_pred, cv_y, False)
            with open(RESULTS_FILE, "a") as f:
                f.write(f'\nRMSE evaluated on {text} data is {str(rmse)}' +
                        f' and R-squared is {r_sq}')


def transfer_learning_to_lesley_data():
    regressors = [
        lm.BayesianRidge(),
        lm.LinearRegression(),
        lm.Ridge(random_state=101),
        lm.ElasticNet(random_state=101, max_iter=10000, alpha=0.3),
        lm.Lasso(alpha=0.3, random_state=101, max_iter=10000),
        lm.LassoLars(random_state=101, normalize=False),
        lm.OrthogonalMatchingPursuit(normalize=False),
        lm.ARDRegression(),
        tree.DecisionTreeRegressor(max_features="sqrt", random_state=101),
        ens.ExtraTreesRegressor(n_estimators=10, random_state=101),
        nghb.KNeighborsRegressor(n_neighbors=5),
        ens.GradientBoostingRegressor(random_state=101),
        ens.RandomForestRegressor(max_depth=10, random_state=101),
        ens.AdaBoostRegressor(random_state=101, base_estimator=lm.ElasticNet(alpha=0.3)),
        ens.BaggingRegressor(base_estimator=lm.ElasticNet(alpha=0.3)),
    ]

    RESULTS_FILE = 'results_lesley.txt'

    for regressor in regressors:
        with open(RESULTS_FILE, "a") as f:
            f.write('\n' + regressor.__class__.__name__)
        for testing in ['lesley']:
            with open(RESULTS_FILE, "a") as f:
                f.write('\n' + testing)
            for imp_est in estimators + non_iterative_imputers:  # ['ElasticNet']:
                with open(RESULTS_FILE, "a") as f:
                    f.write('\n' + imp_est)
                train_X, train_y = read_data(f'train_{imp_est}_fitted.csv')
                cv_X, cv_y = read_data(f'{testing}.csv')
                cv_X = cv_X.dropna(axis='columns')
                cv_y = cv_y.dropna(axis='columns')
                train_X = train_X[train_X.columns.intersection(cv_X.columns)]
                train_y = train_y[train_y.columns.intersection(cv_y.columns)]
                y_pred = run_model(train_X, train_y, cv_X, regressor)
                for text, cv_y in {'non-missing': cv_y}.items():
                    rmse, _, _ = evaluate_rmse(y_pred, cv_y, False)
                    r_sq = evaluate_r_squread(y_pred, cv_y, False)
                    with open(RESULTS_FILE, "a") as f:
                        f.write(f'\nRMSE evaluated on {text} data and {imp_est} is {str(rmse)}' +
                                f' and R-squared is {r_sq}')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # model_imputed()
    # model_c4_5()
    # xgboost()
    # light_gbm()
    # naive_bayes()
    # imputers = ['Ridge', 'EN', 'Lasso', 'OMP', 'ARD', 'DT', 'ET', 'KNN', 'GB', 'RF', 'Ada', 'Bagging']
    # graph_best_and_average_prediction_over_imputers(imputers)
    # graph_best_and_average_prediction_over_imputers_withdrawal(imputers)
    # graph_imputation_heatmap()

    # predictions_on_lesley_data()
    transfer_learning_to_lesley_data()