from autoimpute.imputations import MultipleImputer, MiceImputer
from sklearn.experimental import enable_iterative_imputer
import sklearn.impute as imp
from fancyimpute import KNN, SoftImpute, IterativeSVD, MatrixFactorization, NuclearNormMinimization
from sklearn import tree as tree
from sklearn import ensemble as ens
from sklearn import neighbors as nghb
from sklearn import linear_model as lm
import warnings
import statsmodels.api as sm
from impyute.imputation.cs import em, fast_knn
import pandas as pd
import numpy as np
import sys
import sklearn.neighbors._base
from sklearn.metrics import r2_score
from cleaning import studies_list

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import KNNImputer, MissForest



RESULTS_FILE = 'results.txt'

# skip this cols when imputing because they don't have missing data
non_missing_cols = ['txsessions', 'childage', 'childsex', 'studycode_1', 'studycode_2', 'studycode_3', 'studycode_4',
                    'studycode_5', 'studycode_6', 'studycode_7', 'studycode_8', 'studycode_9', 'txcond_0', 'txcond_1',
                    'txcond_2', 'txcond_3', 'txcond_4', 'txcond_6', 'txcond_7', 'txcond_8',
                    'childracecomb_1', 'childracecomb_2', 'childracecomb_3']
# label columns
y_cols = ['post_compcsr_sep_dsm4', 'post_compcsr_soc_dsm4', 'post_compcsr_gad_dsm4']
index_cols = ['Unnamed: 0']
binary_cols = ['childethn']

autoimpute_pairs = [('default predictive', 'default predictive'), ('binary logistic', 'least squares'),
                    ('binary logistic', 'stochastic'), ('bayesian binary logistic', 'bayesian least squares'),
                    ('binary logistic', 'lrd')]

estimators = (
    'BayesianRidge',
    'LinearRegression', 'Ridge',
    'ElasticNet',
    'Lasso', 'LassoLars',
    'OrthogonalMatchingPursuit', 'ARDRegression', 'DecisionTreeRegressor',
    'ExtraTreesRegressor',
    'KNeighborsRegressor',
    'AdaBoostRegressor',
    'GradientBoostingRegressor',
    'RandomForestRegressor',
    'BaggingRegressor',
)


def mask_data(filename):
    data = pd.read_csv(filename)
    data = data.drop(y_cols, axis=1)
    data = data.drop(index_cols, axis=1)
    columns = list(set(data.columns) - set(non_missing_cols))
    mask = data[columns].mask(np.random.random(data[columns].shape) < .9)
    mask[non_missing_cols] = None
    masked = data.copy()
    masked[mask.notnull()] = None
    for col in data.columns:
        if masked[col].isnull().sum() == masked.shape[1]:
            with open(RESULTS_FILE, "a") as f:
                f.write(f'Empty column{col}\n')
    return mask, masked


def evaluate_r_squread(predicted, true, mask=True):
    if mask:
        columns = list(set(true.columns) - set(non_missing_cols))
    else:
        columns = true.columns
    true_values = []
    predicted_values = []
    for col in columns:
        not_masked_true = true[col].notnull().tolist()
        not_masked_predicted = predicted[col].notnull().tolist()
        for i in range(true.shape[0]):
            if not_masked_true[i] and not_masked_predicted[i]:
                true_values.append(true.loc[i, col])
                predicted_values.append(predicted.loc[i, col])
    if len(true_values) > 0:
        return r2_score(true_values, predicted_values)
    return 0


def evaluate_rmse(predicted, true, mask=True):
    if mask:
        columns = list(set(true.columns) - set(non_missing_cols))
    else:
        columns = true.columns
    sq_diff_sum = {}
    cnt = {}
    rmse = {}
    for col in columns:
        sq_diff_sum[col] = 0
        cnt[col] = 0
    for col in columns:
        not_masked_true = true[col].notnull().tolist()
        not_masked_predicted = predicted[col].notnull().tolist()
        for i in range(true.shape[0]):
            if not_masked_true[i] and not_masked_predicted[i]:
                sq_diff_sum[col] += (true.loc[i, col] - predicted.loc[i, col]) ** 2
                cnt[col] += 1
        if cnt[col] != 0:
            rmse[col] = np.sqrt(sq_diff_sum[col] / cnt[col])
            # print(rmse[col], col)
    sq_sum = sum(list(sq_diff_sum.values()))
    n = sum(list(cnt.values()))
    rmse_tot = 0
    if n != 0:
        rmse_tot = np.sqrt(sq_sum / n)
    return rmse_tot, sq_sum, n


def set_strategy(columns, categorical='binary logistic', numerical='least squares'):
    strategies = {}
    for col in columns:
        if col in binary_cols:
            strategies[col] = categorical
        else:
            strategies[col] = numerical
    return strategies


def autoimpute_multiple_imputer(masked, mask=None, imputer=None):
    for pair in autoimpute_pairs:
        columns = list(set(masked.columns) - set(non_missing_cols))
        strategy = set_strategy(columns, pair[0], pair[1])
        if imputer is None:
            imputer = MultipleImputer(strategy=strategy, n=5, predictors='all', seed=101, visit='default')
            imputer.fit(masked)
        fitted = imputer.transform(masked)
        if mask is not None:
            rmse, _, _ = evaluate_rmse(fitted, mask)
            r_sq = evaluate_r_squread(fitted, mask)
            with open(RESULTS_FILE, "a") as f:
                f.write(f'\n {pair}, RMSE={rmse}, r-squuared={r_sq}')
        return imputer, fitted


def autoimpute_mice(masked, mask=None, imputer=None):
    for pair in autoimpute_pairs:
        columns = list(set(masked.columns) - set(non_missing_cols))
        strategy = set_strategy(columns, pair[0], pair[1])
        if imputer is None:
            imputer = MiceImputer(strategy=strategy, k=3, n=5, predictors='all', seed=101, visit='default')
            imputer.fit(masked)
        fitted = imputer.transform(masked)
        if mask is not None:
            rmse, _, _ = evaluate_rmse(fitted, mask)
            r_sq = evaluate_r_squread(fitted, mask)
            with open(RESULTS_FILE, "a") as f:
                f.write(f'\n {pair}, RMSE={rmse}, r-squared={r_sq}')
        return imputer, fitted


def sklearn_simple_mean_imputer(masked, mask=None, imputer=None):
    if imputer is None:
        imputer = imp.SimpleImputer()
        imputer.fit(masked)
    fitted = pd.DataFrame(imputer.transform(masked))
    fitted.columns = masked.columns
    if mask is not None:
        rmse, _, _ = evaluate_rmse(fitted, mask)
        r_sq = evaluate_r_squread(fitted, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nscikit-learn simple mean imputer RMSE={rmse} and r-squared={r_sq}')
    return imputer, fitted


def sklearn_simple_median_imputer(masked, mask=None, imputer=None):
    if imputer is None:
        imputer = imp.SimpleImputer(strategy='median')
        imputer.fit(masked)
    fitted = pd.DataFrame(imputer.transform(masked))
    fitted.columns = masked.columns
    if mask is not None:
        rmse, _, _ = evaluate_rmse(fitted, mask)
        r_sq = evaluate_r_squread(fitted, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nscikit-learn simple median imputer RMSE={rmse} and r-squared={r_sq}')
    return imputer, fitted


def sklearn_iterative_imputer(masked, mask=None, imputer=None, est=estimators):
    with open(RESULTS_FILE, "a") as f:
        f.write(f'\nElasticNet, alpha=0.3')
    # print('Bagging estimators=20, Alpha is 0.3, warm start=False, bootstrap_features=True, oob_score=True')
    estimators = [
        lm.BayesianRidge(),  # suggested
        lm.LinearRegression(),
        lm.Ridge(random_state=101),
        lm.ElasticNet(random_state=101, max_iter=10000, alpha=0.3),
        lm.Lasso(random_state=101, max_iter=10000),
        lm.LassoLars(random_state=101, normalize=False),
        lm.OrthogonalMatchingPursuit(normalize=False),
        lm.ARDRegression(),
        tree.DecisionTreeRegressor(max_features="sqrt", random_state=101),  # suggested
        ens.ExtraTreesRegressor(n_estimators=10, random_state=101),  # suggested
        nghb.KNeighborsRegressor(n_neighbors=5),  # suggested
        ens.AdaBoostRegressor(random_state=101, base_estimator=lm.ElasticNet()),
        ens.GradientBoostingRegressor(random_state=101),
        ens.RandomForestRegressor(max_depth=10, random_state=101),  # suggested
        ens.BaggingRegressor(base_estimator=lm.ElasticNet(
            alpha=0.3, max_iter=10000, random_state=101), n_estimators=20,
                             warm_start=False, bootstrap_features=True, oob_score=True)
    ]
    for i in range(len(estimators)):
        if estimators[i].__class__.__name__ in est:
            if imputer is None:
                imputer = imp.IterativeImputer(estimator=estimators[i], random_state=101)
                imputer.fit(masked)
            fitted = pd.DataFrame(imputer.transform(masked))
            fitted.columns = masked.columns
            if mask is not None:
                rmse, _, _ = evaluate_rmse(fitted, mask)
                r_sq = evaluate_r_squread(fitted, mask)
                with open(RESULTS_FILE, "a") as f:
                    f.write(f'\nscikit-learn iterative imputer {estimators[i].__class__.__name__} RMSE={rmse} and ' +
                            f' r-squared={r_sq}')
                imputer = None
            else:
                return imputer, fitted, estimators[i].__class__.__name__


def sklearn_knn_imputer(masked, mask=None, imputer=None):
    # for n in range(2, mask.shape[1]):
    n = 30
    if imputer is None:
        imputer = imp.KNNImputer(n_neighbors=n)
        imputer.fit(masked)
    fitted = pd.DataFrame(imputer.transform(masked))
    fitted.columns = masked.columns
    if mask is not None:
        rmse, _, _ = evaluate_rmse(fitted, mask)
        r_sq = evaluate_r_squread(fitted, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nscikit-learn knn imputer N={n}, RMSE={rmse} and r-squared={r_sq})')
    return imputer, fitted


def fancyimpute_knn_imputer(masked, mask=None):
    # for n in range(2, mask.shape[1]):
    n = 39
    imputer = KNN(k=n)
    maskednp = masked.to_numpy()
    fitted = pd.DataFrame(imputer.fit_transform(maskednp))
    fitted.columns = masked.columns
    if mask is not None:
        rmse, _, _ = evaluate_rmse(fitted, mask)
        r_sq = evaluate_r_squread(fitted, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nsfancyimpute knn imputer N={n}, RMSE={rmse} and r-squared={r_sq})')
    return imputer, fitted


# acording to documentation BiScalar normalizer should be used, but then we can't do RMSE on masked values
def fancyimpute_soft_impute(masked, mask=None):
    imputer = SoftImpute()
    maskednp = masked.to_numpy()
    fitted = pd.DataFrame(imputer.fit_transform(maskednp))
    fitted.columns = masked.columns
    if mask is not None:
        rmse, _, _ = evaluate_rmse(fitted, mask)
        r_sq = evaluate_r_squread(fitted, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nfancyimpute soft imputer RMSE={rmse} and r-squared={r_sq})')
    return imputer, fitted


def fancyimpute_iterative_svd(masked, mask=None):
    imputer = IterativeSVD()
    maskednp = masked.to_numpy()
    fitted = pd.DataFrame(imputer.fit_transform(maskednp))
    fitted.columns = masked.columns
    if mask is not None:
        rmse, _, _ = evaluate_rmse(fitted, mask)
        r_sq = evaluate_r_squread(fitted, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nfancyimpute iterative svd imputer, RMSE={rmse} and r-squared={r_sq})')
    return imputer, fitted


# gives infinite RMSE, ie. probably +/- inf predicton, so can't be used
def fancyimpute_matrix_factorization(masked, mask=None):
    imputer = MatrixFactorization()
    maskednp = masked.to_numpy()
    fitted = pd.DataFrame(imputer.fit_transform(maskednp))
    fitted.columns = masked.columns
    if mask is not None:
        rmse, _, _ = evaluate_rmse(fitted, mask)
        r_sq = evaluate_r_squread(fitted, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nfancyimpute matrix factorization imputer, RMSE={rmse} and r-squared={r_sq})')
    return imputer, fitted


# gives an error that number of elements exceeds INT_MAX
def fancyimpute_nuclear_norm_minimization(masked, mask=None):
    imputer = NuclearNormMinimization()
    maskednp = masked.to_numpy()
    fitted = pd.DataFrame(imputer.fit_transform(maskednp))
    fitted.columns = masked.columns
    if mask is not None:
        rmse, _, _ = evaluate_rmse(fitted, mask)
        r_sq = evaluate_r_squread(fitted, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nfancyimpute nuclear norm minimization imputer, RMSE={rmse} and r-squared={r_sq})')
    return imputer, fitted


def missingpy_knn_imputer(masked, mask=None, imputer=None):
    for n in range(2, mask.shape[1]):
        if imputer is None:
            imputer = KNNImputer(n_neighbors=n, row_max_missing=0.9, col_max_missing=0.9)
            imputer.fit(masked)
        fitted = pd.DataFrame(imputer.transform(masked))
        fitted.columns = masked.columns
        if mask is not None:
            rmse, _, _ = evaluate_rmse(fitted, mask)
            r_sq = evaluate_r_squread(fitted, mask)
            with open(RESULTS_FILE, "a") as f:
                f.write(f'\nmissingpy knn imputer, RMSE={rmse} and r-squared={r_sq})')
        return imputer, fitted


def missingpy_missforest(masked, mask=None, imputer=None):
    if imputer is None:
        imputer = MissForest()
        imputer.fit(masked)
    fitted = pd.DataFrame(imputer.transform(masked))
    fitted.columns = masked.columns
    if mask is not None:
        rmse, _, _ = evaluate_rmse(fitted, mask)
        r_sq = evaluate_r_squread(fitted, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nmissingpy missforest, RMSE={rmse} and r-squared={r_sq})')
    return imputer, fitted


def statsmodels_bayes_gauss_mi(masked, mask=None):
    imputer = sm.MI(sm.BayesGaussMI(masked), sm.OLS)
    fitted = imputer.fit()
    if mask is not None:
        rmse, _, _ = evaluate_rmse(fitted, mask)
        r_sq = evaluate_r_squread(fitted, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nstatsmodels bayes gauss, RMSE={rmse} and r-squared={r_sq})')
    return imputer, fitted


def statsmodels_mice(masked, mask=None):
    fm1 = 'y ~ x1 + x2 + x3 + x4'  # this doesn't make any sense here as we don't know that
    imputer = sm.MICE(fm1, sm.OLS, sm.MICEData(masked))
    fitted = imputer.fit()
    if mask is not None:
        rmse, _, _ = evaluate_rmse(fitted, mask)
        r_sq = evaluate_r_squread(fitted, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nstatsmodels bayes gauss, RMSE={rmse} and r-squared={r_sq})')
    return imputer, fitted


# def ki(masked, mask=None):
#     fitted = KI(masked)
#     if mask is not None:
#         rmse, _, _ = evaluate_rmse(fitted, mask)
#         with open(RESULTS_FILE, "a") as f:
#             f.write(f'\nKI, {rmse})')
#     return fitted
#
#
# def fcki(masked, mask=None):
#     fitted = FCKI_cluster(masked)
#     if mask is not None:
#         rmse, _, _ = evaluate_rmse(fitted, mask)
#         with open(RESULTS_FILE, "a") as f:
#             f.write(f'\nFCKI, {rmse})')
#     return fitted


# def impyute_buck_iterative(masked, mask=None):
#     maskednp = masked.to_numpy()
#     fitted = pd.DataFrame(buck_iterative(maskednp))
#
#     fitted.columns = masked.columns
#     if mask is not None:
#         rmse, _, _ = evaluate_rmse(fitted, mask)
#         with open(RESULTS_FILE, "a") as f:
#             f.write(f'\nimpyute buck iterative imputer, {rmse})')
#     return fitted


def impyute_em(masked, mask=None):
    maskednp = masked.to_numpy()
    fitted = pd.DataFrame(em(maskednp))

    fitted.columns = masked.columns
    if mask is not None:
        rmse, _, _ = evaluate_rmse(fitted, mask)
        r_sq = evaluate_r_squread(fitted, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'impyute \nexpectation maximization imputer, RMSE={rmse} and r-squared={r_sq})')
    return fitted


def impyute_fast_knn(masked, mask=None):
    maskednp = masked.to_numpy()
    fitted = pd.DataFrame(fast_knn(maskednp))

    fitted.columns = masked.columns
    if mask is not None:
        rmse, _, _ = evaluate_rmse(fitted, mask)
        r_sq = evaluate_r_squread(fitted, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nimpyute knn imputer, RMSE={rmse} and r-squared={r_sq})')
    return fitted


def call_imputing_functions(masked, mask):
    # can't be used because it performs listwise deletion of samples before fitting the model to the true samples
    # however, as this dataset doesn't have any patients with all information observed, this method isn't applicable
    # creates and evaluates imputer; univariate strategies are meaningless, only predictive strategies are used
    # autoimpute_multiple_imputer(masked, mask)
    # autoimpute_mice(masked, mask)

    sklearn_simple_mean_imputer(masked, mask)
    sklearn_simple_median_imputer(masked, mask)
    sklearn_iterative_imputer(masked, mask, None)
    sklearn_knn_imputer(masked, mask)

    # # fancyimpute models have to be fitted and transformed on the same dataset
    fancyimpute_knn_imputer(masked, mask)
    fancyimpute_soft_impute(masked, mask)
    fancyimpute_iterative_svd(masked, mask)

    # can't be used; gives an error
    # missingpy_knn_imputer(masked, mask)
    # missingpy_missforest(masked, mask)

    # can't be used because it requires some users which data are fully observed in order to do PMM
    # similar cause as form autoimpute forms, but implementation is a bit different so errors produced are different
    # in addition, MICE requires knowledge of dependency between variables which we don't have
    # stats models have to be fitted and transformed on the same dataset
    # statsmodels_bayes_gauss_mi(masked, mask)
    # statsmodels_mice(masked, mask)

    # original code is very messy, had to update it
    # can't be used because it requires some users (at least 2) which data are fully observed
    # models have to be fitted and transformed on the same dataset
    # ki(masked, mask)
    # fcki(masked, mask)

    # original code is going into infinite loops, had to update it
    # impyute models have to be fitted and transformed on the same dataset
    # buck_iterative can't be imported from the module
    # impyute_buck_iterative(masked, mask)
    impyute_em(masked, mask)
    impyute_fast_knn(masked, mask)


def imputing_analysis():
    for i in range(3, 5):
        with open(RESULTS_FILE, "a") as f:
            f.write(f"\nExperiments {i}\n")
        # test different imputing models
        mask, masked = mask_data('clean_data_normalized.csv')
        call_imputing_functions(masked, mask)


def withdrawn_imputing_analysis():
    for i in range(5):
        with open(RESULTS_FILE, "a") as f:
            f.write(f"\n\nExperiment {i}\n\n")
            f.write('\nImputation of withdrawn.csv')
        mask, masked = mask_data('withdrawn.csv')
        call_imputing_functions(masked, mask)
    for i in range(5):
        train_x, train = get_x_and_full('train.csv')
        mask, masked = mask_data('withdrawn.csv')
        for j in range(len(estimators)):
            imputer_x, _, _ = sklearn_iterative_imputer(train_x, None, None, [estimators[j]])
            _, fitted, _ = sklearn_iterative_imputer(masked, None, imputer_x, [estimators[j]])
            rmse, _, _ = evaluate_rmse(fitted, mask)
            r_sq = evaluate_r_squread(fitted, mask)
            with open(RESULTS_FILE, "a") as f:
                f.write(f'\nrmse of {estimators[j]} imputation transfer to withdrawn is RMSE={rmse}, r-squared={r_sq}')
        # simple mean
        imputer = imp.SimpleImputer()
        imputer.fit(train_x)
        fitted_w = pd.DataFrame(imputer.transform(masked))
        fitted_w.columns = train_x.columns
        rmse, _, _ = evaluate_rmse(fitted_w, mask)
        r_sq = evaluate_r_squread(fitted_w, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nrmse of withdrawn when mean imputer is trained on train is RMSE={rmse}, r-squared={r_sq}')

        # simple median
        imputer = imp.SimpleImputer(strategy='median')
        imputer.fit(train_x)
        fitted_w = pd.DataFrame(imputer.transform(masked))
        fitted_w.columns = train_x.columns
        rmse, _, _ = evaluate_rmse(fitted_w, mask)
        r_sq = evaluate_r_squread(fitted_w, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nrmse of withdrawn when median imputer is trained on train is RMSE={rmse}, r-squuared={r_sq}')

        imputer = imp.KNNImputer(n_neighbors=5)
        imputer.fit(train_x)
        fitted_w = pd.DataFrame(imputer.transform(masked))
        fitted_w.columns = train_x.columns
        rmse, _, _ = evaluate_rmse(fitted_w, mask)
        r_sq = evaluate_r_squread(fitted_w, mask)
        with open(RESULTS_FILE, "a") as f:
            f.write(f'\nrmse of withdrawn when knn imputer is trained on train is RMSE={rmse}, r-squared={r_sq}')


def imputing_analysis_studies_separately():
    for i in range(5):
        mask, masked = mask_data('clean_data.csv')
        for study in studies_list:
            with open(RESULTS_FILE, "a") as f:
                f.write('\n' + study)
            smasked = masked[masked[study] == 1]
            cols_to_keep = smasked.any()[smasked.any() is True].index.tolist()
            smask = mask.loc[smasked.index, cols_to_keep].reset_index(drop=True)
            with open(RESULTS_FILE, "a") as f:
                f.write(f'\nPercent of columns covered {len(cols_to_keep) / len(mask.columns)} and percent of '
                  f'missingness covered {smask.notnull().sum().sum() / mask.notnull().sum().sum()}')
            smasked = smasked[cols_to_keep].reset_index(drop=True)
            call_imputing_functions(smasked, smask)


def get_x_and_full(filename):
    data = pd.read_csv(filename)
    X = data.drop(y_cols, axis=1)
    X = X.drop(index_cols, axis=1)
    return X, data


def impute(data_x, data, imputer_estimator, datatype, imputer_x=None, imputer_y=None, y_cols=y_cols):
    imputer_x, fitted_x, name = sklearn_iterative_imputer(data_x, None, imputer_x, [imputer_estimator])
    imputer_y, fitted_y, name = sklearn_iterative_imputer(data[y_cols], None, imputer_y, [imputer_estimator])
    fitted_x[y_cols] = fitted_y
    fitted_x[index_cols] = data[index_cols]
    fitted_x.to_csv(datatype + name + '_fitted.csv', index=False)
    return imputer_x, imputer_y


def impute_data_with_best_models():
    train_x, train = get_x_and_full('train.csv')
    cv_x, cv = get_x_and_full('cv.csv')
    test_x, test = get_x_and_full('test.csv')
    withdrawn_x, withdrawn = get_x_and_full('withdrawn.csv')
    for i in range(len(estimators)):
        imputer_x, imputer_y = impute(train_x, train, estimators[i], 'train_')
        impute(cv_x, cv, estimators[i], 'cv_', imputer_x, imputer_y)
        impute(test_x, test, estimators[i], 'test_', imputer_x, imputer_y)
        impute(withdrawn_x, withdrawn, estimators[i], 'withdrawn_', imputer_x, imputer_y)
    #     # I've tried to impute each study separately, but some studies had too little examples comparing to
    #     # features number, so when split in trainin:cv with ratio 85%:15% training has less examples than features
    #     # and algorithms can't be run


def impute_with_scikit_noniterative_imputer(imputer_x, imputer_y, data_x, data, filename):
    fitted = pd.DataFrame(imputer_x.transform(data_x))
    fitted.columns = data_x.columns
    fitted[y_cols] = pd.DataFrame(imputer_y.transform(data[y_cols]))
    fitted[index_cols] = data[index_cols]
    fitted.to_csv(filename, index=False)
    return fitted


def impute_with_fancy_imputer(imputer_x, imputer_y, data_x, data, filename):
    npdata_x = data_x.to_numpy()
    npdata_y = data[y_cols].to_numpy()
    fitted = pd.DataFrame(imputer_x.fit_transform(npdata_x))
    fitted.columns = data_x.columns
    fitted[y_cols] = pd.DataFrame(imputer_y.fit_transform(npdata_y))
    fitted[index_cols] = data[index_cols]
    fitted.to_csv(filename, index=False)
    return fitted


def impute_with_impyute_imputer(data_x, data, filename):
    npdata_x = data_x.to_numpy()
    npdata_y = data[y_cols].to_numpy()
    fitted = pd.DataFrame(em(npdata_x))
    fitted.columns = data_x.columns
    fitted[y_cols] = pd.DataFrame(em(npdata_y))
    fitted[index_cols] = data[index_cols]
    fitted.to_csv(filename, index=False)
    return fitted


def impute_with_noniterative():
    train_x, train = get_x_and_full('train.csv')
    cv_x, cv = get_x_and_full('cv.csv')
    test_x, test = get_x_and_full('test.csv')
    withdrawn_x, withdrawn = get_x_and_full('withdrawn.csv')

    # mean imputer
    imputer_x = imp.SimpleImputer()
    imputer_x.fit_transform(train_x)
    imputer_y = imp.SimpleImputer()
    imputer_y.fit(train[y_cols])
    impute_with_scikit_noniterative_imputer(imputer_x, imputer_y, train_x, train, 'train_mean_fitted.csv')
    impute_with_scikit_noniterative_imputer(imputer_x, imputer_y, cv_x, cv, 'cv_mean_fitted.csv')
    impute_with_scikit_noniterative_imputer(imputer_x, imputer_y, test_x, test, 'test_mean_fitted.csv')
    impute_with_scikit_noniterative_imputer(imputer_x, imputer_y, withdrawn_x, withdrawn, 'withdrawn_mean_fitted.csv')

    # median imputer
    imputer_x = imp.SimpleImputer(strategy='median')
    imputer_x.fit_transform(train_x)
    imputer_y = imp.SimpleImputer(strategy='median')
    imputer_y.fit(train[y_cols])
    impute_with_scikit_noniterative_imputer(imputer_x, imputer_y, train_x, train, 'train_median_fitted.csv')
    impute_with_scikit_noniterative_imputer(imputer_x, imputer_y, cv_x, cv, 'cv_median_fitted.csv')
    impute_with_scikit_noniterative_imputer(imputer_x, imputer_y, test_x, test, 'test_median_fitted.csv')
    impute_with_scikit_noniterative_imputer(imputer_x, imputer_y, withdrawn_x, withdrawn, 'withdrawn_median_fitted.csv')

    # knn imputer
    imputer_x = imp.KNNImputer(n_neighbors=5)
    imputer_x.fit_transform(train_x)
    imputer_y = imp.KNNImputer(n_neighbors=5)
    imputer_y.fit(train[y_cols])
    impute_with_scikit_noniterative_imputer(imputer_x, imputer_y, train_x, train, 'train_knn_fitted.csv')
    impute_with_scikit_noniterative_imputer(imputer_x, imputer_y, cv_x, cv, 'cv_knn_fitted.csv')
    impute_with_scikit_noniterative_imputer(imputer_x, imputer_y, test_x, test, 'test_knn_fitted.csv')
    impute_with_scikit_noniterative_imputer(imputer_x, imputer_y, withdrawn_x, withdrawn, 'withdrawn_knn_fitted.csv')

    # fancy soft imputer
    imputer_x = SoftImpute()
    imputer_y = SoftImpute()
    impute_with_fancy_imputer(imputer_x, imputer_y, train_x, train, 'train_softimpute_fitted.csv')
    impute_with_fancy_imputer(imputer_x, imputer_y, cv_x, cv, 'cv_softimpute_fitted.csv')
    impute_with_fancy_imputer(imputer_x, imputer_y, test_x, test, 'test_softimpute_fitted.csv')
    impute_with_fancy_imputer(imputer_x, imputer_y, withdrawn_x, withdrawn, 'withdrawn_softimpute_fitted.csv')

    # fancy svd imputer; doesn't work; dimensionality error for y part
    # imputer_x = IterativeSVD()
    # imputer_y = IterativeSVD()
    # impute_with_fancy_imputer(imputer_x, imputer_y, train_x, train, 'train_svd_fitted.csv')
    # impute_with_fancy_imputer(imputer_x, imputer_y, cv_x, cv, 'cv_svd_fitted.csv')
    # impute_with_fancy_imputer(imputer_x, imputer_y, test_x, test, 'test_svd_fitted.csv')
    # impute_with_fancy_imputer(imputer_x, imputer_y, withdrawn_x, withdrawn, 'withdrawn_svd_fitted.csv')

    # em imputer
    impute_with_impyute_imputer(train_x, train, 'train_em_fitted.csv')
    impute_with_impyute_imputer(cv_x, cv, 'cv_em_fitted.csv')
    impute_with_impyute_imputer(test_x, test, 'test_em_fitted.csv')
    impute_with_impyute_imputer(withdrawn_x, withdrawn, 'withdrawn_em_fitted.csv')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # imputing_analysis()
    # withdrawn_imputing_analysis()
    # imputing_analysis_studies_separately()
    # impute_data_with_best_models()
    # impute_with_noniterative()
