import pandas as pd
import matplotlib.pyplot as plt
from cleaning import studies_list, txconds_list
from imputing import index_cols, y_cols
import seaborn as sns
from modeling import RESULTS_FILE


def split_dataset():
    data = pd.read_csv('clean_data.csv').sample(frac=1)
    patients = data.shape[0]
    per80 = int(patients * 0.8)
    per90 = int(patients * 0.9)
    train = data[:per80].copy()
    validation = data[per80:per90].copy()
    test = data[per90:].copy()
    return train, validation, test


def missingness(data, prefix):
    length = data.shape[0]
    data = data.drop(['Unnamed: 0'], axis=1)
    row_miss = {}
    col_miss = {}
    for col in data.columns:
        try:
            col_miss[col] = data[col].isna().sum() / length * 100
            if prefix == 'train_':
                with open(RESULTS_FILE, "a") as f:
                    f.write(f'\n{col}, {int(col_miss[col])}%')
        except KeyError:
            with open(RESULTS_FILE, "a") as f:
                f.write(f'\nFeature with all data: {col} \n')
    for i, row in data.iterrows():
        try:
            row_miss[i] = row.isna().sum() / data.shape[1] * 100
        except KeyError:
            with open(RESULTS_FILE, "a") as f:
                f.write(f'\nPatient with all data: {i} \n')
    studies_sizes = [49, 99, 138, 422, 137, 38, 114, 76, 88]
    x = [10, 20, 30, 40, 50, 60, 70, 80]
    plt.plot(row_miss.keys(), row_miss.values(), 'ro')
    for s in range(1, len(studies_sizes)):
        y = sum(studies_sizes[:s])
        y = [y] * 8
        plt.plot(y, x, linestyle='dashed')
    plt.xlabel('Patients')
    plt.ylabel('Percent of features missing')
    plt.title('Percent of features missing for each patient')
    plt.savefig(prefix + 'patients_missingness.png')
    plt.clf()
    plt.plot(range(len(col_miss)), col_miss.values(), 'ro')
    plt.xlabel('Features (names omitted for clarity)')
    plt.ylabel('Percent of patients missing the feature')
    plt.title('Percent of patients missing the feature')
    plt.savefig(prefix + 'features_missingness.png')
    plt.clf()


def normalize(data):
    normalized = pd.DataFrame()
    for col in data.columns:
        if col != 'Unnamed: 0' and col not in y_cols and data[col].std() != 0:
            normalized[col] = (data[col] - data[col].mean()) / data[col].std()
        elif col == 'txsessions':
            normalized[col] = 1
        else:
            normalized[col] = data[col]
    return normalized


def percent_of_missing(data, data_name):
    per = data.isnull().sum().sum()/(data.shape[0] * data.shape[1])
    with open(RESULTS_FILE, "a") as f:
        f.write(f'\nMissingness for {data_name} is {per}')
    return per


def calculate_missing():
    data = pd.read_csv('clean_data.csv')
    percent_of_missing(data.drop(studies_list, axis=1), 'clean_data.csv')
    for study in studies_list:
        st_data = data[data[study] == 1].drop(studies_list, axis=1)
        percent_of_missing(st_data, study)


def plot_missingness_matrix():
    data = pd.read_csv('clean_data.csv')
    data = data.drop(studies_list + txconds_list + index_cols, axis=1)
    plt.figure(figsize=(20, 15))
    sns.heatmap(data[y_cols].isnull(), cbar=False, cmap="RdYlGn", center=0.5)
    plt.xlabel('Features', fontsize=30)
    plt.ylabel('Patients', fontsize=30)
    plt.title('Missingness heatmap of labels (green are missing)\n', fontsize=40)
    plt.savefig('y_missingness_heatmap.png')
    plt.clf()
    data = data.drop(y_cols, axis=1)
    plt.figure(figsize=(50, 33))
    sns.heatmap(data.isnull(), cbar=False, cmap="RdBu", center=0.5)
    plt.xlabel('Features', fontsize=70)
    plt.ylabel('Patients', fontsize=70)
    plt.title('Missingness heatmap of explanatory variables (blue are missing)\n', fontsize=86)
    plt.savefig('x_missingness_heatmap.png')
    plt.clf()


if __name__ == '__main__':
    # data = pd.read_csv('clean_data.csv')
    # data = normalize(data)
    # data.to_csv('clean_data_normalized.csv', index=False)
    #
    # train, cv, test = split_dataset()
    #
    # missingness(train, 'train_')
    # train = normalize(train)
    # train.to_csv('train.csv', index=False)
    #
    # missingness(cv, 'cv_')
    # cv = normalize(cv)
    # cv.to_csv('cv.csv', index=False)
    #
    # missingness(test, 'test_')
    # test = normalize(test)
    # test.to_csv('test.csv', index=False)
    #
    # calculate_missing()
    #
    # plot_missingness_matrix()
    #
    # withdrawn = pd.read_csv('withdrawn_pat_data.csv')
    # withdrawn = normalize(withdrawn)
    # withdrawn.to_csv('withdrawn.csv', index=False)

    new_data = pd.read_csv('clean_new_data.csv')
    new_data = normalize(new_data)
    new_data.to_csv('lesley.csv', index=False)
