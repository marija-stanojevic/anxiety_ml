import numpy as np
import pandas as pd
import re

txconds_list = ['txcond_0', 'txcond_1', 'txcond_2', 'txcond_3', 'txcond_4', 'txcond_6', 'txcond_7', 'txcond_8']
studies_list = ['studycode_1', 'studycode_2', 'studycode_3', 'studycode_4', 'studycode_5', 'studycode_6', 'studycode_7',
                'studycode_8', 'studycode_9', 'studycode_10']


# Set datatype to 'new' if it's from Lesley's study
def clean_data(full_data, codebook, datatype=''):
    data = full_data[full_data.columns.intersection(codebook)].copy()
    data['pre_compcsr_sp_dsm4'] = -1
    data['parent1_sp'] = -1
    data['parent2_sp'] = -1
    if datatype == 'new_':
        studycodes = pd.DataFrame(0, index=np.arange(len(data)), columns=studies_list)
        studycodes['studycode_10'] = 1
        txconds = pd.DataFrame(0, index=np.arange(len(data)), columns=txconds_list)
        txconds['txcond_1'] = 1
    else:
        studycodes = pd.get_dummies(data.loc[:, 'studycode'])
        studycodes['studycode_10'] = 0
        txconds = pd.get_dummies(data.loc[:, 'txcond'])

    data[studies_list] = studycodes
    data[txconds_list] = txconds
    data.loc[(data['childracecomb'] > 2), 'childracecomb'] = 3  # 1 - white; 2 - black; 3 - other

    for col in data.columns:  # if composite not given, replace with max of parent and child feature
        if col == 'pre_compcsr_pdd_dsm4':
            col1 = col.replace('compcsr', 'pcsr')
            for index, row in data.iterrows():
                if data.loc[index, col] == -1 and col1 in full_data.columns:
                    data.loc[index, col] = full_data.loc[index, col1]
        elif col == 'pre_compcsr_sp_dsm4':
            for index, row in data.iterrows():
                col1, col2, col3 = 'pre_ccsr_sp1_dsm4', 'pre_ccsr_sp2_dsm4', 'pre_ccsr_sp3_dsm4'
                col4, col5, col6 = 'pre_ccsr_sp4_dsm4', 'pre_ccsr_sp5_dsm4', 'pre_pcsr_sp1_dsm4'
                col7, col8, col9 = 'pre_pcsr_sp2_dsm4', 'pre_pcsr_sp3_dsm4', 'pre_pcsr_sp4_dsm4'
                col10, col11, col12 = 'pre_pcsr_sp5_dsm4', 'pre_compcsr_sp1_dsm4', 'pre_compcsr_sp2_dsm4'
                col13, col14 = 'pre_compcsr_sp3_dsm4', 'pre_compcsr_sp4_dsm4'
                if col1 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col1], data.loc[index, col])
                if col2 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col2], data.loc[index, col])
                if col3 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col3], data.loc[index, col])
                if col4 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col4], data.loc[index, col])
                if col5 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col5], data.loc[index, col])
                if col6 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col6], data.loc[index, col])
                if col7 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col7], data.loc[index, col])
                if col8 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col8], data.loc[index, col])
                if col9 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col9], data.loc[index, col])
                if col10 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col10], data.loc[index, col])
                if col11 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col11], data.loc[index, col])
                if col12 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col12], data.loc[index, col])
                if col13 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col13], data.loc[index, col])
                if col14 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col14], data.loc[index, col])
        elif col == 'post_compcsr_sep_dsm4':
            col1, col2, col3 = 'post_ccsr_sep_dsm4', 'post_pcsr_sep_dsm4', 'post_compcsr_sep_dsm4'
            for index, row in data.iterrows():
                if col1 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col1], data.loc[index, col])
                if col2 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col2], data.loc[index, col])
                if col3 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col3], data.loc[index, col])
        elif col == 'post_compcsr_soc_dsm4':
            col1, col2, col3 = 'post_ccsr_soc_dsm4', 'post_pcsr_soc_dsm4', 'post_compcsr_soc_dsm4'
            for index, row in data.iterrows():
                if col1 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col1], data.loc[index, col])
                if col2 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col2], data.loc[index, col])
                if col3 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col3], data.loc[index, col])
        elif col == 'post_compcsr_gad_dsm4':
            col1, col2, col3 = 'post_ccsr_gad_dsm4', 'post_pcsr_gad_dsm4', 'post_compcsr_gad_dsm4'
            for index, row in data.iterrows():
                if col1 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col1], data.loc[index, col])
                if col2 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col2], data.loc[index, col])
                if col3 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col3], data.loc[index, col])
        elif col == 'parent1_sp':
            col1, col2, col3 = 'parent1_sp1', 'parent1_sp2', 'parent1_sp3'
            for index, row in data.iterrows():
                if col1 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col1], data.loc[index, col])
                if col2 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col2], data.loc[index, col])
                if col3 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col3], data.loc[index, col])
        elif col == 'parent2_sp':
            col1, col2, col3 = 'parent2_sp1', 'parent2_sp2', 'parent2_sp3'
            for index, row in data.iterrows():
                if col1 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col1], data.loc[index, col])
                if col2 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col2], data.loc[index, col])
                if col3 in full_data.columns:
                    data.loc[index, col] = max(full_data.loc[index, col3], data.loc[index, col])
        elif col != 'pre_compcsr_tour_dsm4' and re.match('pre_compcsr_.', col):
            col1 = col.replace('compcsr', 'pcsr')
            col2 = col.replace('compcsr', 'ccsr')
            for index, row in data.iterrows():
                if data.loc[index, col] == -1:
                    if col1 in full_data.columns and col2 in full_data.columns:
                        data.loc[index, col] = max(full_data.loc[index, col1], full_data.loc[index, col2])
                    elif col1 in full_data.columns:
                        data.loc[index, col] = full_data.loc[index, col1]
                    elif col2 in full_data.columns:
                        data.loc[index, col] = full_data.loc[index, col2]

    data = data.replace(-1, None)
    data[['childracecomb_1', 'childracecomb_2', 'childracecomb_3']] = pd.get_dummies(data.loc[:, 'childracecomb'])
    data = data.drop(['studycode', 'txcond', 'childracecomb'], axis=1)
    data = data.drop(['p1educ2', 'p2educ2', 'pre_compcsr_slpterr_dsm4', 'pre_compcsr_subabuse_dsm4',
                      'pre_compcsr_bp_dsm4', 'pre_compcsr_sz_dsm4', 'pre_compcsr_eatingdo_dsm4', 'pre_compcsr_pdd_dsm4',
                      'pre_compcsr_mddpast_dsm4', 'pre_compcsr_dysthymiapast_dsm4', 'pre_compcsr_tour_dsm4'], axis=1)

    if datatype == '':
        data = data.drop(['txbinary'], axis=1)
        not_finished = data[data['txwithdraw'] != 0].drop(['txwithdraw'], axis=1)
        data = data[data['txwithdraw'] == 0].drop(['txwithdraw'], axis=1)
        not_finished.to_csv('withdrawn_pat_data.csv')
    data.to_csv(f'clean_{datatype}data.csv')


if __name__ == '__main__':
    codebook = pd.read_csv('codebook.csv').iloc[:, 0].tolist()

    # data = pd.read_csv('dissertationdata_allmerged_5.16.csv')
    # data = data.replace(888, -1)
    # data = data.replace(999, -1)
    # data.loc[(data['txcond'] == 5), 'txcond'] = 0
    # data.to_csv('data.csv', index=False, header=True)
    # clean_data(data, codebook)

    new_data = pd.read_csv('ExternalValidationSet_Final_2.20.csv')
    new_data = new_data.drop(['DEMOGRAPHIC_INFO', 'CHILD_DIAGNOSTIC_INFO_DSM4', 'PARENT_DIAGNOSTIC_INFO'], axis=1)
    new_data = new_data.replace(888, -1)
    new_data = new_data.replace(999, -1)
    new_data.to_csv('new_data.csv', index=False, header=True)
    clean_data(new_data, codebook, 'new_')
