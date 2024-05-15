# This script is used to analyse the results obtained from validate_results.py.
import pandas as pd


def get_stats_df(df, base_name='objective', group_columns=['method', 'train pool size']):
    df_means = df.groupby(group_columns).mean().reset_index()
    for column_name in df_means.columns:
        if column_name.__contains__(base_name):
            df_means = df_means.rename(columns={column_name: column_name.replace(base_name, f'{base_name} mean')})

    df_sem = df.groupby(group_columns).sem().reset_index()
    for column_name in df_sem.columns:
        if column_name.__contains__(base_name):
            df_sem = df_sem.rename(columns={column_name: column_name.replace(base_name, f'{base_name} sem')})

    df_min = df.groupby(group_columns).min().reset_index()
    for column_name in df_min.columns:
        if column_name.__contains__(base_name):
            df_min = df_min.rename(columns={column_name: column_name.replace(base_name, f'{base_name} min')})

    first_merge_df = df_means.merge(df_sem, on=group_columns, how='inner')
    full_df = first_merge_df.merge(df_min, on=group_columns, how='inner')

    return full_df


problem = 'ESPU'
relative_to_optimal = True
validations_directory = 'Data/Output/Validations'
analyses_directory = 'Data/Output/Analyses'
validation_name = f'{problem}.csv'
stats_columns = ['budget', 'method', 'instance size', 'pool sampling', 'train pool size',
                 'sampling setting']#, 'instance number']
analysis_columns = ['budget', 'instance size', 'method', 'pool sampling', 'train pool size',
                 'sampling setting', 'train objective mean',
                 'test objective mean', 'test objective sem', 'test vs train objective mean',
                 'elapsed time']#, 'instance number']
df = pd.read_csv(f'{validations_directory}/{validation_name}')
df['train objective'] = df['objective']
if relative_to_optimal:
    if problem == 'ESPU':
        det_optimal = pd.read_csv('Data/Input/ESPU/Objectives_deterministic.csv', index_col=0)
        for instance_size in det_optimal.index:
            optimal = det_optimal.loc[instance_size, 'deterministic optimal']
            df.loc[df['instance size'] == instance_size, 'train objective'] = \
                df.loc[df['instance size'] == instance_size, 'train objective'] / optimal - 1
            df.loc[df['instance size'] == instance_size, 'test objective'] = \
                df.loc[df['instance size'] == instance_size, 'test objective'] / optimal - 1
    elif problem == 'SRCPSP':
        df['optimal'] = 0
        numbers_optimal = {1: 43, 2: 47, 3: 47, 4: 62, 5: 39}
        for number, optimal_value in numbers_optimal.items():
            number_bool = df['instance number'] == number
            df.loc[number_bool, 'train objective'] = df.loc[number_bool, 'train objective'] / optimal_value - 1
            df.loc[number_bool, 'test objective'] = df.loc[number_bool, 'test objective'] / optimal_value - 1
            df.loc[number_bool, 'optimal'] = optimal_value
    else:
        df['optimal'] = 0
        ids_optimal = {'R101': 1637.7, 'R102': 1466.6, 'R103': 1208.7, 'R104': 971.5, 'R105': 1355.3}
        for id, optimal_value in ids_optimal.items():
            id_bool = df['instance id'] == id
            df.loc[id_bool, 'train objective'] = df.loc[id_bool, 'train objective'] / optimal_value - 1
            df.loc[id_bool, 'test objective'] = df.loc[id_bool, 'test objective'] / optimal_value - 1
            df.loc[id_bool, 'optimal'] = optimal_value
df['sampling setting'] = ''
df.loc[df['method'] == 'seqpre', 'sampling setting'] = [i.split('_')[4].split('=')[1] for i in df['method settings'] if not isinstance(i, float) and 'dc' in i]
df['test vs train objective'] = (df['test objective'] - df['train objective']) / df['train objective']
stats = get_stats_df(df, group_columns=stats_columns)
analysis = stats[analysis_columns]
analysis_name = f'{validation_name}'
analysis.to_csv(f'{analyses_directory}/{analysis_name}')
