# This script is used to produce a table given the results obtained by analyse_results.py.
import pandas as pd

take_average = True
problem = 'ESPU'

if problem != 'SRCPSP':
    df = pd.read_csv(f'Data/Output/Analyses/{problem}.csv', index_col=0)
    df['sampling setting'] = df['sampling setting'].astype(str)
    df['sampling setting'] = df['sampling setting'].replace({'updates': 'DU', 'matrix completion': 'MC', 'nan': ''})
    df['method'] = df['method'].replace({'seqpre': 'SeqPre', 'deterministic': 'Det', 'seqdif': 'SeqDif'})
    df['pool sampling'] = df['pool sampling'].replace({'descriptive': 'Des', 'random': 'Ran', 'average': 'Det'})
    seqpre = (df['method'] == 'SeqPre')
    seqdif = (df['method'] == 'SeqDif')
    des = (df['method'] == 'Det') & (df['pool sampling'] == 'Des')
    avg = (df['method'] == 'Det') & (df['pool sampling'] == 'Det')
    df['approach'] = 0
    df.loc[seqpre, 'approach'] = df.loc[seqpre, 'method'] + df.loc[seqpre, 'sampling setting'].astype(str)
    df.loc[seqdif, 'approach'] = df.loc[seqdif, 'method']
    df.loc[des, 'approach'] = df.loc[des, 'pool sampling'] + df.loc[des, 'train pool size'].astype(str)
    df.loc[avg, 'approach'] = df.loc[avg, 'pool sampling']
    df = df.rename(columns={'train objective mean': 'train', 'test objective mean': 'test',
                            'test vs train objective mean': 'test/train', 'elapsed time': 'time',
                            'train pool size': 'pool size', 'pool sampling': 'sampling',
                            'test objective sem': 'sem'})
    if problem == 'SRCPSP':
        df = df.rename(columns={'instance': 'old instance', 'instance number': 'instance'})
    else:
        df = df.rename(columns={'instance size': 'instance'})
    if take_average:
        df = df.groupby(['budget', 'approach']).mean().reset_index()
        output_columns = ['budget', 'approach', 'test', 'sem', 'train', 'time']
        pivot_list = ['approach']
    else:
        output_columns = ['budget', 'instance', 'approach', 'test', 'sem', 'train', 'time']
        pivot_list = ['instance', 'approach']
    df = df[(df['approach'] != 0) & (df['approach'] != 'Des1')]
    df['test'] = df['test'] * 100
    df['train'] = df['train'] * 100
    df['sem'] = df['sem'] * 100
    df['test'] = df['test'].round(1).astype(str) + ' $\\pm$'
    df['sem'] = df['sem'].round(1).astype(str)
    df['train'] = df['train'].round(1).astype(str)
    df = df[df['budget'] < 10000000]

    table = df[output_columns].copy()
    table = table.round({'time': 0})
    table = table.pivot(index=pivot_list, columns=['budget'])
    table.columns = table.columns.reorder_levels([1, 0])
    table = table.reindex(columns=table.columns.sort_values(0))
    table = table.reindex(columns=table.columns.reindex(['test', 'sem', 'train', 'time'], level=1)[0])
    if not take_average:
        table = table.reindex(table.index.reindex(['Det', 'Des10',
                                                   'Des100', 'SeqDif', 'SeqPreDU', 'SeqPreMC'], level=1)[0])
    else:
        table = table.reindex(table.index.reindex(['Det', 'Des10', 'Des100', 'SeqDif', 'SeqPreDU'])[0])
    table.to_csv(f'Data/Output/Tables/{problem}.csv', float_format='%g')
else:
    df = pd.read_csv('Data/Output/Analyses/SRCPSP.csv', index_col=0)
    df['sampling setting'] = df['sampling setting'].replace({'updates': 'DU', 'matrix completion': 'MC', 'nan': ''})
    df['method'] = df['method'].replace({'seqpre': 'SeqPre', 'deterministic': 'Det', 'seqdif': 'SeqDif'})
    df['pool sampling'] = df['pool sampling'].replace({'descriptive': 'Des', 'random': 'Ran'})
    seqpre = (df['method'] == 'SeqPre')
    seqdif = (df['method'] == 'SeqDif')
    det = (df['method'] == 'Det')
    df['approach'] = 0
    df.loc[seqpre, 'approach'] = df.loc[seqpre, 'method'] + '' + df.loc[seqpre, 'sampling setting'].astype(str)
    df.loc[seqdif, 'approach'] = df.loc[seqdif, 'method']
    df.loc[det, 'approach'] = df.loc[det, 'method'] + '' + df.loc[det, 'pool sampling'] + '' \
                              + df.loc[det, 'train pool size'].astype(str)
    df = df.rename(columns={'train objective mean': 'train', 'test objective mean': 'test',
                            'test vs train objective mean': 'test/train', 'elapsed time': 'time',
                            'instance size': 'instance', 'train pool size': 'pool size', 'pool sampling': 'sampling',
                            'test objective sem': 'sem'})
    df = df.sort_values(['budget'])
    output_columns = ['budget', 'approach', 'test', 'sem', 'train', 'time']

    table = df[output_columns].copy()
    table = table.round({'train': 1, 'test': 1, 'sem': 1,
                         'test/train': 0, 'time': 0})
    table = table.pivot(index=['approach'], columns=['budget'])
    table.columns = table.columns.reorder_levels([1, 0])
    table = table.reindex(columns=table.columns.sort_values(0))
    table = table.reindex(columns=table.columns.reindex(['test', 'sem', 'train', 'time'], level=1)[0])
    table.to_csv('Data/Output/Tables/SRCPSP.csv', float_format='%g')
