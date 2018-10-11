import inline as inline
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os.path as osp
import sys 
import seaborn as sns
import argparse
import pickle
from pandas import Series,DataFrame
from utils import convert_var_by_day
from regression import svm_regression
from data import get_data_headers

data_dir = ""
'''
data_path = 'data_normal_all.pkl'
db_data = pd.read_pickle(data_path)
'''
headers = get_data_headers()

db_data = pd.read_csv(osp.join(data_dir, 'data_all.csv'))

db_data['TIME'] = db_data['TIME'].apply(lambda x: pd.to_datetime(x))
db_data['TIME'] = db_data['TIME'].astype('datetime64[ns]')

db_data = db_data.sort_values(['TIME'], ascending=[True])

#ids = db_data['DB_ID'].unique()

#headers,ids
db_id = db_data['DB_ID']

train_data = None
for i in range(len(headers)):
    head = headers[i]
    pandas_data = convert_var_by_day(db_data, head['name'], db_id, show_gb=head['show'], aggr=head['aggr'], draw=True, day=0)
    pandas_data.columns = ['TIME', head]
    print(pandas_data)
    if train_data is None:
        train_data = pandas_data
    else:
        train_data = pd.concat([train_data, pandas_data], axis=1)
'''
db_id = db_data['DB_ID']
test_data = None
for i in range(len(headers)):
    head = headers[i]
    pandas_data = convert_var_by_day(db_data, head['name'], db_id, show_gb=head['show'], aggr=head['aggr'], draw=True, day=0)
    pandas_data.columns = ['TIME', head]
    print(pandas_data)
    if train_data is None:
        test_data = pandas_data
    else:
        test_data = pd.concat([test_data, pandas_data], axis=1)
print(type(test_data))
'''
data = {'DB_ID': ['1'], 'EXECUTE_COUNT': ['1904'], 'SESSION_LOGICAL_READS': ['93691'],
        'USER_CPU': ['468.3333333'], 'TOTAL_SESSIONS': ['6635'], 'USER_CALLS': ['22022'], 'CONSISTENT_GETS': ['92742'],
        'PARSE_COUNT_TOTAL': ['525'], 'PARSE_COUNT_HARD': ['46']}
test_data = DataFrame(data)
svm_regression(train_data, test_data)


