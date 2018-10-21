from pandas import Series,DataFrame
from load_model import svm_test


data = {'DB_ID': ['1'], 'EXECUTE_COUNT': ['1904'], 'SESSION_LOGICAL_READS': ['93691'],
        'USER_CPU': ['468.3333333'], 'TOTAL_SESSIONS': ['6635'], 'USER_CALLS': ['22022'], 'CONSISTENT_GETS': ['92742'],
        'PARSE_COUNT_TOTAL': ['525'], 'PARSE_COUNT_HARD': ['46']}
print(type(data))
svm_test(data)
