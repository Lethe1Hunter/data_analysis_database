from pandas import Series,DataFrame
from load_model import svm_test


#data = {'DB_ID': ['1'], 'EXECUTE_COUNT': ['1904'], 'SESSION_LOGICAL_READS': ['93691'],
#       'USER_CPU': ['468.3333333'], 'TOTAL_SESSIONS': ['6635'], 'USER_CALLS': ['22022'], 'CONSISTENT_GETS': ['92742'],
#       'PARSE_COUNT_TOTAL': ['525'], 'PARSE_COUNT_HARD': ['46']}
data = {'DB_ID': ['1'], 'EXECUTE_COUNT': ['69568'], 'SESSION_LOGICAL_READS': ['1476653'],
       'USER_CPU': ['5309.666667'], 'TOTAL_SESSIONS': ['6782'], 'USER_CALLS': ['112764'], 'CONSISTENT_GETS': ['1402533'],
       'PARSE_COUNT_TOTAL': ['32256'], 'PARSE_COUNT_HARD': ['636']}
print(type(data))
svm_test(data)
'''
cpu max = 7161.333333      min = 281.166667
total_sessions max = 7017       min = 6539
session_logical_reads max = 2124772     min = 47754
consistent_gets max = 2047706   min = 47343
parse_count_total max = 44104   min = 92
parse_count_hard max = 935      min = 28
execute_count  max = 127039     min = 758
user_call max = 155700  min = 876
'''