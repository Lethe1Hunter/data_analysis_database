#used variables
def get_headers():
    headers = []
    return headers

def get_data_headers():
    headers = []
    headers.append({'name': 'DB_ID', 'gb': False, 'show': False, 'aggr': 'sum'})
    #headers.append({'name': 'ACTIVE_SESSION_COUNT', 'gb': False, 'show': False, 'aggr': 'sum'})
    headers.append({'name': 'EXECUTE_COUNT', 'gb': False, 'show': False, 'aggr': 'sum'})
    headers.append({'name': 'SESSION_LOGICAL_READS', 'gb': False, 'show': False, 'aggr': 'sum'})
    headers.append({'name': 'USER_CPU', 'gb': False, 'show': False, 'aggr': 'sum'})
    #headers.append({'name': 'SYS_CPU', 'gb': False, 'show': False, 'aggr': 'sum'})
    #headers.append({'name': 'IDLE_CPU', 'gb': False, 'show': False, 'aggr': 'sum'})
    #headers.append({'name': 'IOWAIT_CPU', 'gb': False, 'show': False, 'aggr': 'sum'})
    #headers.append({'name': 'REDO_SIZE', 'gb': False, 'show': False, 'aggr': 'sum'})
    headers.append({'name': 'TOTAL_SESSIONS', 'gb': False, 'show': False, 'aggr': 'sum'})
    headers.append({'name': 'USER_CALLS', 'gb': False, 'show': False, 'aggr': 'sum'})
    headers.append({'name': 'CONSISTENT_GETS', 'gb': False, 'show': False, 'aggr': 'sum'})
    headers.append({'name': 'PARSE_COUNT_TOTAL', 'gb': False, 'show': False, 'aggr': 'sum'})
    #headers.append({'name': 'FREE_MEM_SIZE', 'gb': False, 'show': False, 'aggr': 'sum'})
    headers.append({'name': 'PARSE_COUNT_HARD', 'gb': False, 'show': False, 'aggr': 'sum'})
    return headers