from sklearn.externals import joblib
from pandas import Series,DataFrame


def min_max_func(headers, db):
    #from sklearn.preprocessing import MinMaxScaler
    #min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    #consistent_gets
    db[headers[0]] = (float(db[headers[0]].values) - 47343)/(2047706 - 47343)
    #db[headers[1]] = (float(db[headers[1]].values) - 1) / (1 - 1)
    #execute_count
    db[headers[2]] = (float(db[headers[2]].values) - 758)/(127039 - 758)
    #parse_count_hard
    db[headers[3]] = (float(db[headers[3]].values) - 28)/(935 - 28)
    #parse_count_total
    db[headers[4]] = (float(db[headers[4]].values) - 92)/(44104 - 92)
    #session_logical_reads
    db[headers[5]] = (float(db[headers[5]].values) - 47754) / (2124772 - 47754)
    #total_sessions
    db[headers[6]] = (float(db[headers[6]].values) - 6539) / (7017 - 6539)
    #user_calls
    db[headers[7]] = (float(db[headers[7]].values) - 876) / (155700 - 876)
    #user_cpu
    db[headers[8]] = (float(db[headers[8]].values) - 281.166667) / (7161.333333 - 281.166667)

    '''
    for i in range(0, len(headers)):
        print(db[headers[i]])
        db[headers[i]] = min_max_scaler.fit_transform(db[headers[i]].values.reshape(-1, 1))
        print(db[headers[i]])
    '''
    return db

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
def prepare_data(data):
    objective = data['USER_CPU']
    # drop objective and time
    features = data
    features = data.drop('USER_CPU', 1)
    features = features.drop('DB_ID', 1)
    # features = data.drop('ACTIVE_SESSION_COUNT',1)
    # features = data.drop('TIME', 1)
    return features, objective


def svm_test(test_data):
    # find max_min_value


    #Dataframe transfer
    test_data = DataFrame(test_data)
    # print(max, min)
    # scale all data
    test_data = min_max_func(test_data.columns, test_data)
    print(test_data)

    #do stuff

    X_test, y_test = prepare_data(test_data)
    # print('bbb')

    # scale
    # train_X = min_max_func(train_X.columns, train_X)
    # test_X = min_max_func(test_X.columns, test_X)
    # print(features_scaled.values)

    # regression model
    clf = joblib.load('svr.pkl')
    #clf = SVR(C=1.0, epsilon=0.2, kernel="linear")

    #clf.fit(X_train, y_train)

    #save model
    #joblib.dump(clf, 'svr.pkl')
    # predict

    y_pred = clf.predict(X_test)
    #print(y_pred)
    # restore predict
    prediction = ((y_pred + 1)/2) * (7161.333333 - 281.166667) + 281.166667
    print(y_pred)
    print(prediction)

    features_importance = clf.coef_[0]
    print(features_importance)
    features_intercept = clf.intercept_
    print(features_intercept)
    return features_importance, prediction
    #draw_features_importance(clf)


