from sklearn.externals import joblib
from pandas import Series,DataFrame


def min_max_func(headers, db):
    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in range(0, len(headers)):
        db[headers[i]] = min_max_scaler.fit_transform(db[headers[i]].values.reshape(-1, 1))
    return db


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
    prediction = ((y_pred + 1)/2) * (7324 - 315) + 315
    print(y_pred)
    print(prediction)

    features_importance = clf.coef_[0]
    print(features_importance)
    return features_importance, prediction
    #draw_features_importance(clf)


