import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path as osp
import seaborn as sns

#scale features to the range [-1, 1]
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR


def min_max_func(headers, db):
    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in range(0, len(headers)):
        db[headers[i]] = min_max_scaler.fit_transform(db[headers[i]].values.reshape(-1, 1))
    return db


def max_restore(headers, db):
    for i in range(0, len(headers)):
        max_value = max(db[headers[i]])
    return max_value

def min_restore(headers, db):
    for i in range(0, len(headers)):
        min_value = min(db[headers[i]])
    return min_value

def prepare_data(data):
    objective = data['USER_CPU']
    # drop objective and time
    features = data
    features = data.drop('USER_CPU', 1)
    features = features.drop('DB_ID', 1)
    # features = data.drop('ACTIVE_SESSION_COUNT',1)
    # features = data.drop('TIME', 1)
    return features, objective


def svm_regression(train_data):
    # find max_min_value
    max = max_restore(train_data.columns, train_data)
    min = min_restore(train_data.columns, train_data)

    print(max, min)
    # scale all data
    train_data = min_max_func(train_data.columns, train_data)
    #test_data = min_max_func(test_data.columns, test_data)


    #do stuff
    X_train, y_train = prepare_data(train_data)
    #X_test, y_test = prepare_data(test_data)
    # print('bbb')

    # scale
    # train_X = min_max_func(train_X.columns, train_X)
    # test_X = min_max_func(test_X.columns, test_X)
    # print(features_scaled.values)

    # regression model
    clf = SVR(C=1.0, epsilon=0.2, kernel="linear")

    clf.fit(X_train, y_train)

    #save model
    joblib.dump(clf, 'svr.pkl')
    # predict

    #y_pred = clf.predict(X_test)
    #print(y_pred)
    # restore predict
    #prediction = ((y_pred + 1)/2) * (max - min) + min
    #print(prediction)

    # calculate score
    #print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
    #draw_pred(y_test, y_pred, X_test)
    #draw_features_importance(clf, X_test)


def draw_pred(y_true, y_pred, X_test):
    x_ticks = np.arange(0, len(X_test))
    f, ax = plt.subplots(figsize=(16, 4))
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1.0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)
    # ax.set(xlabel='{}-{}-{}. Hour'.format(times.year[0], times.month[0], days[i]), ylabel='%')
    # print(x_ticks)
    ax.set(xticks=x_ticks)
    # print(len(y_true), len(x_ticks))
    sns.tsplot(y_true, ax=ax, time=x_ticks, condition="ground-truth")
    sns.tsplot(y_pred, ax=ax, time=x_ticks, color="m", legend="Prediction", condition="svm")
    # plt.savefig(osp.join(var + '_' + freq + '_' + aggr + '_%d.pdf'%i))
    plt.show()
    plt.close()


def draw_features_importance(estimator, features):
    # Plot feature importance
    plt.figure(figsize=(20, 10))
    feature_importance = estimator.coef_[0]
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    features_names = features.columns.values
    print(feature_importance)
    print(sorted_idx)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, features_names[sorted_idx], fontsize=15, family='SimHei')
    plt.xlabel('Relative Importance ')
    plt.title('Variable Importance')
    plt.show()
    # plt.savefig("imp.png", dpi = 500)