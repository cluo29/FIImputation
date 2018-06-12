# FIMDI.py
# Chu Luo

# using feature importance for MDI

from numpy import genfromtxt
import numpy as np

def run_FIMDI(X_train, X_test_m, Y_train, Y_test, column_id, label, X_test_c):
    # X_test_m  has missing data already
    # column_id is the feature missing data
    # label is how missing data is marked
    # X_test_c is complete X test set
    print()

def make_random_missing(column_id, label, X_test_c, rate_of_missing=0.1):
    # to make a test set miss some data in a feature
    rows = len(X_test_c)
    # so 0, ... ,  rows-1 are good to choose
    # we choose rate * len ones
    # a = np.random.choice(5, 4, replace=False)
    # will get 0-4
    # unique ones! by using False
    outputSet = X_test_c

    missing_count = int(np.floor(rate_of_missing * rows))

    print('missing_count = ', missing_count)

    missing_rows = np.random.choice(rows, missing_count, replace=False)

    # do want 0st row to miss
    if 0 in missing_rows:
        b = np.array([0])
        missing_rows = np.setdiff1d(missing_rows, b)


    for i in missing_rows:

            outputSet[i,column_id] = label

    return outputSet


def read_and_run(train_file, test_file, column_id, label=-1, seed=0):
    # read train and test file from csv

    np.random.seed(seed)

    train_set = genfromtxt(train_file, delimiter=',')

    train_col = train_set.shape[1]

    X_train = np.delete(train_set, train_col-1, 1)

    Y_train = train_set[:,-1]

    Y_train = Y_train.reshape(Y_train.shape[0], -1)

    test_set = genfromtxt(test_file, delimiter=',')

    X_test_c = np.delete(test_set, train_col-1, 1)

    Y_test = test_set[:,-1]

    Y_test = Y_test.reshape(Y_test.shape[0], -1)

    print(X_test_c)
    #print(Y_test)

    # now make missing data
    X_test_m = make_random_missing(column_id, label, X_test_c, 0.5)

    print(X_test_m)



# test code

read_and_run('A.csv','B.csv', 0)
