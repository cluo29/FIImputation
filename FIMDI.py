# FIMDI.py
# Chu Luo

# using feature importance for MDI

from numpy import genfromtxt
import numpy as np
import feature_impact
import MDIOnline

# our feature impact impute, actually it is an impute evaluator


class FIMDImputer():

    # initialization at training time
    def __init__(self, X_train, Y_train, column_id, label):
        # 1. train a FI_detector using train set
        # get how many classes are in Y
        setAllY = set(Y_train)
        num_class = len(setAllY)

        self.detector = feature_impact.Detector_C(num_class)

        self.detector.train(X_train, Y_train, 0)

        # 2. train all the imputation models

        self.mdi = MDIOnline.MDImputer(X_train, column_id)



def FI_impute():
    print('TODO')

def run_FIMDI(X_train, X_test_m, Y_train, Y_test, column_id, label, X_test_c):
    # X_test_m  has missing data already
    # column_id is the feature missing data
    # label is how missing data is marked
    # X_test_c is complete X test set

    # 1 train my imputer with train set
    FIMDI1 = FIMDImputer(X_train, Y_train, column_id, label)

    FI = FIMDI1.detector.get_FI()

    print(FI)

    print(FI[column_id])

    # 2 feed test set row by row
    # TODO
    #print(X_test_m)

    test_rows = X_test_m.shape[0]

    # get row by row
    for i in range(test_rows):

        # one row is got, do basic MDI
        current_row = X_test_m[i, :]

        print(current_row)

        # check if this row misses data
        if current_row[column_id] == label:
            # missing data
            # do imputation
            # get FI
            a=123
        else:
            # complete row
            # update FI and last observation(basic imputation)

            # get label
            ground_truth = Y_test[i]
            #print(ground_truth)

            # update FI, measure it, then use set_FI
            measure_X = np.array([current_row])

            measure_Y = np.array([[ground_truth]])

            FI1 = FIMDI1.detector.measure_FI(measure_X, measure_Y, 0)
            print('FI1 = ', FI1)

            #FI1 is a dictionary, turn it to list then use set_FI

            # last observation reading
            FIMDI1.mdi.hot_deck_read(current_row)







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

    Y_train = train_set[:, -1]

    #Y_train = Y_train.reshape(Y_train.shape[0], -1)

    test_set = genfromtxt(test_file, delimiter=',')

    X_test_c = np.delete(test_set, train_col-1, 1)

    Y_test = test_set[:, -1]

    #Y_test = Y_test.reshape(Y_test.shape[0], -1)

    #print(X_test_c)
    #print(Y_test)

    # now make missing data
    X_test_m = make_random_missing(column_id, label, X_test_c, 0.5)

    #print(X_test_m)

    run_FIMDI(X_train, X_test_m, Y_train, Y_test, column_id, label, X_test_c)



# test code

read_and_run('A.csv','B.csv', 0)
