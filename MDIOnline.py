# MDIOnline.py
# basic missing data imputation for online classification
# their results should be filtered by FIMDI
# the models are trained using train set only (train set has no data missing)

# Chu Luo
import numpy as np
from scipy import stats
from sklearn import *
from sklearn import preprocessing
from sklearn.metrics import *


###
# CLASS DEFINITION
###

class MDImputer():
    """
    basic MDI for online classification.
    """
    # train the imputer
    def __init__(self, X_train, column_id):
        a = np.array(X_train, dtype='f')
        self.column_id = column_id
        self.mean = np.mean(a[:,column_id])
        self.median = np.median(a[:,column_id])

        mode, count = stats.mode(a[:, column_id])
        self.mode = mode
        self.last = a[-1, column_id]

        X_MDI_train = np.delete(a, column_id, 1)
        Y_MDI_train = a[:, column_id]

        # LR model
        self.LRregr = linear_model.LinearRegression()
        self.LRregr.fit(X_MDI_train, Y_MDI_train)

        # KNN regressor
        self.KNregr = neighbors.KNeighborsRegressor( n_neighbors=10)
        self.KNregr.fit(X_MDI_train, Y_MDI_train)

        # MLP
        self.MLPregr = neural_network.MLPRegressor()
        self.MLPregr.fit(X_MDI_train, Y_MDI_train)



    def all_impute(self, inputRow):
        # do all MDI using functions

        # the list of all impute
        outputList = []
        outputList.append(self.mean_impute())
        outputList.append(self.median_impute())
        outputList.append(self.mode_impute())
        outputList.append(self.hot_deck_impute())
        outputList.append(self.lr_impute(inputRow))
        outputList.append(self.knn_impute(inputRow))
        outputList.append(self.MLP_impute(inputRow))

        return outputList

    # first, mode, mean, median, hot deck impute.
    # every time needs MDI, we accept only one row
    # output imputation
    def mean_impute(self):
        return self.mean

    def median_impute(self):
        return self.median

    def mode_impute(self):
        return self.mode[0]

    # get a complete row for hot_deck_impute (last observation)
    # if not having this, will use last row of train set
    def hot_deck_read(self, inputRow):
        self.last = inputRow[self.column_id]

    def hot_deck_impute(self):
        return self.last

    def lr_impute(self, inputRow):
        inputRowNP = np.array([inputRow], dtype='f')
        Row_X = np.delete(inputRowNP, self.column_id, 1)
        Row_Y = self.LRregr.predict(Row_X)
        return Row_Y[0]

    def knn_impute(self, inputRow):
        inputRowNP = np.array([inputRow], dtype='f')
        Row_X = np.delete(inputRowNP, self.column_id, 1)
        Row_Y = self.KNregr.predict(Row_X)
        return Row_Y[0]

    def MLP_impute(self, inputRow):
        # ain gonna normalize
        inputRowNP = np.array([inputRow], dtype='f')
        Row_X = np.delete(inputRowNP, self.column_id, 1)
        Row_Y = self.MLPregr.predict(Row_X)
        return Row_Y[0]



# test code




"""
X_train = [[1,1,1],[1,2,2],[3,3,3],[5,5,5]]

for i in range(6):
    X_train.append([5,5,5])

cjj = MDImputer(X_train,0)

row = [0,4,4]
#
# cjj.hot_deck_read([9,9,9])
b=cjj.all_impute(row)
print(b)
"""
