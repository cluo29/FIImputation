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
    # first, mode, mean, median, hot deck impute.
    def __init__(self, X_train, column_id):
        a = np.array(X_train)
        self.column_id = column_id
        self.mean = np.mean(a[:,column_id])

    # every time needs MDI, we accept only one row
    def mean_impute(self, inputRow):
        outputRow = np.array(inputRow)
        outputRow[self.column_id] = self.mean
        return outputRow



# test code
X_train = [[1,2,3],[3,2,1],[0,1,1]]
cjj = MDImputer(X_train,0)
