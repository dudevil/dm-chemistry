__author__ = 'dudevil'


import pandas as pd
import numpy as np
import gc
from sklearn.linear_model import LogisticRegression
import os

n_samples = 10000

X = pd.read_csv("./data/algomostchem_train.txt", header=None, index_col=None, dtype=np.float, nrows=n_samples)

Y = pd.read_csv("./data/algomostchem_trainY.txt", header=None, index_col=None, names=['target'], nrows=n_samples)
Y = Y.values.flatten()

logreg = LogisticRegression()
logreg.fit(X, Y)

print(logreg.classes_)
os._exit()

del X
del Y
gc.collect()

X_test = pd.read_csv("./data/algomostchem_test.txt", header=None, index_col=None, dtype=np.float, nrows=n_samples)
submission = logreg.predict_proba(X_test)
np.savetxt("20150113_dmlt_test.txt", submission, delimiter=",")
