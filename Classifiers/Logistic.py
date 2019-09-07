import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import urllib as ulib

def main():
    if not(os.path.exists("/home/xvpher/Intern_Project/Dataset/spamdata.csv")):
        print("Could not find the data file")
        return
    df = pd.read_csv("/home/xvpher/Intern_Project/Dataset/spamdata.csv")
    names = df.columns
    names = names[0:len(names)-1]
    X = df.loc[:,names].values
    y = df.loc[:,'spam'].values
    X,y = shuffle(X,y)
    n = 3680
    X_train = X[:n,:]
    y_train = y[:n]
    X_test = X[n:,:]
    y_test = y[n:]
    model  = LogisticRegression(solver='lbfgs', max_iter=2000, tol=0.0001)
    model.fit(X_train,y_train)
    print (model.score(X_test,y_test))
    # predictions = model.predict(X_test)
    # for i in range(100):
    #     print ("{}------{}".format(y_test[i], predictions[i]))

if __name__ == '__main__':
    main()
