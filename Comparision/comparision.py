import pandas as pd
import numpy
import os
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression #1
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #2
from sklearn.tree import DecisionTreeClassifier #3
from sklearn.neighbors import KNeighborsClassifier #4
from sklearn.naive_bayes import MultinomialNB #5
from sklearn.ensemble import RandomForestClassifier #6
from sklearn.svm import SVC #7
import time
start_time = time.time()


def main():
    if not(os.path.exists("/home/xvpher/Intern_Project/Dataset/spamdata.csv")):
        print("data file not found")
        return
    df = pd.read_csv("/home/xvpher/Intern_Project/Dataset/spamdata.csv")
    names = df.columns
    names = names[0:len(names)-1]
    X = df.loc[:,names].values
    y = df.loc[:,'spam'].values
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30, shuffle=False)
    # X_train = normalize(X_train, axis=0, norm='l1')
    # X_test = normalize(X_test, axis=0, norm='l1')
    models = []
    models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=2000, tol=0.0001)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('KNC', KNeighborsClassifier()))
    models.append(('MNB', MultinomialNB()))
    models.append(('RFC', RandomForestClassifier(n_estimators=100)))
    models.append(('SVC', SVC(gamma='scale', kernel='rbf')))

    results = []
    names = []
    for name, model in models:
        fold = model_selection.StratifiedKFold(n_splits=10, random_state=7)
        res = model_selection.cross_val_score(model, X, y, cv=fold, scoring='accuracy')
        results.append(res)
        names.append(name)
        print("{}---{}---{}".format(name, res.mean(), res.std()))

    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


if __name__== '__main__':
    main()
