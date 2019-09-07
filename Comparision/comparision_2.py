import pandas as pd
import numpy
import os
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression #1
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #2
from sklearn.tree import DecisionTreeClassifier #3
from sklearn.neighbors import KNeighborsClassifier #4
from sklearn.naive_bayes import MultinomialNB #5
from sklearn.ensemble import RandomForestClassifier #6
from sklearn.svm import SVC #7
from statsmodels.stats.contingency_tables import mcnemar


def main():
    if not(os.path.exists("/home/xvpher/Intern_Project/Dataset/spamdata.csv")):
        print("data file not found")
        return
    df = pd.read_csv("/home/xvpher/Intern_Project/Dataset/spamdata.csv")
    col_names = df.columns
    col_names = col_names[0:len(col_names)-1]
    features = df.loc[:,col_names].values
    labels = df.loc[:,'spam'].values
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.25, shuffle=True, random_state=3)
    models = []
    names = []
    models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=2000, tol=0.0001)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('KNC', KNeighborsClassifier()))
    models.append(('MNB', MultinomialNB()))
    models.append(('RFC', RandomForestClassifier(n_estimators=100)))
    models.append(('SVC', SVC(gamma='scale', kernel='rbf')))

    predictions = pd.DataFrame(data=y_test, columns=['y_test'])
    for name,model in models:
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        names.append(name)

    for k in range(7):
        for j in range(k+1,7):
            table = numpy.zeros((2,2), dtype=numpy.int64)
            for i in range(len(y_test)):
                a = int(not(predictions.loc[i,'y_test'] ^ predictions.iloc[i,k+1]))
                b = int(not(predictions.loc[i,'y_test'] ^ predictions.iloc[i,j+1]))
                # predictions.loc[i,'score1'] = a
                # predictions.loc[i,'score2'] = b
                if (a==1 and b==1):
                    table[0][0] += 1
                elif (a==1 and b==0):
                    table[0][1] += 1
                elif (a==0 and b==1):
                    table[1][0] += 1
                else :
                    table[1][1] += 1

            score = mcnemar(table, exact=False)
            print ("-------({},{})--------".format(names[k],names[j]))
            print (table)
            print (score)
    # print (predictions)

if __name__== '__main__':
    main()
