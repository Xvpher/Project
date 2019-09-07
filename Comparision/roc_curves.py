import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as LR #1
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA #2
from sklearn.tree import DecisionTreeClassifier as DTC #3
from sklearn.neighbors import KNeighborsClassifier as KNC #4
from sklearn.naive_bayes import MultinomialNB as MNB #5
from sklearn.ensemble import RandomForestClassifier as RFC #6
from sklearn.svm import SVC #7
from sklearn.metrics import roc_curve, auc, roc_auc_score

def main():
    df = pd.read_csv("/home/saxobeat/PythonML/MLCodes/Spambase/Dataset/spamdata.csv")
    features = df.iloc[:,0:57].values
    labels = df.iloc[:,57].values
    X_train, X_test, y_train, y_test = tts(features, labels, test_size=0.25, shuffle=True, random_state=8)
    models = []
    models.append(('LR', LR(solver='lbfgs', max_iter=2000, tol=0.0001)))
    models.append(('LDA', LDA()))
    models.append(('DTC', DTC()))
    models.append(('KNC', KNC()))
    models.append(('MNB', MNB()))
    models.append(('RFC', RFC(n_estimators=100)))
    models.append(('SVC', SVC(gamma='scale', kernel='rbf', probability=True)))
    x0 = np.linspace(0,1,10)
    plt.plot([0,1],[0,1],'k',linestyle='--')
    for name,model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        y_score = y_pred[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        label = "{}({})".format(name,auc(fpr, tpr))
        plt.plot(fpr,tpr,label=label)
        plt.legend()
        # plt.legend(name)

    plt.title("Reciever Operating Characteristics")
    plt.grid()
    plt.cool()
    plt.xlabel("Fasle Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("roc.pdf")
    # plt.show()

if __name__ == '__main__':
    main()
