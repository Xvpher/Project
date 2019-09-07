import os
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def main():
    if not(os.path.exists("/home/xvpher/PythonML/MLCodes/Spambase/Dataset/spamdata.csv")):
        print("Could not find the data file")
        return
    df = pd.read_csv("/home/xvpher/PythonML/MLCodes/Spambase/Dataset/spamdata.csv")
    features = df.iloc[:,0:57].values
    labels = df.iloc[:,57].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = tts(features, labels, test_size=0.25, shuffle=True, random_state=8)
    model = tf.keras.models.load_model("Neural_Network.model")
    predictions = model.predict(X_test)
    # print (model.evaluate(X_train,y_train))
    y_pred = model.predict(X_test)
    y_pred = (y_pred>0.5)
    cm = confusion_matrix(y_test, y_pred)
    tn,fp,fn,tp = cm.ravel()
    accuracy = (tp+tn)/len(y_test)
    precision = (tp)/(fp+tp)
    recall = (tp)/(fn+tp)
    print ("-------------------")
    print (model.summary())
    print (cm)
    print ("Accuracy is = {} \nPrecision is = {}\nRecall is = {}".format(accuracy,precision,recall))

if __name__ == '__main__':
    main()
