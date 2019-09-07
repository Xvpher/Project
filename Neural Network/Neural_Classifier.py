import os
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler


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
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32,activation='relu',input_dim=57,kernel_initializer='random_normal'))
    model.add(tf.keras.layers.Dense(32,activation='relu',kernel_initializer='random_normal'))
    model.add(tf.keras.layers.Dense(1,activation='sigmoid',kernel_initializer='random_normal'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(X_train,y_train,batch_size=10,epochs=50)
    model.save("Neural_Network.model")


if __name__ == '__main__':
    main()
