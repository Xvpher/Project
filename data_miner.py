import os
import time
import _pickle as pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler

def main():
    path = os.getcwd()
    new_path = path+"/Dataset"
    datafile_path = path+"/Dataset/spamdata.csv"
    if not(os.path.exists(datafile_path)):
        print("Could not find the data file")
        return
    df = pd.read_csv(datafile_path)
    features = df.iloc[:,0:57].values
    labels = df.iloc[:,57].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = tts(features, labels, test_size=0.2, shuffle=True, random_state=69)
    X_train = np.reshape(X_train, (-1,57,1))
    X_test = np.reshape(X_test, (-1,57,1))
    var_names = [X_train, X_test, y_train, y_test]
    filenames = ["X_train", "X_test", "y_train", "y_test"]
    for i in range(4):
        file = new_path+"/"+filenames[i]
        with open(file, 'wb') as outfile:
            pickle.dump(var_names[i], outfile)

if __name__ == '__main__':
    main()
