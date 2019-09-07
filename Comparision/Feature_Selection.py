import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts

def main():
    df = pd.read_csv("/home/xvpher/Intern_Project/Dataset/spamdata.csv")
    features = df.iloc[:,0:57].values
    labels = df.iloc[:,57].values
    X_train, X_test, y_train, y_test = tts(features, labels, test_size=0.25, shuffle=True, random_state=8)

    df1 = df.corr()
    correlation = df1.values
    ticks = df.columns
    fig, ax = plt.subplots()
    im = ax.imshow(correlation, cmap='hot')
    ax.set_xticks(np.arange(len(ticks)))
    ax.set_yticks(np.arange(len(ticks)))
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right', rotation_mode='anchor')
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig('map.png')

if __name__ == '__main__':
    main()
