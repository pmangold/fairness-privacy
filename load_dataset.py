import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder


def get_celebA(seed):

    df = pd.read_csv("data/celebA_preprocessed.txt", sep=",")
    X = df.loc[:, df.columns != "Smiling"]
    y = df["Smiling"]

    groups = X["Male"]
    X = X.drop("Male", axis=1)

    norms = np.linalg.norm(X, axis=1)
    X[norms>2] *= 6 / norms[norms>6][:,None]

    return tuple(np.array(r) for r in train_test_split(X, y, groups,
                                                       random_state=seed,
                                                       test_size=0.1))

def get_folktables(seed):
    df = pd.read_csv("data/data.csv")

    target = np.array(df["TARGET"])
    y = np.zeros(len(target))
    y[target == True] = 1
    y[target == False] = -1

    groups = df["SEX"]
    X = df.drop(["TARGET", "SEX"], axis=1)

    X = np.array(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    norms = np.linalg.norm(X, axis=1)
    X[norms>=3] *= 3/norms[norms>=3][:,None]

    return tuple(np.array(r) for r in train_test_split(X, y, groups,
                                                       random_state=seed,
                                                       test_size=0.1))



def get_dataset(dataset, seed):

    if dataset == "celebA":
        return get_celebA(seed)

    if dataset == "folktables":
        return get_folktables(seed)
