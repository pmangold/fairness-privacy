import sys, os
import numpy as np
import pandas as pd
from copy import deepcopy

import seaborn as sns
palette = sns.color_palette("colorblind")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from load_dataset import *
from compute_bounds import *

# initialize rng
rng = np.random.default_rng(seed=42)

# load dataset from command line
dataset = sys.argv[1]
fairness_measure = sys.argv[2]
Xtrain, Xtest, ytrain, ytest, gtrain, gtest = get_dataset(dataset, 42)

# number of samples from 1000 to size of dataset
ns = np.logspace(1, np.log10(Xtrain.shape[0]), 20, dtype=int)

# constants of the problem
mu = 1
L = max(np.linalg.norm(Xtrain, axis=1, ord=2))
p = Xtrain.shape[1]

zeta = 0.01

# different values for different datasets
epsilon = {
    "celebA": 1,
    "folktables": 1
}[dataset]


print(f"mu={mu}\nL={L}\np={p}\nzeta={zeta}")

# number of points to evalue
npoints = 100

# initialize models
clfs = [LogisticRegression(fit_intercept=False, C=n/mu, max_iter=10000) for n in ns]
clfs_priv = [
    [LogisticRegression(fit_intercept=False, C=n/mu, max_iter=10000) for n in ns]
    for _ in range(npoints)
]


for i, n in enumerate(ns):

    # update delta as a function of n
    delta = 1.0 / n ** 2

    # take first n elements (that are random because of shuffling)
    features = Xtrain[:n,:]
    labels = ytrain[:n]

    # fit non private classifier with these elements
    clfs[i].fit(features, labels)

    # fit the other classifiers by output perturbation
    for j in range(npoints):
        clfs_priv[j][i] = deepcopy(clfs[i])

        clfs_priv[j][i].coef_ = clfs[i].coef_ + rng.normal(
            loc=0,
            scale=L*np.sqrt(8 * np.log(1.25/delta)) / (n*mu*epsilon),
            size=Xtrain.shape[1]
       )

# compute the upper bound
bound = np.zeros(len(ns))
empirical_bound = np.zeros(len(ns))
proj_bound = np.zeros(len(ns))


# base distance computed from tail property of exp(x^2)
dists = [
    L*np.sqrt(32*p*np.log(1.25/delta)*np.log(2/zeta)) / (epsilon*mu*n)
    for n in ns
]
# empirical distance
empirical_dists = [
    np.quantile([
        get_dist(clfs[i], clfs_priv[j][i])
        for j in range(npoints)
    ], 1-zeta)
    for i in range(len(ns))
]

print("DISTANCES:")
print(dists)
print(empirical_dists)

# compute the bounds for both upper and empirical distances
for i, n in enumerate(ns):

    print(f"\n----------- Current n = {n} -----------\n")

    # compute
    ret_true = chernoff_fairness_bound(clfs[i], dists[i], Xtest, ytest, gtest,
                                       fairness_measure=fairness_measure)
    bound[i] = abs_dict_mean(ret_true)

    ret_emp = chernoff_fairness_bound(clfs[i], empirical_dists[i], Xtest, ytest, gtest,
                                      fairness_measure=fairness_measure)

    empirical_bound[i] = abs_dict_mean(ret_emp)

    index_biggest = np.argmax(empirical_dists[i])
    ret_orthos = chernoff_fairness_ortho_bound(clfs[i], clfs_priv[index_biggest][i], Xtest, ytest, gtest,
                                               fairness_measure=fairness_measure)

    proj_bound[i] = abs_dict_mean(ret_orthos)

    print(dataset, bound[i], empirical_bound[i], proj_bound[i])

# compute scores
scores = [clfs[i].score(Xtest, ytest) for i in range(len(ns))]
scores_priv = [
    [clfs_priv[j][i].score(Xtest, ytest) for i in range(len(ns))]
    for j in range(npoints)
]

# compute absolute fairness of non private models
fairness_values = np.array([
    abs_dict_mean(fairness(model, Xtest, ytest, gtest, fairness_measure))
    for model in clfs
])

# compute absolute fairness of private models
model = clfs_priv[0][0]
fairness_values_priv = np.array([
    [abs_dict_mean(fairness(model, Xtest, ytest, gtest, fairness_measure))
     for model in clfs_priv[j]]
    for j in range(npoints)
])

print(fairness_values)
print(np.mean(fairness_values_priv, axis=0))

################################################################################
# save the results
################################################################################

import pickle

os.makedirs(f"results/fairness_fct_n/mu{mu}/", exist_ok=True)

with open(f"results/fairness_fct_n/mu{mu}/{dataset}_{fairness_measure}.pickle","wb") as f:
    pickle.dump(
        {
            "epsilon": epsilon,
            "delta": delta,
            "mu": mu,
            "L": L,
            "ns": ns,
            "npoints": npoints,
            "bound": bound,
            "empirical_bound": empirical_bound,
            "proj_bound": proj_bound,
            "clfs": clfs,
            "clfs_priv": clfs_priv,
            "fairness_values": fairness_values,
            "fairness_values_priv": fairness_values_priv,
            "dataset": dataset,
            "fairness_measure": fairness_measure
        },
        f)
