import pickle
import sys, os
import numpy as np

datasets = ["celebA", "folktables"]
fairness_measures = ["equality_opportunity",
                     "equalized_odds",
                     "demographic_parity",
                     "accuracy_parity",
                     "accuracy"]

n_samples = {
    "celebA": "182,339",
    "folktables": "1,498,050"
}

print(";", end="")
print(";".join([f.replace("_", " ").capitalize() for f in fairness_measures]))

for dataset in datasets:
    print(dataset, f" ($n={n_samples[dataset]}$)", sep="", end=";")

    dataset_bound = []
    for fairness_measure in fairness_measures:
        # load results
        with open(f"results/fairness_fct_n/mu1/{dataset}_{fairness_measure}.pickle", "rb") as f:
            results = pickle.load(f)

        dataset_bound.append(str('{:.4f}'.format(results["bound"][-1])))

    print(";".join(dataset_bound))
