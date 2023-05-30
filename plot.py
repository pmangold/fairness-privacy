import pickle
import sys, os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
palette = sns.color_palette("colorblind")
muted_palette = sns.color_palette("muted")
dark_palette = sns.color_palette("muted")

################################################################################
# plot parameters
################################################################################

mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16

################################################################################
# labels
################################################################################

yaxis_label = {
    "equalized_odds": "Equalized Odds",
    "equality_opportunity": "Equality of Opportunity",
    "accuracy_parity": "Accuracy Parity",
    "demographic_parity": "Demographic Parity",
    "accuracy": "Accuracy"
}


################################################################################
# load dataset and dataset parameters
################################################################################


# load dataset from command line
dataset = sys.argv[1]

print(dataset)

if dataset != "legend":
    fairness_measure = sys.argv[2]
    mu = sys.argv[3]


ylim_n = {
    "celebA": (0, 0.4),
    "folktables": (0, 0.1)
}

ylim_epsilon = {
    "celebA": (0, 0.15),
    "folktables": (0, 0.04)
}

ylim_accuracy_n = {
    "celebA": (0.4, 1),
    "folktables": (0.65, 0.75),
}

ylim_accuracy_epsilon = {
    "celebA": (0.75, 0.9),
    "folktables": (0.7, 0.75),
}

epsilon_range = {
    "celebA": (0, 10),
    "folktables": (0, 1)
}

################################################################################
# if dataset is legend, generate the legend and exit
################################################################################

if dataset == "legend":
    fig, ax = plt.subplots(figsize=(4,3))
    leg_fig, leg_ax = plt.subplots(figsize=(13.9,0.6))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.axis("off")
    leg_ax.xaxis.set_visible(False)
    leg_ax.yaxis.set_visible(False)
    leg_ax.axis("off")

    # bound
    hdl_bound, = ax.plot([0],[0], color=palette[0])
    hdl_emp, = ax.plot([0], [0], color=palette[0], ls="dashed")
    hdl_proj, = ax.plot([0], [0], color=palette[0], ls="dotted")
    hdl_non_priv, = ax.plot([0], [0],
                            color=palette[1], lw=2,
                            marker="x", markersize=5, markerfacecolor=palette[1])

    hdl_priv, = ax.plot([0], [0],
                        markersize=5,
                        color=palette[2],
                        lw=0,
                        marker=".")
    fill_priv, = ax.fill(np.NaN, np.NaN, color=palette[2], alpha=0.5)

    print(fill_priv, hdl_priv)

    legend = leg_fig.legend(
        [hdl_bound,
         hdl_emp,
         hdl_proj
         ],
        ["Theoretical Upper Bound",
         "Bound with Empirical Distance",
         "Improved Bound Knowning both Models"],
        ncol=3, frameon=False, loc="lower center",
        bbox_to_anchor=[0.5,0])
    legend2 = leg_fig.legend(
        [hdl_non_priv,
         (hdl_priv, fill_priv)],
        ["Non-private Model Fairness",
         "Private Models Fairness"],
        ncol=2, frameon=False, loc="lower center",
        bbox_to_anchor=[0.5,-0.7])
    leg_ax.add_artist(legend)

    leg_fig.savefig("plots/legend.pdf", bbox_inches="tight")
    exit()


################################################################################
# plot fairness around absolute fairness of the non private model
# as a function of n
################################################################################


# load results
with open(f"results/fairness_fct_n/mu{mu}/{dataset}_{fairness_measure}.pickle", "rb") as f:
    results = pickle.load(f)

print(results.keys())


ns = results["ns"]
npoints = results["npoints"]
bound = results["bound"]
empirical_bound = results["empirical_bound"]
proj_bound = results["proj_bound"]
clfs = results["clfs"]
clfs_priv = results["clfs_priv"]
fairness_values = results["fairness_values"]
fairness_values_priv = results["fairness_values_priv"]
epsilon = results["epsilon"]
delta = results["delta"]
mu = results["mu"]
L = results["L"]


if fairness_measure == "accuracy_parity" or fairness_measure == "demographic_parity":
    print(dataset, fairness_measure, bound)
    print(dataset, fairness_measure, empirical_bound)
    print()

print(fairness_measure == results["fairness_measure"])
print(dataset == results["dataset"])


os.makedirs(f"plots/fairness_fct_n/{fairness_measure}", exist_ok=True)

fig, ax = plt.subplots(figsize=(4,3))

# bound
hdl_bound, = plt.plot(ns,
                      np.maximum(0, fairness_values - bound),
                      color=palette[0])
plt.plot(ns,
         np.minimum(1, fairness_values + bound),
         color=palette[0])

# empirical bound with dashed lines (should be smaller)
hdl_emp, = plt.plot(ns,
                    np.maximum(0, fairness_values - empirical_bound),
                    color=palette[0],
                    ls="dashed")
plt.plot(ns,
         np.minimum(1, fairness_values + empirical_bound),
         color=palette[0],
         ls="dashed")

# proj bound with dotted lines (should be even smaller)
hdl_proj, = plt.plot(ns,
                    np.maximum(0, fairness_values - proj_bound),
                    color=palette[0],
                    ls="dotted")
plt.plot(ns,
         np.minimum(1, fairness_values + proj_bound),
         color=palette[0],
         ls="dotted")

# fairness values
hdl_non_priv, = plt.plot(ns,
                         fairness_values,
                         color=palette[1],
                         lw=2,
                         marker="x",
                         markersize=5,
                         markerfacecolor=palette[1],
                         zorder=20)

# private values
plt.fill_between(ns,
                 np.min(fairness_values_priv, axis=0),
                 np.max(fairness_values_priv, axis=0),
                 color=palette[2],
                 edgecolor=palette[2],
                 alpha=0.5,
                 zorder=10)
hdl_priv = plt.scatter(np.tile(ns, npoints),
                       fairness_values_priv.flatten(),
                       s=20,
                       color=palette[2],
                       linewidths=1,
                       marker=".",
                       zorder=10)

#                         markeredgecolor="grey")


plt.xlabel("Number of training samples")
plt.ylabel(yaxis_label[fairness_measure])


if fairness_measure == "accuracy":
    plt.ylim(ylim_accuracy_n[dataset])
else:
    plt.ylim(ylim_n[dataset])

plt.savefig(f"plots/fairness_fct_n_{fairness_measure}_fairness_fct_n_{dataset}_{epsilon}.pdf", bbox_inches="tight")


plt.xscale("log")
plt.savefig(f"plots/fairness_fct_n_{fairness_measure}_fairness_fct_n_{dataset}_{epsilon}_logscale.pdf", bbox_inches="tight")



################################################################################
# plot fairness around absolute fairness of the non private model
# as a function of epsilon
################################################################################

# load results
with open(f"results/fairness_fct_epsilon/mu{mu}/{dataset}_{fairness_measure}.pickle", "rb") as f:
    results = pickle.load(f)

print(results.keys())

epsilons = results["epsilons"]
npoints = results["npoints"]
bound = results["bound"]
empirical_bound = results["empirical_bound"]
proj_bound = results["proj_bound"]
clfs = results["clfs"]
clfs_priv = results["clfs_priv"]
fairness_values = results["fairness_values"]
fairness_values_priv = results["fairness_values_priv"]
delta = results["delta"]
mu = results["mu"]
L = results["L"]

print(fairness_measure == results["fairness_measure"])
print(dataset == results["dataset"])

fig, ax = plt.subplots(figsize=(4,3))

# bound
hdl_bound, = plt.plot(epsilons,
                      np.maximum(0, fairness_values - bound),
                      color=palette[0])
plt.plot(epsilons,
         np.minimum(1, fairness_values + bound),
         color=palette[0])

# empirical bound with dashed lines (should be smaller)
hdl_emp, = plt.plot(epsilons,
                    np.maximum(0, fairness_values - empirical_bound),
                    color=palette[0],
                    ls="dashed")
plt.plot(epsilons,
         np.minimum(1, fairness_values + empirical_bound),
         color=palette[0],
         ls="dashed")

# projected bound with dashed lines (should be smaller)
hdl_proj, = plt.plot(epsilons,
                    np.maximum(0, fairness_values - proj_bound),
                    color=palette[0],
                    ls="dotted")
plt.plot(epsilons,
         np.minimum(1, fairness_values + proj_bound),
         color=palette[0],
         ls="dotted")

# fairness values
hdl_non_priv, = plt.plot(epsilons,
                         fairness_values,
                         color=palette[1],
                         lw=2,
                         marker="x",
                         markersize=5,
                         markerfacecolor=palette[1],
                         zorder=20)
#                         markeredgecolor=palette[1])

plt.fill_between(epsilons,
                 np.min(fairness_values_priv, axis=0),
                 np.max(fairness_values_priv, axis=0),
                 color=palette[2],
                 edgecolor=palette[2],
                 alpha=0.5,
                 zorder=10)
hdl_priv = plt.scatter(np.tile(epsilons, npoints),
                       fairness_values_priv.flatten(),
                       s=20,
                       color=palette[2],
                       linewidths=1,
                       marker=".",
                       zorder=10)


plt.xlabel("Value of $\epsilon$")
plt.ylabel(yaxis_label[fairness_measure])

if fairness_measure == "accuracy":
    plt.ylim(ylim_accuracy_epsilon[dataset])
else:
    plt.ylim(ylim_epsilon[dataset])

emin, emax = epsilon_range[dataset]
xmin = emin - 0.05 * (emax - emin)
xmax = emax + 0.05 * (emax - emin)

plt.xlim(xmin, xmax)

plt.savefig(f"plots/fairness_fct_epsilon_{fairness_measure}_fairness_fct_epsilon_{dataset}_{epsilon}.pdf", bbox_inches="tight")

plt.xscale("log")
plt.xlim(6e-4, xmax)

plt.savefig(f"plots/fairness_fct_epsilon_{fairness_measure}_fairness_fct_epsilon_{dataset}_{epsilon}_logscale.pdf", bbox_inches="tight")
