import numpy as np
from sklearn.linear_model import LogisticRegression

# in this code, think of:
#   - h as a function
#   - f as the function that is computed in E(f(X))
#   - X a design matrix
#   - y the labels
#   - s the sensitive attribute

################################################################################
# Tool functions
################################################################################


# vectorized functions (used in expectations to compute E(f(x)))
abs_inverse = np.vectorize(lambda w: 1.0 / np.abs(w))
exp_minus_abs = np.vectorize(lambda w: np.exp(- np.abs(w)))


# compute the mean of two nested dictionaries
def abs_dict_mean(d):

    if not hasattr(d[list(d.keys())[0]], "__len__"):
        return np.mean([np.abs(d[r]) for r in d])

    else:
        return np.mean([[np.abs(d[r1][r2]) for r2 in d[r1]] for r1 in d])


# golden section search (find minimum of u-shaped function)
# implementation adapted from
#   https://en.wikipedia.org/wiki/Golden-section_search
def gss(f, a, b, tol=1e-7):

    # find a big enough value for b
    fa = f(a)
    while f(b) < fa and f(b) > 1e-10 and not(np.isnan(f(b))):
        b *= 10

    # if end up in nan because overflow, go back
    if np.isnan(b):
        a, b = a/10, b/10

    # compute initial values of c d (from golden section search)
    gr = (np.sqrt(5) + 1) / 2
    c, d = b - (b - a) / gr, a + (b - a) / gr

    # while not enough precision run the algo
    fa, fb = f(a), f(b)
    while abs(fb - fa) > 1e-5 and abs(fb - fa) > tol:

        fc, fd = f(c), f(d)

        if fc < fd:
            b = d
            fb = fd
        else:
            a = c
            fa = fc

        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2


# compute distance between two models' paramters
def get_dist(h1, h2, ord=2):

    return np.linalg.norm(h1.coef_ - h2.coef_, ord=ord)




################################################################################
# To get fairness groups
################################################################################

# return the groups concerned by each notion of fairness
# the tuple (y_val, None) or (y_val, y_val) or (None, None)
# is used as follows:
#  - in computation of bounds, it is only used to constitute the groups,
#    as the bounds do not rely on checking a specific value of H(X)=y.
#  - in computation of fairness measures, the second one is used as the
#    value to use: H(X)=y if second one is not None, H(X)=Y otherwise.
def get_fairness_groups(y, s, fairness_measure):

    uniq_y, uniq_s = np.unique(y), np.unique(s)

    full_pop = {}
    sub_pop = {}

    if fairness_measure == "equality_opportunity":

        for y_val in uniq_y[uniq_y > 0]:
            sub_pop[(y_val, None)] = {}
            full_pop[(y_val, None)] = (y == y_val)

            for s_val in uniq_s:
                sub_pop[(y_val, None)][s_val] = (y == y_val) & (s == s_val)

    elif fairness_measure == "equalized_odds":

        for y_val in uniq_y:
            sub_pop[(y_val, None)] = {}
            full_pop[(y_val, None)] = np.full(len(y), True)

            for s_val in uniq_s:
                sub_pop[(y_val, None)][s_val] = (s == s_val)

    elif fairness_measure == "demographic_parity":

        for y_val in uniq_y:
            sub_pop[(y_val, y_val)] = {}
            full_pop[(y_val, y_val)] = np.full(len(y), True)

            for s_val in uniq_s:
                sub_pop[(y_val, y_val)][s_val] = (s == s_val)

    elif fairness_measure == "accuracy_parity":

        sub_pop[(None, None)] = {}
        full_pop[(None, None)] = np.full(len(y), True)

        for s_val in uniq_s:
            sub_pop[(None, None)][s_val] = (s == s_val)

    elif fairness_measure == "accuracy":

        sub_pop = {}
        full_pop[(None, None)] = np.full(len(y), True)


    else:
        print("ERROR!!")
        exit(0)

    return full_pop, sub_pop


################################################################################
# To compute predictions and bounds
################################################################################

# compute expectation of E(f(conf_h(X))),
# where conf_h gives the confidence of h on X
def expect(confs, f, ignore=None):

    fc = f(confs)

    # ignore some records
    # this is a nice trick where points where the mode is confident enough
    # can be considered as having infinite confidence
    if ignore is not None:
        fc[ignore] = 0

    # return the finite sample estimation of E(f(conf_h(x)))
    return np.mean(fc)



# compute the chernoff bound of conf(h) over X
def chernoff_bound(h, dist, X, Ls):

    # create the bound as a function of the parameter t
    f = lambda t: np.exp(t*dist) * expect( h, X, f=lambda x: np.exp(-t*np.abs(x)), Ls=Ls)

    # find the best parameters
    t = gss(f, 0, 1e-5)

    # return the best chernoff bound
    return f(t)




################################################################################
# Confidence of the model
################################################################################

# function that computes the confidence of h(X) as a predictor
# of the correct label y, defined as h_y(x) - sup_{i!=} h_i(x)
#
# there are two possibilities
#   (i)  either h(x)=y then confidence is positive: "how much a value
#        should change to get wrong prediction"
#   (ii) either h(x)!=y then confidence is negative: "how much h_y(x)
#        should change to get correct prediction"
def get_confidences(h, X, y):

    # compute predictions
    decisions = h.decision_function(X)

    if len(h.classes_) == 2:
        decisions = np.vstack((-decisions/2, decisions/2)).T

    # get positions of the right label
    mask = (y[:, None] == h.classes_)

    # confidence level for the right label
    conf_y = decisions[mask]

    # confidence of second best label
    conf_without_y = decisions[~mask].reshape(decisions.shape[0], decisions.shape[1]-1)
    conf_second_best = np.max(conf_without_y, axis=1)

    # return difference between the two
    return conf_y - conf_second_best


# compute equality of opportunity for a given classifier, over
# some dataset X, y, with sensitivity attribute s and some
# desirable outcomes (by default positive labels)
def fairness(h, X, y, s, fairness_measure):

    # get the population groups
    full_pop_dict, sub_pop_dict = get_fairness_groups(y, s, fairness_measure)

    # build the dictionary that will be returned
    ret = { y_val : {} for y_val in full_pop_dict }

    # get decision functions and format it properly in binary case
    decisions = h.decision_function(X)
    if len(h.classes_) == 2:
        decisions = np.vstack((-decisions/2, decisions/2)).T

    # get predictions
    preds = np.array([h.classes_[elt] for elt in np.argmax(decisions, axis=1)])

    # compute predictions for each desirable outcome...
    for ((y_val, y_req_val), full_pop) in full_pop_dict.items():

        y_req = y if y_req_val is None else np.full(len(y), y_req_val)

        # P( H(X) = Y | xxx )
        p_popu = (preds[full_pop] == y_req[full_pop]).mean()

        # ... and for each group
        # if accuracy is requested, there is no subgroups
        if not sub_pop_dict:
            ret[(y_val, y_req_val)] = p_popu
        else:
            for (s_val, sub_pop) in sub_pop_dict[(y_val, y_req_val)].items():

                # P( H(X) = Y | xxx, S = s )
                p_group = (preds[sub_pop] == y_req[sub_pop]).mean()

                # P( H(X) = y | xxx, S = s ) - P( H(X) = y | xxx )
                ret[(y_val, y_req_val)][s_val] = p_group - p_popu

    return ret


# chernoff bound
def chernoff_fairness_bound(h, dist, X, y, s, fairness_measure):

    uniq_y, uniq_s = np.unique(y), np.unique(s)

    # compute confidence
    confs = get_confidences(h, X, y)

    # compute pointwise lipschitz constants
    Ls = 2*np.sqrt(len(uniq_y)) * np.linalg.norm(X, axis=1)

    # get the population groups
    full_pop_dict, sub_pop_dict = get_fairness_groups(y, s, fairness_measure)

    # intialize return dictionaries
    ret = { y_val : {} for y_val in full_pop_dict.keys() }

    for ((y_val, y_req_val), full_pop) in full_pop_dict.items():

        # expectation for the global population
        pop_conf = confs[full_pop]/Ls[full_pop]

        fct_full_pop = \
            lambda t: np.exp(t*dist) \
                      * expect(t*pop_conf,
                               f=exp_minus_abs,
                               ignore=np.abs(pop_conf) > dist)
        t0 = fct_full_pop(gss(fct_full_pop, 0, 1e-5))

        # if accuracy is requested, there is no subgroups
        if not sub_pop_dict:
            ret[(y_val, y_req_val)] = t0

        # otherwise compute according to the correct fairness notion
        else:
            for (s_val, sub_pop) in sub_pop_dict[(y_val, y_req_val)].items():

                # compute which samples are in the current group
                current_conf = confs[sub_pop]/Ls[sub_pop]

                # expectation for the considered group
                fct_group = \
                    lambda t: np.exp(t*dist) \
                    * expect(t*current_conf,
                             f=exp_minus_abs,
                             ignore=np.abs(current_conf) > dist)
                t1 = fct_group(gss(fct_group, 0, 1e-5))

                # compute the bound
                ret[(y_val, y_req_val)][s_val] = t0 + t1

    return ret


# return the projection
def get_orthos_dist(h, h2, X):

    ortho = np.zeros(X.shape)

    for i in range(X.shape[0]):
        Vi = X[i,:] / np.linalg.norm(X[i,:])

        ortho[i,:] = np.outer(Vi, Vi) @ (h.coef_ - h2.coef_).flatten()

    return ortho

# chernoff bound
def chernoff_fairness_ortho_bound(h, h2, X, y, s, fairness_measure):

    uniq_y, uniq_s = np.unique(y), np.unique(s)

    # compute confidence
    confs = get_confidences(h, X, y)

    # compute data norms
    norms = np.linalg.norm(X, axis=1)

    # orthos
    orthos = get_orthos_dist(h, h2, X)

    # compute distances with orthogonal projection
    dists = norms * np.linalg.norm(orthos, axis=1)

    # get the population groups
    full_pop_dict, sub_pop_dict = get_fairness_groups(y, s, fairness_measure)

    # intialize return dictionaries
    ret = { y_val : {} for y_val in full_pop_dict.keys() }

    for ((y_val, y_req_val), full_pop) in full_pop_dict.items():

        # expectation for the global population
        pop_conf = confs[full_pop]/dists[full_pop]

        fct_full_pop = \
            lambda t: np.exp(t) \
                      * expect(t*pop_conf,
                               f=exp_minus_abs,
                               ignore=np.abs(pop_conf) > 1)
        t0 = fct_full_pop(gss(fct_full_pop, 0, 1e-5))

        # if accuracy is requested, there is no subgroups
        if not sub_pop_dict:
            ret[(y_val, y_req_val)] = t0

        # otherwise compute according to the correct fairness notion
        else:
            for (s_val, sub_pop) in sub_pop_dict[(y_val, y_req_val)].items():

                # compute which samples are in the current group
                current_conf = confs[sub_pop]/dists[sub_pop]

                # expectation for the considered group
                fct_group = \
                    lambda t: np.exp(t) \
                    * expect(t*current_conf,
                             f=exp_minus_abs,
                             ignore=np.abs(current_conf) > 1)
                t1 = fct_group(gss(fct_group, 0, 1e-5))

                # compute the bound
                ret[(y_val, y_req_val)][s_val] = t0 + t1

    return ret
