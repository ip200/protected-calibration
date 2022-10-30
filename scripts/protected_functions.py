import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

THETA = [[0, 1], [0, 0.5], [0, 2], [1, 1], [1, 0.5], [1, 2], [-1, 1], [-1, 0.5], [-1, 2]]
J_RATES = [10 ** (-2), 10 ** (-3), 10 ** (-4)]  # the jumping rates; canonical: 0.01, 0.001, 0.0001

alphas = [0, 1, -1]  # make sure to start from 0 (the neutral value)
betas = [1, 0.5, 2]  # make sure to start from 1 (the neutral value)
n_cal_alpha = len(alphas)
n_cal_beta = len(betas) # the number of beta calibrators
n_cal = n_cal_alpha * n_cal_beta


# the Brier loss function:
def brier_loss(y, p, k):
    loss = 0
    for i in range(k):
        if y == i:
            loss += (1 - p[i]) ** 2
        else:
            loss += p[i] ** 2
    return loss


# Arithmetic average of numbers given on the log10 scale:
def log_mean(x):
    m = np.max(x)
    return m + np.log10(np.mean(np.exp(np.log(10) * (x - m))))


def cox(p, alpha, beta, k):
    p_mod = np.empty(k)  # initializing the modified probability measure
    p_mod[1] = p[1]**beta*np.exp(alpha)
    p_mod[0] = p[0]**beta
    return p_mod/np.sum(p_mod)


def my_cal(p, cal_id, k):  # exactly as in OCM 32
    return cox(p, THETA[cal_id][0], THETA[cal_id][1], k)


def calc_pp(p_pred, n_test, k):
    # Transforming the base predictions to the multiclass form:
    pp = np.empty((n_test, k)) # initializing the base prediction as vector
    for n in range(n_test):   # going through all test observations
        pp[n, 1] = p_pred[n]     # base prediction (no truncation)
        pp[n, 0] = 1 - p_pred[n]   # base prediction (no truncation)
    return pp


def calc_martingale(p_pred, y_test, n_test, k, plot_charts=False):
    # Parameters
    # pi = 0.5  # assumed but not used explicitly
    n_jumpers = len(J_RATES)

    # initializing the SJ and CJ test martingales on the log scale (including the initial value of 0):
    log_sj_martingale = np.zeros((n_jumpers, (n_test + 1)))
    log_cj_martingale = np.zeros(n_test + 1)

    pp = calc_pp(p_pred, n_test, k)

    # Processing the dataset
    for j_index in range(n_jumpers):  # going over all jumping rates
        j_rate = J_RATES[j_index]  # the current jumping rate
        mart_cap = np.zeros(n_cal)  # the normalized capital in each state (after normalization at the previous step)
        mart_cap[0] = 1  # the initial distribution for each jumping rate is concentrated at 0
        # MartCap[:] = 1/Ncal  # the initial distribution for each jumping rate is uniform (old)
        for n in range(n_test):
            # Jump mixing starts
            capital = np.sum(mart_cap[:])  # redundant; Capital = 1
            mart_cap[:] = (1 - j_rate) * mart_cap[:] + (j_rate / n_cal * capital)
            # Jump mixing ends
            # ppp = truncate(p_pred[n])   # base prediction
            for cal_id in range(n_cal):
                # new_ppp = my_cal(ppp,k)    # our new prediction
                new_ppp = my_cal(pp[n], cal_id, k)  # our new prediction
                # at this point I know that Bern(ppp,y_test[n])!=0
                # MartCap[k] *= Bern(new_ppp,y_test[n]) / Bern(ppp,y_test[n])
                # for i in range(K):
                mart_cap[k] *= np.exp(-brier_loss(y_test[n], new_ppp, k)) / np.exp(-brier_loss(y_test[n], pp[n], k))
            increase = np.sum(mart_cap[:])  # relative increase in my capital
            log_sj_martingale[j_index, n + 1] = log_sj_martingale[j_index, n] + np.log10(increase)
            mart_cap[:] /= increase

    for n in range(n_test + 1):
        log_cj_martingale[n] = log_mean([0, log_mean(log_sj_martingale[:, n])])  # 1 becomes 0 on the log scale

    if plot_charts:
        fig, ax = plt.subplots(figsize=(15, 5))
        plt.plot(log_cj_martingale, c='b')  # for CJ martingale
        plt.ylabel('log10 test martingale')  # choose singular or plural
        plt.title("CJ martingale")
        plt.show()

        # Interesting values for the caption or text:
        print("Final values of SJs on the log scale:", np.round(log_sj_martingale[:, n_test]))
        print("Final value of CJ on the log scale:", np.round(log_cj_martingale[n_test]))

    return log_sj_martingale, log_cj_martingale


def calc_losses(p_pred, y_test, p_prime, pp, n_test, k):
    cum_loss_base = 0  # initialization of the loss
    for n in range(n_test):
        cum_loss_base += brier_loss(y_test[n], pp[n], k)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, p_pred, pos_label=1)
    roc_auc_base = metrics.auc(fpr, tpr)
    print("Base Brier loss:", cum_loss_base)
    print("Base AUC:", roc_auc_base)

    cum_loss_prot = 0  # initialization of the loss
    for n in range(n_test):
        cum_loss_prot += brier_loss(y_test[n], p_prime[n], k)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, p_prime[:, 1], pos_label=1)
    roc_auc_prot = metrics.auc(fpr, tpr)
    print("Protected Brier loss:", cum_loss_prot)
    print("Protected AUC:", roc_auc_prot)

    return cum_loss_base, roc_auc_base, cum_loss_prot, roc_auc_prot


def calibrate_probs(p_pred, y_test, n_test, k):
    # Parameters
    pi = 0.5  # default: 0.5
    n_jumpers = len(J_RATES)

    pp = calc_pp(p_pred, n_test, k)
    # initializing the predictive probability measures:
    p_prime = np.empty((n_test, k))

    # Processing the dataset
    p_weight = pi  # amount set aside (passive weight)
    a_weight = np.zeros((n_jumpers, n_cal)) # the weight of each active state
    a_weight[:, 0] = (1 - pi) / n_jumpers  # initial weights
    for n in range(n_test):  # going through all test observations
        # Jump mixing starts
        for j_index in range(n_jumpers):
            capital = np.sum(a_weight[j_index, :])  # active capital for this jumping rate
            j_rate = J_RATES[j_index]
            a_weight[j_index, :] = (1 - j_rate) * a_weight[j_index, :] + capital * j_rate / n_cal
        # Jump mixing ends
        g = np.empty(k)  # pseudoprediction initialized
        for i in range(k):
            g[i] = p_weight * np.exp(-brier_loss(i, pp[n], k))  # initializing the pseudoprediction to its passive component
            for cal_id in range(n_cal):
                cal_pp_k = my_cal(pp[n], cal_id, k)  # prediction calibrated by the k-th calibrator
                for j_index in range(n_jumpers):
                    g[i] += a_weight[j_index, k] * np.exp(-brier_loss(i, cal_pp_k, k))  # accumulating predictions calibrated by the calibrators
            g[i] = -np.log(g[i])
        # We need to solve equation for s, let's first try a shortcut:
        s = (2 + np.sum(g)) / k
        for i in range(k):
            p_prime[n, i] = (s - g[i]) / 2  # my prediction
        if s - np.max(g) < 0:
            print("Wrong s for n =", n)
        # Updating the weights:
        p_weight *= np.exp(-brier_loss(y_test[n], pp[n], k))  # updating the passive capital
        for cal_id in range(n_cal):
            cal_pp_k = my_cal(pp[n], cal_id, k)  # base prediction calibrated by the k-th calibrator
            for j_index in range(n_jumpers):
                a_weight[j_index, k] *= np.exp(-brier_loss(y_test[n], cal_pp_k, k))  # updating the active capital
        # Normalizing at each step (not needed):
        capital = p_weight + np.sum(a_weight[:, :])  # the overall weight
        p_weight /= capital  # normalization of the passive weight
        a_weight[:, :] /= capital  # normalization of the active weights

    p_prime[p_prime < 0] = 0
    p_prime[p_prime > 1] = 1

    cum_loss_base, roc_auc_base, cum_loss_prot, roc_auc_prot = calc_losses(p_pred, y_test, p_prime, pp, n_test, k)

    return p_prime, cum_loss_base, roc_auc_base, cum_loss_prot, roc_auc_prot


def make_binary(x_train, y_train, x_test, y_test, labels, ratio, permute_ratio):

    train_ind = np.concatenate(
        (np.where(y_train == labels[0])[0],
         np.random.choice(np.where(y_train == labels[1])[0], int(len(y_train[y_train == labels[1]]) / ratio)))
    )

    test_ind = np.concatenate(
        (np.where(y_test == labels[0])[0],
         np.random.choice(np.where(y_test == labels[1])[0], int(len(y_test[y_test == labels[1]]) / ratio)))
    )

    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    x_train = x_train[train_ind]
    y_train = y_train[train_ind]
    x_test = x_test[test_ind]
    y_test = y_test[test_ind]

    if permute_ratio != 0:
        np.random.shuffle(y_test[-int(len(y_test) / permute_ratio):])

    return x_train, y_train, x_test, y_test
