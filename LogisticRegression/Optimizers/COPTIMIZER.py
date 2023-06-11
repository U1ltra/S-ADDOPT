################################################################################################################################
##---------------------------------------------------Centralized Optimizers---------------------------------------------------##
################################################################################################################################

import numpy as np
import copy as cp
import utilities as ut
import time
import math
from utilities import save_npy, save_state, load_state, plot_figure_data
from analysis import error


## Centralized gradient descent
def CGD(pr, learning_rate, K, theta_0):
    """
    Centralized SGD Optimizer with Full Training Data Batch.

    This is the ground truth model in our experiments.

    @param
    :pr                 logistic model object
    :learning_rate      learning rate
    :K                  number of epochs
    :theta_0            parameters of the logistic function

    @return
    :theta              list of logistic function parameters along the training
    :theta_opt          best logistic function parameters after the training (last round param)
    :F_opt              best logistic function value
    """
    start = time.time()
    theta_copy = cp.deepcopy(theta_0)
    theta = [theta_copy]

    for k in range(K):
        theta.append(theta[-1] - learning_rate * pr.grad(theta[-1]))
        ut.monitor("CGD", k, K)

    theta_opt = theta[-1]  # last set of parameters after training
    F_opt = pr.F_val(theta[-1])  # value of the objective function
    print(f"Time Span: {time.time() - start}")

    return theta, theta_opt, F_opt


def SGD(pr, learning_rate, K, theta_0, batch_size, lr_dec, save_path, exp_name, save_every):
    """
    Centralized mini-batch SGD Optimizer. This optimizer trains on all
    data globally in a batched manner.

    Note that the batch size and learning rate can affect the convergence
    greatly.

    @param
    :pr                 logistic model object
    :learning_rate      learning rate
    :K                  number of epochs
    :theta_0            parameters of the logistic function
    :batch_size         batch size of mini-batch SGD
    :save_path          path to save the experiment results
    :exp_name           name of the experiment
    :save_every         save the experiment results every save_every epochs

    @return
    :theta              list of logistic function parameters along the training
    :theta_opt          best logistic function parameters after the training (last round param)
    :F_opt              best logistic function value
    """
    theta_copy = cp.deepcopy(theta_0)
    theta = [theta_copy]

    update_round = math.ceil(pr.N / batch_size)  # in order to line up with RR case
    print(f"update round {update_round} | pr.N {pr.N}")

    start = time.time()
    for k in range(K):  # k local training rounds
        if lr_dec:
            learning_rate = 1 / ((k + 1)/100 + 2)

        temp = theta[-1]
        # gradient updates happening in one local training round
        for i in range(update_round):
            permutation = np.random.permutation(pr.N)
            temp = temp - learning_rate * pr.grad(
                temp, permute=permutation[0:batch_size], permute_flag=True
            )

        theta.append(temp)
        ut.monitor("SGD", k, K)
        if k % save_every == 0:
            # save_state(theta, save_path, exp_name)
            error_lr = error(pr, theta[-1], pr.F_val(theta[-1]))
            plot_figure_data(
                [error_lr.cost_gap_path(theta)],
                ["-vb"],
                [f"{exp_name}{k}"],
                f"{save_path}/{exp_name}{k}.pdf",
                100,
            )

    print(f"{k} Round | {update_round}# Updates | {batch_size} Batch Size")
    print(f"Time Span: {time.time() - start}")
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    return theta, theta_opt, F_opt


def C_RR(pr, learning_rate, K, theta_0, batch_size, lr_dec, save_path, exp_name, save_every):
    """
    Centralized Random Reshuflling Optimizer.

    @param
    :pr                 logistic model object
    :learning_rate      learning rate
    :K                  number of epochs
    :theta_0            parameters of the logistic function
    :batch_size         batch size of RR

    @return
    :theta              list of logistic function parameters along the training
    :theta_opt          best logistic function parameters after the training (last round param)
    :F_opt              best logistic function value
    """
    theta_copy = cp.deepcopy(theta_0)
    theta = [theta_copy]

    start = time.time()
    for k in range(K):
        if lr_dec:
            learning_rate = 1 / ((k + 1)/100 + 2)

        cnt = 0
        permutation = np.random.permutation(pr.N)
        temp = theta[-1]
        while cnt < pr.N:
            temp = temp - learning_rate * pr.grad(
                temp, permute=permutation[cnt : cnt + batch_size], permute_flag=True
            )
            cnt = cnt + batch_size

        ut.monitor("C_RR", k, K)
        theta.append(temp)
        if k % save_every == 0:
            # save_state(theta, save_path, exp_name)
            error_lr = error(pr, theta[-1], pr.F_val(theta[-1]))
            plot_figure_data(
                [error_lr.cost_gap_path(theta)],
                ["-vb"],
                [f"{exp_name}{k}"],
                f"{save_path}/{exp_name}{k}.pdf",
                100,
            )
    print(f"{k} Round | {cnt / batch_size}# Updates | {batch_size} Batch Size")
    print(f"Time Span: {time.time() - start}")
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    return theta, theta_opt, F_opt


## Centralized gradient descent with momentum
def CNGD(pr, learning_rate, momentum, K, theta_0):
    theta = [theta_0]
    theta_aux = cp.deepcopy(theta_0)
    for k in range(K):
        grad = pr.grad(theta[-1])
        theta_aux_last = cp.deepcopy(theta_aux)
        theta_aux = theta[-1] - learning_rate * grad
        theta.append(theta_aux + momentum * (theta_aux - theta_aux_last))
        ut.monitor("CNGD", k, K)
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    return theta, theta_opt, F_opt


## Centralized stochastic gradient descent
def CSGD(pr, learning_rate, K, theta_0):
    N = pr.N
    theta = cp.deepcopy(theta_0)
    theta_epoch = [theta_0]
    for k in range(K):
        idx = np.random.randint(0, N)
        grad = pr.grad(theta, idx)
        theta -= learning_rate * grad
        if (k + 1) % N == 0:
            theta_epoch.append(cp.deepcopy(theta))
        ut.monitor("CSGD", k, K)
    return theta_epoch
