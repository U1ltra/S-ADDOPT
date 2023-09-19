################################################################################################################################
##---------------------------------------------------Centralized Optimizers---------------------------------------------------##
################################################################################################################################

import numpy as np
import copy as cp
import utilities as ut
import time
import math
from utilities import (
    save_npy,
    save_state,
    load_state,
    plot_figure_data,
    model_converged,
)
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
        ut.monitor("CGD", k, K, start)

    theta_opt = theta[-1]  # last set of parameters after training
    F_opt = pr.F_val(theta[-1])  # value of the objective function
    print(f"Time Span: {time.time() - start}")

    return theta, theta_opt, F_opt


def SGD(
    pr,
    learning_rate,
    K,
    theta_0,
    batch_size,
    lr_dec,
    lr_staged,
    save_path,
    exp_name,
    save_every,
    error_lr_0,
    stop_at_converge=False,
    lr_list=None,
    lr_dec_epochs=None,
    node_num=None,
):
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
    if lr_staged:
        lr_idx = 0
        learning_rate = lr_list[lr_idx]
        lr_change_round = 0

    update_round = math.ceil(pr.N / batch_size)  # in order to line up with RR case
    print(f"update round {update_round} | pr.N {pr.N}")

    start = time.time()
    track_time = start
    for k in range(K):  # k local training rounds
        if lr_dec:
            assert lr_staged is False
            learning_rate = 1 / (50 * k + 400)

        if lr_dec_epochs is None:
            if lr_staged and lr_idx < len(lr_list) - 1 and k - lr_change_round > 20:
                assert lr_list is not None
                assert lr_dec is False
                if model_converged(pr, theta):
                    lr_idx += 1
                    learning_rate = lr_list[lr_idx]
                    lr_change_round = k
                    print(f"Learning Rate Decreased to {learning_rate} at {k} round")
        else:
            if k in lr_dec_epochs:
                assert lr_list is not None
                assert lr_dec is False
                lr_idx = lr_dec_epochs.index(k) + 1
                learning_rate = lr_list[lr_idx]
                print(
                    f"Learning Rate Decreased to {learning_rate} at {k} round - {lr_idx}th decrease"
                )

        temp = theta[-1]
        # gradient updates happening in one local training round
        for it in range(node_num):
            for i in range(update_round):
                permutation = np.random.permutation(pr.N)
                grad = pr.grad(
                    temp,
                    permute=permutation[0:batch_size],
                    permute_flag=True,
                )
                temp = temp - learning_rate * grad

        theta.append(temp)

        ut.monitor("SGD", k, K, track_time)
        if save_every != -1 and ((k + 1) % save_every == 0 or k + 1 == K):
            # save_state(theta, save_path, exp_name)
            error_lr = error(pr, theta[-1], pr.F_val(theta[-1]))
            plot_figure_data(
                [error_lr.cost_gap_path(theta, gap_type="F")],
                ["-vb"],
                [f"{exp_name}{k}"],
                f"{save_path}/{exp_name}{k}.pdf",
                100,
            )

        if stop_at_converge:
            cost_path = error_lr_0.cost_gap_path(theta, gap_type="theta")
            if cost_path[-1] < 1e-1:
                print(f"Converged at {k} round")
                return theta, theta[-1], pr.F_val(theta[-1])

    print(f"{k} Round | {update_round}# Updates | {batch_size} Batch Size")
    print(f"Time Span: {time.time() - start}")
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    return theta, theta_opt, F_opt


def C_RR(
    pr,
    learning_rate,
    K,
    theta_0,
    batch_size,
    lr_dec,
    lr_staged,
    save_path,
    exp_name,
    save_every,
    error_lr_0,
    stop_at_converge=False,
    lr_list=None,
    lr_dec_epochs=None,
    node_num=None,
):
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
    if lr_staged:
        lr_idx = 0
        learning_rate = lr_list[lr_idx]
        lr_change_round = 0

    start = time.time()
    track_time = start
    for k in range(K):
        if lr_dec:
            assert lr_staged is False
            learning_rate = 1 / (50 * k + 400)

        if lr_dec_epochs is None:
            if lr_staged and lr_idx < len(lr_list) - 1 and k - lr_change_round > 20:
                assert lr_list is not None
                assert lr_dec is False
                if model_converged(pr, theta):
                    lr_idx += 1
                    learning_rate = lr_list[lr_idx]
                    lr_change_round = k
                    print(f"Learning Rate Decreased to {learning_rate} at {k} round")
        else:
            if k in lr_dec_epochs:
                assert lr_list is not None
                assert lr_dec is False
                lr_idx = lr_dec_epochs.index(k) + 1
                learning_rate = lr_list[lr_idx]
                print(
                    f"Learning Rate Decreased to {learning_rate} at {k} round - {lr_idx}th decrease"
                )

        temp = theta[-1]
        for it in range(node_num):
            cnt = 0
            permutation = np.random.permutation(pr.N)
            while cnt < pr.N:
                grad = pr.grad(
                    temp,
                    permute=permutation[cnt : cnt + batch_size],
                    permute_flag=True,
                )
                temp = temp - learning_rate * grad
                cnt = cnt + batch_size

        theta.append(temp)

        ut.monitor("C_RR", k, K, track_time)
        if save_every != -1 and ((k + 1) % save_every == 0 or k + 1 == K):
            # save_state(theta, save_path, exp_name)
            error_lr = error(pr, theta[-1], pr.F_val(theta[-1]))
            plot_figure_data(
                [error_lr.cost_gap_path(theta, gap_type="F")],
                ["-vb"],
                [f"{exp_name}{k}"],
                f"{save_path}/{exp_name}{k}.pdf",
                100,
            )

        if stop_at_converge:
            cost_path = error_lr_0.cost_gap_path(theta, gap_type="theta")
            if cost_path[-1] < 1e-1:
                print(f"Stop at Convergence: Converged at {k} round")
                return theta, theta[-1], pr.F_val(theta[-1])

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
