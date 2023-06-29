################################################################################################################################
##---------------------------------------------------Decentralized Optimizers-------------------------------------------------##
################################################################################################################################

import numpy as np
import copy as cp
import utilities as ut
from numpy import linalg as LA
import time
import math
from utilities import save_npy, save_state, load_state, plot_figure_data
from analysis import error


def D_SGD(
    prd,
    weight,
    learning_rate,
    K,
    theta_0,
    batch_size,
    comm_round,
    lr_dec,
    grad_track,
    save_path,
    exp_name,
    save_every,
    error_lr_0,
    stop_at_converge=False,
    comm_type="graph_avg",
):
    """
    Distributed SGD Optimizer

    @param
    :prd                logistic model object
    :weight             the column stocastic weight matrix used to represent the graph network
    :learning_rate      learning rate
    :K                  number of epochs
    :theta_0            parameters of the logistic function (each row stands for one distributed node's param)
    :batch_size         batch size of mini-batch SGD
    :comm_round         gradient info communication perioid

    @return
    :theta              list of logistic function parameters along the training
    """
    theta_copy = cp.deepcopy(theta_0)
    theta = [theta_copy]

    node_num = prd.n
    update_round = math.ceil(len(prd.X[0]) / batch_size)
    start = time.time()
    track_time = start

    grad_track_y = np.zeros(theta_0.shape)
    grad_prev = np.zeros(theta_0.shape)

    for k in range(K):
        temp = theta[-1]
        if lr_dec:
            learning_rate = 1 / ((k + 1) / 100 + 2)

        for node in range(node_num):
            for i in range(update_round):
                sample_vec = [
                    np.random.permutation(prd.data_distr[i]) for i in range(prd.n)
                ]
                sample_vec = [val[:batch_size] for i, val in enumerate(sample_vec)]
                grad = prd.networkgrad(temp, permute=sample_vec, permute_flag=True)

                if grad_track:
                    grad_track_y = np.matmul(weight, grad_track_y + grad - grad_prev)
                    grad_prev = cp.deepcopy(grad)
                    temp = temp - learning_rate * grad_track_y
                else:
                    temp = temp - learning_rate * grad

                if (i + 1) % comm_round == 0:
                    # averaging from neighbours
                    # this probably caused significant performance drop
                    if comm_type == "graph_avg":
                        temp = np.matmul(weight, temp)
                    elif comm_type == "all_avg":
                        theta_avg = np.sum(temp, axis=0) / node_num
                        temp = np.array([theta_avg for i in range(node_num)])
                    elif comm_type == "no_comm":
                        pass

                if stop_at_converge:
                    cost_path = error_lr_0.cost_gap_path(temp, gap_type="theta")
                    if cost_path[-1] < 1e-1:
                        print(f"Converged at {k} round")
                        return theta, theta[-1], prd.F_val(theta[-1])

        ut.monitor("D_SGD", k, K, track_time)
        theta.append(cp.deepcopy(temp))

        if k % save_every == 0 or k + 1 == K:
            # save_state(theta, save_path, exp_name)
            avg_theta = np.sum(theta[-1], axis=0) / prd.n
            error_lr = error(prd, avg_theta, prd.F_val(avg_theta))
            plot_figure_data(
                [error_lr.cost_gap_path(np.sum(theta, axis=1) / prd.n, gap_type="F")],
                ["-vb"],
                [f"{exp_name}{k}"],
                f"{save_path}/{exp_name}_{k}.pdf",
                100,
            )

    print(f"{k} Round | {update_round}# Updates | {batch_size} Batch Size")
    print(f"Time Span: {time.time() - start}")

    return theta


def D_RR(
    prd,
    weight,
    learning_rate,
    K,
    theta_0,
    batch_size,
    comm_round,
    lr_dec,
    grad_track,
    save_path,
    exp_name,
    save_every,
    error_lr_0,
    stop_at_converge=False,
    comm_type="graph_avg",
):
    """
    Distributed DRR Optimizer

    @param
    :prd                logistic model object
    :weight             the column stocastic weight matrix used to represent the graph network
    :learning_rate      learning rate
    :K                  number of epochs
    :theta_0            parameters of the logistic function (each row stands for one distributed node's param)
    :batch_size         batch size of mini-batch DRR
    :comm_round         gradient info communication perioid

    @return
    :theta_epoch        list of logistic function parameters along the training
    """
    theta_copy = cp.deepcopy(theta_0)
    theta = [theta_copy]

    node_num = prd.n
    update_round = math.ceil(len(prd.X[0]) / batch_size)
    start = time.time()
    track_time = start

    grad_track_y = np.zeros(theta_0.shape)
    grad_prev = np.zeros(theta_0.shape)

    for k in range(K):
        temp = theta[-1]
        if lr_dec:
            learning_rate = 1 / ((k + 1) / 100 + 2)

        for node in range(node_num):
            sample_vec = [
                np.random.permutation(prd.data_distr[i]) for i in range(prd.n)
            ]
            for round in range(update_round):
                permutes = [
                    val[round * batch_size : (round + 1) * batch_size]
                    for i, val in enumerate(sample_vec)
                ]

                grad = prd.networkgrad(temp, permute=permutes, permute_flag=True)

                if grad_track:
                    grad_track_y = np.matmul(weight, grad_track_y + grad - grad_prev)
                    grad_prev = cp.deepcopy(grad)
                    temp = temp - learning_rate * grad_track_y
                else:
                    temp = temp - learning_rate * grad

                if (round + 1) % comm_round == 0:
                    # averaging from neighbours
                    if comm_type == "graph_avg":
                        temp = np.matmul(weight, temp)
                    elif comm_type == "all_avg":
                        theta_avg = np.sum(temp, axis=0) / node_num
                        temp = np.array([theta_avg for i in range(node_num)])
                    elif comm_type == "no_comm":
                        pass

                if stop_at_converge:
                    cost_path = error_lr_0.cost_gap_path(temp, gap_type="theta")
                    if cost_path[-1] < 1e-1:
                        print(f"Converged at {k} round")
                        return theta, theta[-1], prd.F_val(theta[-1])

        ut.monitor("D_RR", k, K, track_time)
        theta.append(cp.deepcopy(temp))

        if k % save_every == 0 or k + 1 == K:
            # save_state(theta, save_path, exp_name)
            avg_theta = np.sum(theta[-1], axis=0) / prd.n
            error_lr = error(prd, avg_theta, prd.F_val(avg_theta))
            plot_figure_data(
                [error_lr.cost_gap_path(np.sum(theta, axis=1) / prd.n, gap_type="F")],
                ["-vb"],
                [f"{exp_name}{k}"],
                f"{save_path}/{exp_name}_{k}.pdf",
                100,
            )

    print(f"Time Span: {time.time() - start}")
    return theta


def DPG_RR():
    # DRR with different communication frequency
    pass


def SADDOPT(prd, B1, B2, learning_rate, K, theta_0):
    theta = cp.deepcopy(theta_0)
    theta_epoch = [cp.deepcopy(theta)]
    sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
    grad = prd.networkgrad(theta, sample_vec)
    tracker = cp.deepcopy(grad)
    Y = np.ones(B1.shape[1])
    for k in range(K):
        theta = np.matmul(B1, theta) - learning_rate * tracker
        grad_last = cp.deepcopy(grad)
        Y = np.matmul(B1, Y)
        YY = np.diag(Y)
        z = np.matmul(LA.inv(YY), theta)
        sample_vec = np.array(
            [np.random.choice(prd.data_distr[i]) for i in range(prd.n)]
        )
        grad = prd.networkgrad(z, sample_vec)
        tracker = np.matmul(B2, tracker) + grad - grad_last
        ut.monitor("SADDOPT", k, K)
        if (k + 1) % prd.b == 0:
            theta_epoch.append(cp.deepcopy(theta))
    return theta_epoch


def GP(prd, B, learning_rate, K, theta_0):
    theta = [cp.deepcopy(theta_0)]
    grad = prd.networkgrad(theta[-1])
    Y = np.ones(B.shape[1])
    for k in range(K):
        theta.append(np.matmul(B, theta[-1]) - learning_rate * grad)
        Y = np.matmul(B, Y)
        YY = np.diag(Y)
        z = np.matmul(LA.inv(YY), theta[-1])
        grad = prd.networkgrad(z)
        ut.monitor("GP", k, K)
    return theta


def ADDOPT(prd, B1, B2, learning_rate, K, theta_0):
    theta = [cp.deepcopy(theta_0)]
    grad = prd.networkgrad(theta[-1])
    tracker = cp.deepcopy(grad)
    Y = np.ones(B1.shape[1])
    for k in range(K):
        theta.append(np.matmul(B1, theta[-1]) - learning_rate * tracker)
        grad_last = cp.deepcopy(grad)
        Y = np.matmul(B1, Y)
        YY = np.diag(Y)
        z = np.matmul(LA.inv(YY), theta[-1])
        grad = prd.networkgrad(z)
        tracker = np.matmul(B2, tracker) + grad - grad_last
        ut.monitor("ADDOPT", k, K)
    return theta


def SGP(prd, B, learning_rate, K, theta_0):
    theta = cp.deepcopy(theta_0)
    theta_epoch = [cp.deepcopy(theta)]
    sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
    grad = prd.networkgrad(theta, sample_vec)
    Y = np.ones(B.shape[1])
    for k in range(K):
        theta = np.matmul(B, theta) - learning_rate * grad
        Y = np.matmul(B, Y)
        YY = np.diag(Y)
        z = np.matmul(LA.inv(YY), theta)
        sample_vec = np.array(
            [np.random.choice(prd.data_distr[i]) for i in range(prd.n)]
        )
        grad = prd.networkgrad(z, sample_vec)
        ut.monitor("SGP", k, K)
        if (k + 1) % prd.b == 0:
            theta_epoch.append(cp.deepcopy(theta))
    return theta_epoch
