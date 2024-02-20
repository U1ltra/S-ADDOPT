################################################################################################################################
##---------------------------------------------------Decentralized Optimizers-------------------------------------------------##
################################################################################################################################

import numpy as np
import copy as cp
import utilities as ut
from numpy import linalg as LA
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


def D_SGD(
    prd,
    weight,
    learning_rate,
    K,
    theta_0,
    batch_size,
    comm_round,
    lr_dec,
    lr_staged,
    grad_track,
    save_path,
    exp_name,
    save_every,
    error_lr_0,
    stop_at_converge=False,
    comm_type="graph_avg",
    lr_list=None,
    lr_dec_epochs=None,
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
    if lr_staged:
        lr_idx = 0
        learning_rate = lr_list[lr_idx]
        lr_change_round = 0

    node_num = prd.n
    update_round = math.ceil(len(prd.X[0]) / batch_size)
    start = time.time()
    track_time = start

    grad_track_y = np.zeros(theta_0.shape)
    grad_prev = np.zeros(theta_0.shape)

    for k in range(K):
        temp = theta[-1]
        if lr_dec:
            assert lr_staged is False
            learning_rate = 1 / (50 * k + 400)

        if lr_dec_epochs is None:
            if lr_staged and lr_idx < len(lr_list) - 1 and k - lr_change_round > 20:
                assert lr_list is not None
                assert lr_dec is False
                if model_converged(prd, theta):
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

                if comm_round > 0:
                    if (i + 1) % comm_round == 0:
                        # averaging from neighbours
                        # this probably caused significant performance drop
                        if comm_type == "graph_avg":
                            temp = np.matmul(weight, temp)
                        elif comm_type == "all_avg":
                            theta_avg = np.sum(temp, axis=0) / node_num
                            temp = np.array([theta_avg for i in range(node_num)])
                            raise NotImplementedError
                        elif comm_type == "no_comm":
                            pass
                        elif comm_type == "one_shot":
                            if (
                                k == K - 1
                                and i == update_round - 1
                                and node == node_num - 1
                            ):
                                temp = np.matmul(weight, temp)
                                print("One Shot Communication")
                        else:
                            raise NotImplementedError
                elif comm_round < 0:
                    for i in range(-comm_round):
                        temp = np.matmul(weight, temp)
                else:
                    raise ValueError

                if stop_at_converge:
                    cost_path = error_lr_0.cost_gap_path(temp, gap_type="theta")
                    if cost_path[-1] < 1e-1:
                        print(f"Converged at {k} round")
                        return theta, theta[-1], prd.F_val(theta[-1])

        ut.monitor("D_SGD", k, K, track_time)
        theta.append(cp.deepcopy(temp))

        if save_every != -1 and ((k + 1) % save_every == 0 or k + 1 == K):
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
    lr_staged,
    grad_track,
    save_path,
    exp_name,
    save_every,
    error_lr_0,
    stop_at_converge=False,
    comm_type="graph_avg",
    lr_list=None,
    lr_dec_epochs=None,
    exact_diff=False,
    comm_every_epoch=False,
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
    if lr_staged:
        lr_idx = 0
        learning_rate = lr_list[lr_idx]
        lr_change_round = 0

    node_num = prd.n
    update_round = math.ceil(len(prd.X[0]) / batch_size)
    start = time.time()
    track_time = start

    for k in range(K):
        temp = theta[-1]
        if grad_track or exact_diff:
            grad_track_y = np.zeros(theta_0.shape)
            grad_prev = np.zeros(theta_0.shape)
        if exact_diff:
            theta_prev = cp.deepcopy(temp)

        if lr_dec:
            assert lr_staged is False
            learning_rate = 1 / (50 * k + 400)

        if lr_dec_epochs is None:
            if lr_staged and lr_idx < len(lr_list) - 1 and k - lr_change_round > 20:
                assert lr_list is not None
                assert lr_dec is False
                if model_converged(prd, theta):
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

        # sample_vec = [
        #         np.random.permutation(prd.data_distr[i]) for i in range(prd.n)
        #     ] # fix the sample vector for all iteration over the number of nodes
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
                elif exact_diff:
                    if round == 0:
                        temp = temp - learning_rate * grad
                    else:
                        temp = (
                            2 * temp - theta_prev - learning_rate * (grad - grad_prev)
                        )
                    if np.any(np.isnan(temp)):
                        print(f"epoch {k} | node {node} | round {round} | theta {temp}")
                        print(f"nan at {k} round")
                        raise ValueError

                    theta_prev = cp.deepcopy(temp)
                    grad_prev = cp.deepcopy(grad)
                else:
                    temp = temp - learning_rate * grad

                if not comm_every_epoch:
                    if comm_round > 0:
                        if (round + 1) % comm_round == 0:
                            # averaging from neighbours
                            if comm_type == "graph_avg":
                                temp = np.matmul(weight, temp)
                            elif comm_type == "all_avg":
                                theta_avg = np.sum(temp, axis=0) / node_num
                                temp = np.array([theta_avg for i in range(node_num)])
                                raise NotImplementedError
                            elif comm_type == "no_comm":
                                pass
                            elif comm_type == "one_shot":
                                if (
                                    k == K - 1
                                    and round == update_round - 1
                                    and node == node_num - 1
                                ):
                                    temp = np.matmul(weight, temp)
                                    print("One Shot Communication")
                            else:
                                raise NotImplementedError
                    elif comm_round < 0:
                        for i in range(-comm_round):
                            temp = np.matmul(weight, temp)
                    else:
                        raise ValueError

                if stop_at_converge:
                    cost_path = error_lr_0.cost_gap_path(temp, gap_type="theta")
                    if cost_path[-1] < 1e-1:
                        print(f"Converged at {k} round")
                        return theta, theta[-1], prd.F_val(theta[-1])
            
        if comm_every_epoch:
            if comm_round > 0:
                if comm_type == "graph_avg":
                    temp = np.matmul(weight, temp)
                else:
                    raise ValueError
            elif comm_round < 0:
                for i in range(-comm_round):
                    temp = np.matmul(weight, temp)

        ut.monitor("D_RR", k, K, track_time)
        theta.append(cp.deepcopy(temp))

        if save_every != -1 and ((k + 1) % save_every == 0 or k + 1 == K):
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
