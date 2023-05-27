########################################################################################################################
####----------------------------------------------Exponential Network-----------------------------------------------####
########################################################################################################################

## Generates all the plots to compare different algorithms over Exponential directed graphs using logistic regression.

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from graph import Weight_matrix, Geometric_graph, Exponential_graph
from analysis import error
from Problems.logistic_regression import LR_L2
from Problems.log_reg_cifar import LR_L4
from Optimizers import COPTIMIZER as copt
from Optimizers import DOPTIMIZER as dopt
from utilities import plot_figure, save_npy, save_state, load_state

########################################################################################################################
####----------------------------------------------MNIST Classification----------------------------------------------####
########################################################################################################################



"""
Data processing for MNIST
"""
node_num = 16  ## number of nodes
logis_model = LR_L2(
    node_num, limited_labels=False, balanced=True
)  ## instantiate the problem class
dim = logis_model.p  ## dimension of the model
L = logis_model.L  ## L-smooth constant
total_train_sample = logis_model.N  ## total number of training samples
avg_local_sample = logis_model.b  ## average number of local samples

"""
Initializing variables
"""
CEPOCH_base = 10000
DEPOCH_base = 10000
# depoch = 100

model_para_central = np.random.normal(0, 1, dim)
model_para_dis = np.random.normal(0, 1, (node_num, dim))
undir_graph = Exponential_graph(node_num).undirected()
communication_matrix = Weight_matrix(undir_graph).column_stochastic()

ground_truth_lr = 0.01 * 1 / L
C_lr = 0.01 / L
D_lr = 0.01 / L  ## selecting an appropriate step-size

line_formats = ["-vb", "-^m", "-dy", "-sr", "-1k", "-2g", "-3r", "-.kp", "-+c", "-xm", "-|y", "-_r"]
exp_log_path = "/home/ubuntu/Desktop/DRR/experiment/lr01"
ckp_load_path = "/home/ubuntu/Desktop/DRR/experiment/debug"
plot_every = 250
save_every = 5000

"""
Centralized solutions
"""
theta_CSGD_0 = None
if ckp_load_path is not None:
    theta_CSGD_0, theta_opt = load_state(ckp_load_path, "optimum")
theta_CSGD_0 = None
error_lr_0 = error(logis_model, theta_opt, logis_model.F_val(theta_opt))


def CSGD_check():
    all_res_F_SGD = []
    batch_sizes = [50, 100, 200]
    for bz in batch_sizes:
        lr = C_lr * (bz/100)
        theta_SGD, theta_opt, F_opt = copt.SGD(
            logis_model,
            lr,
            CEPOCH_base,
            model_para_central,
            bz,
            exp_log_path,
            f"CSGD_bz{bz}_check",
            save_every,
        )
        res_F_SGD = error_lr_0.cost_gap_path(theta_SGD)
        all_res_F_SGD.append(res_F_SGD)

    # original code CGD path
    theta_CGD, theta_opt, F_opt = copt.CGD(
        logis_model, ground_truth_lr, CEPOCH_base, model_para_central
    )
    res_F_SGD = error_lr_0.cost_gap_path(theta_CGD)
    all_res_F_SGD.append(res_F_SGD)

    exp_save_path = f"{exp_log_path}/central_SGD"
    if not os.path.exists(exp_save_path):
        os.mkdir(exp_save_path)

    save_npy(
        all_res_F_SGD,
        exp_save_path,
        [f"bz{bz}" for bz in batch_sizes] + ["CGD"],
    )
    plot_figure(
        all_res_F_SGD,
        line_formats,
        [f"bz = {bz}" for bz in batch_sizes] + ["CGD"],
        f"{exp_save_path}/convergence_SGD.pdf",
        plot_every,
    )


def CRR_check():
    all_res_F_CRR = []
    batch_sizes = [50, 100, 200]
    for bz in batch_sizes:
        lr = C_lr * (bz/100)
        theta_CRR, theta_opt, F_opt = copt.C_RR(
            logis_model,
            lr,
            CEPOCH_base,
            model_para_central,
            bz,
            exp_log_path,
            f"CRR_bz{bz}_check",
            save_every,
        )
        res_F_CRR = error_lr_0.cost_gap_path(theta_CRR)
        all_res_F_CRR.append(res_F_CRR)

    exp_save_path = f"{exp_log_path}/central_CRR"
    if not os.path.exists(exp_save_path):
        os.mkdir(exp_save_path)

    save_npy(all_res_F_CRR, exp_save_path, [f"bz{bz}" for bz in batch_sizes])
    plot_figure(
        all_res_F_CRR,
        line_formats,
        [f"bz = {bz}" for bz in batch_sizes],
        f"{exp_save_path}/convergence_CRR.pdf",
        plot_every,
    )


"""
Decentralized Algorithms
"""


def DSGD_check():
    all_res_F_DSGD = []
    batch_sizes = [50, 100, 200]
    update_rounds = [int(logis_model.data_distr[0] / bz) for bz in batch_sizes]
    update_rounds = [1]
    exp_names = []
    legends = []

    for idx, bz in enumerate(batch_sizes):
        for comm_round in update_rounds:
            lr = D_lr * (bz/100)
            theta_D_SGD = dopt.D_SGD(
                logis_model,
                communication_matrix,
                lr,
                int(DEPOCH_base),
                model_para_dis,
                bz,
                comm_round,
                exp_log_path,
                f"DSGD_bz{bz}_ur{comm_round}",
                save_every,
            )

            res_F_D_SGD = error_lr_0.cost_gap_path(np.sum(theta_D_SGD, axis=1) / node_num)

            all_res_F_DSGD.append(res_F_D_SGD)
            exp_names.append(f"bz{bz}_ur{comm_round}")
            legends.append(f"bz = {bz}, ur = {comm_round}")

    exp_save_path = f"{exp_log_path}/DSGD"
    if not os.path.exists(exp_save_path):
        os.mkdir(exp_save_path)

    save_npy(all_res_F_DSGD, exp_save_path, exp_names)
    plot_figure(
        all_res_F_DSGD,
        line_formats,
        legends,
        f"{exp_save_path}/convergence_DSGD.pdf",
        plot_every,
    )


def DRR_check():
    all_res_F_DRR = []
    batch_sizes = [50, 100, 200]
    update_rounds = [int(logis_model.data_distr[0] / bz) for bz in batch_sizes]
    update_rounds = [1]
    exp_names = []
    legends = []

    for idx, bz in enumerate(batch_sizes):
        for comm_round in update_rounds:
            lr = D_lr * (bz/100)
            comm_round = int(comm_round)
            theta_D_RR = dopt.D_RR(
                logis_model,
                communication_matrix,
                lr,
                int(DEPOCH_base),
                model_para_dis,
                bz,
                comm_round,
                exp_log_path,
                f"DRR_bz{bz}_ur{comm_round}",
                save_every,
            )
            res_F_D_RR = error_lr_0.cost_gap_path(np.sum(theta_D_RR, axis=1) / node_num)

            all_res_F_DRR.append(res_F_D_RR)
            exp_names.append(f"bz{bz}_ur{comm_round}")
            legends.append(f"bz = {bz}, ur = {comm_round}")

    exp_save_path = f"{exp_log_path}/DRR"
    if not os.path.exists(exp_save_path):
        os.mkdir(exp_save_path)

    save_npy(all_res_F_DRR, exp_save_path, exp_names)
    plot_figure(
        all_res_F_DRR,
        line_formats,
        legends,
        f"{exp_save_path}/convergence_DRR.pdf",
        plot_every,
    )


CSGD_check()
CRR_check()
DSGD_check()
DRR_check()
