"""train.py

Launch the training process of the logistic regression problem on given graph network.
"""

import os
import pytz
import time
import math
import copy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.font_manager import FontProperties
from graph import (
    Weight_matrix,
    Geometric_graph,
    Exponential_graph,
    Grid_graph,
    Fully_connected_graph,
)
from analysis import error
from Problems.logistic_regression import LR_L2
from Problems.log_reg_cifar import LR_L4
from Optimizers import COPTIMIZER as copt
from Optimizers import DOPTIMIZER as dopt
from utilities import (
    load_state,
    plot_figure_path,
    spectral_norm,
    print_matrix,
    init_comm_matrix,
    gen_gap_names,
    printTime,
)
from Algos import centralized_algo, decentralized_algo

np.random.seed(0)
trial_num = 2
info_log = dict()

np.set_printoptions(threshold=np.inf)

for trial_idx in range(trial_num):

    """
    Data processing for MNIST
    """
    node_num = 16  ## number of nodes
    # node_num = int(input("Enter number of nodes: "))
    C_node_num = node_num

    # LR_L2: MNIST, LR_L4: CIFAR
    logis_model = LR_L4(
        node_num, limited_labels=False, balanced=True, class1=0, class2=9, # nonconvex=True, 
    )  ## instantiate the problem class
    dim = logis_model.p  ## dimension of the model
    L = logis_model.L  ## L-smooth constant
    total_train_sample = logis_model.N  ## total number of training samples
    avg_local_sample = logis_model.b  ## average number of local samples

    """
    Initializing variables
    """
    model_para_central = np.random.normal(
        0, 1, dim
    )  # initialize the model parameter for central algorithms
    model_para_dis = np.array([cp.deepcopy(model_para_central) for i in range(node_num)])

    graph = "ring"  # "ring", "grid", "exponential", "geometric", "erdos_renyi", "fully_connected"
    # f"/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/gen_graphs/geo/geo_7_node{node_num}.npy"
    # f"/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/comm_matrix/comm_matrix_{node_num}.npy"  
    comm_load_path = None
    communication_matrix = init_comm_matrix(node_num, graph, comm_load_path)
    communication_rounds = [  # TODO: one shot communication/averaging
        1, 5, 10, 25, 125, -2, -5
    ]  # list of number of communication rounds for decentralized algorithms experiments
    comm_type = "graph_avg"  # "graph_avg", "all_avg", "one_shot", "no_comm", 

    C_algos = ["SGD", "CRR"]  # "SGD", "CRR"
    D_algos = ["DSGD", "DRR"]  # "DSGD", "DRR"

    CEPOCH_base = [150 for i in range(5)]  # number of epochs for central algorithms. should have the same length as C_batch_size
    DEPOCH_base = [150 for i in range(5)]  # number of epochs for decentralized algorithms

    # [0.001]
    C_lr = [0.001]  # list of learning rate for central algorithms experiments
    D_lr = [0.001]  # list of learning rate for decentralized algorithms experiments

    # only used for nonconvex case
    # [1 / 50, 1 / 250, 1 / 1000]
    C_lr_list = None  # list of learning rate for central algorithms experiments. should have the same length as C_batch_size
    D_lr_list = None # list of learning rate for decentralized algorithms experiments
    # [80, 120]
    C_lr_dec_epochs = None
    D_lr_dec_epochs = None

    bz_list = [10]
    C_batch_size = [bz*node_num for bz in bz_list]  # list of batch size for central algorithms experiments
    D_batch_size = [bz for bz in bz_list]  # list of batch size for decentralized algorithms experiments

    C_lr_dec = False  # whether to decay the learning rate for central algorithms
    D_lr_dec = False  # whether to decay the learning rate for decentralized algorithms
    C_train_load = (
        False  # whether to load the previous model parameter for central algorithms
    )
    D_train_load = (
        False  # whether to load the previous model parameter for decentralized algorithms
    )
    C_stop_at_converge = False  # whether to stop the training when the model converges for central algorithms
    D_stop_at_converge = False  # whether to stop the training when the model converges for decentralized algorithms
    save_theta_path = False  # whether to save the model parameter training path for central and decentralized algorithms
    grad_track = False  # whether to track the gradient norm for decentralized algorithms
    exact_diff = False  # exact diffusion
    load_init_theta = (
        False  # whether to load the initial model parameter from pretrained model
    )
    use_smoother = False  # whether to use the smoothing technique for nonconvex case
    gap_type = "grad2"  # "F", "theta1", "theta2", "grad1", "grad2", "consensus"

    line_formats = [  # list of line formats for plotting
        "-vb",
        "-^m",
        "-dy",
        "-sr",
        "-1k",
        "-2g",
        "-3r",
        "-.kp",
        "-+c",
        "-xm",
        "-|y",
        "-_r",
    ]
    dir_name = "multi_trials"
    exp_name = "exp11_1_convex_cr_ring"
    exp_log_path = f"/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/{dir_name}/{exp_name}/trial{trial_idx+1}"  # path to save the experiment results
    ckp_load_path = "/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/optimum"  # path to load the optimal model parameter
    opt_name = "convex"  # "convex", "nonconvex2"
    # init_theta_path = "/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/init_param/CRR_opt_theta_init.npy"  # path to load the initial model parameter
    init_theta_path = "/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/nonconvex_opt/opt4_opt3_test_F_theta_convergence/exp1/central_SGD/SGD_opt_theta_epoch200_bz10000_lr1.000000.npy"  # path to load the initial model parameter
    plot_every = 1  # plot points
    mark_every = 10  # mark interval
    save_every = -1 # int(CEPOCH_base[0] / 5) if CEPOCH_base[0] > 5 else 1  # save interval
    plot_first = -1  # plot the first 1000 epochs

    """
    Optimum solution
    """
    if ckp_load_path is not None:
        theta_CSGD_0, theta_opt = load_state(ckp_load_path, opt_name, "optimal")
    else:
        print("No optimal solution loaded. Use random optimal solution.")
        theta_opt = model_para_central
        theta_CSGD_0 = None

    error_lr_0 = error(
        logis_model, theta_opt, logis_model.F_val(theta_opt)
    )  # instantiate the error class


    if not os.path.exists(f"{exp_log_path}"):
        os.makedirs(f"{exp_log_path}")

    newYorkTz = pytz.timezone("America/New_York")
    timeInNewYork = datetime.now(newYorkTz)
    currentTimeInNewYork = timeInNewYork.strftime("%H:%M:%S")
    print("The current time in New York is:", currentTimeInNewYork)

    print(f"{'-'*50}")
    print(f"{'-'*50}")
    if logis_model.nonconvex:
        print(f"{'Nonconvex'}")
    else:
        print(f"{'Convex'}")
    print(f"{'-'*50}")
    print(f"{'-'*50}")
    print(f"trial idx = {trial_idx+1}")
    print(f"{graph} Graph")
    print(f"node num = {node_num}")
    print(f"c_node_num = {C_node_num}")
    print(f"dim = {dim}")
    print(f"L = {L}")
    print(f"total train sample = {total_train_sample}")
    print(f"avg local sample {avg_local_sample}")
    print(f"CEPOCH base = {CEPOCH_base}")
    print(f"DEPOCH base {DEPOCH_base}")
    print(f"communication rounds = {communication_rounds}")
    print(f"communication type = {comm_type}")
    print_matrix(communication_matrix, "communication matrix")
    print(f"spec norm = {spectral_norm(communication_matrix)}")
    print(f"C lr = {C_lr}")
    print(f"D lr = {D_lr}")
    print(f"C lr list = {C_lr_list}")
    print(f"D lr list = {D_lr_list}")
    print(f"C lr dec epochs = {C_lr_dec_epochs}")
    print(f"D lr dec epochs = {D_lr_dec_epochs}")
    print(f"C batch size = {C_batch_size}")
    print(f"D batch size = {D_batch_size}")
    print(f"C lr dec = {C_lr_dec}")
    print(f"D lr dec = {D_lr_dec}")
    print(f"C train load = {C_train_load}")
    print(f"D train load = {D_train_load}")
    print(f"C stop at converge = {C_stop_at_converge}")
    print(f"D stop at converge = {D_stop_at_converge}")
    print(f"save theta path = {save_theta_path}")
    print(f"grad track = {grad_track}")
    print(f"exact diff = {exact_diff}")
    print(f"load init theta = {load_init_theta}")
    print(f"use_smoother = {use_smoother}")
    print(f"gap type = {gap_type}")

    print(f"exp name = {exp_name}")
    print(f"exp log path = {exp_log_path}")
    print(f"ckp load path = {ckp_load_path}")
    print(f"init theta path = {init_theta_path}")
    print(f"plot every = {plot_every}")
    print(f"mark every = {mark_every}")
    print(f"save every = {save_every}")
    print(f"plot first = {plot_first}")


    print(f"{'-'*50}", flush=True)
    start = time.time()

    exp_name_all = []
    legend_all = []

    for algo in C_algos:
        exp_names, legends = centralized_algo(
            logis_model,
            model_para_central,
            C_node_num,
            C_lr,
            C_lr_dec,
            C_lr_list,
            C_lr_dec_epochs,
            C_batch_size,
            CEPOCH_base,
            exp_name,
            exp_log_path,
            save_every,
            error_lr_0,
            line_formats,
            plot_every,
            mark_every,
            plot_first,
            C_train_load,
            load_init_theta,
            init_theta_path,
            save_theta_path,
            C_stop_at_converge,
            algo,
            gap_type,
            use_smoother,
        )
        exp_name_all.extend(exp_names)
        legend_all.extend(legends)
        printTime()

    for algo in D_algos:
        exp_names, legends = decentralized_algo(
            logis_model,
            model_para_dis,
            D_lr,
            D_lr_dec,
            D_lr_list,
            D_lr_dec_epochs,
            D_batch_size,
            DEPOCH_base,
            communication_matrix,
            communication_rounds,
            comm_type,
            grad_track,
            exact_diff,
            exp_name,
            exp_log_path,
            save_every,
            error_lr_0,
            line_formats,
            plot_every,
            mark_every,
            plot_first,
            D_train_load,
            load_init_theta,
            init_theta_path,
            save_theta_path,
            D_stop_at_converge,
            algo,
            gap_type,
            use_smoother,
        )
        exp_name_all.extend(exp_names)
        legend_all.extend(legends)
        printTime()
    print(f"{'-'*50}", flush=True)

    info_log[f"trial{trial_idx+1}"] = []
    for item in ["F", "theta1", "theta2", "grad1", "grad2", "consensus"]:
        gap_names = gen_gap_names(exp_name_all, item)
        info_log[f"trial{trial_idx+1}"].append(gap_names)
        plot_figure_path(
            exp_log_path,
            gap_names,
            line_formats,
            legend_all,
            f"{exp_log_path}/cost_gap_all_{exp_name}_{item}.pdf",
            plot_every,
            mark_every,
            plot_first,
            smooth=use_smoother,
        )
    import pickle
    with open(f"{exp_log_path}/info_log.pkl", "wb") as f:
        pickle.dump(info_log, f)

    end = time.time()
    print(f"{'-'*50}")
    print(f"Total time: {end-start:.2f} seconds")
    print(f"{'-'*50}", flush=True)

newYorkTz = pytz.timezone("America/New_York")
timeInNewYork = datetime.now(newYorkTz)
currentTimeInNewYork = timeInNewYork.strftime("%H:%M:%S")
print("\nThe current time in New York is:", currentTimeInNewYork)

# save info log using pickle


# load with pickle
# with open(f"{exp_log_path}/info_log.pkl", "rb") as f:
#     info_log = pickle.load(f)