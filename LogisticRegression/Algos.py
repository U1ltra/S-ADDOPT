########################################################################################################################
####----------------------------------------------Exponential Network-----------------------------------------------####
########################################################################################################################

## Generates all the plots to compare different algorithms over Exponential directed graphs using logistic regression.

import os
import time
import copy as cp
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
from utilities import (
    plot_figure_path,
    save_npy,
    save_state,
    load_state,
    initDir,
    load_optimal,
)


def centralized_algo(
    logis_model,
    model_para,
    node_num,
    C_lr,
    C_lr_dec,
    C_lr_list,
    C_lr_dec_epochs,
    C_batch_size,
    CEPOCH_base,
    exp_name,
    exp_log_path,
    save_every,
    error_lr,
    line_formats,
    plot_every,
    mark_every,
    plot_first,
    train_load,
    load_init_theta,
    init_theta_path,
    save_theta_path,
    stop_at_convergence,
    algo,
    gap_type,
    use_smoother,
):
    exp_save_path = f"{exp_log_path}/central_{algo}"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    if load_init_theta:
        _, model_para_cp = load_state(init_theta_path, "", type="init")
    else:
        model_para_cp = cp.deepcopy(model_para)

    lr_staged = C_lr_list is not None
    if lr_staged:
        assert C_lr is None, "No learning rate when staged learning rate is specified"

    params = []
    if lr_staged:
        for idx, bz in enumerate(C_batch_size):
            params.append((CEPOCH_base[idx], bz, 0))
    else:
        for idx, bz in enumerate(C_batch_size):
            for idx_lr, lr in enumerate(C_lr):
                params.append((CEPOCH_base[idx], bz, lr))

    exp_names = []
    legends = []
    for idx, (epoch, bz, lr) in enumerate(params):
        exp_names.append(f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_{gap_type}.npy")
        if lr_staged:
            legends.append(f"{algo}: bz = {bz}, lr = {C_lr_list}")
        else:
            legends.append(f"{algo}: bz = {bz}, lr = {lr}")
        model_para = model_para_cp
        print(f"\n{'-'*50}")
        print(f"Running {algo} with epoch = {epoch}, batch size = {bz}, lr = {lr}")
        print(f"{'-'*50}")

        if os.path.exists(f"{exp_save_path}/{exp_names[-1]}"):
            print(f"Already exists {exp_save_path}/{exp_names[-1]}")
            continue
        if train_load:
            model_para = load_optimal(
                exp_save_path, f"{algo}_opt_theta_epoch{epoch}_bz{bz}_lr{lr:.6f}.npy"
            )

        if algo == "SGD":
            theta, theta_opt, F_opt = copt.SGD(
                logis_model,
                lr,
                epoch,
                model_para,
                bz,
                C_lr_dec,
                lr_staged,
                train_log_path,
                f"{algo}_bz{bz}_lr_staged_check"
                if lr_staged
                else f"{algo}_bz{bz}_lr{lr:.3f}_check",
                save_every,
                error_lr,
                stop_at_converge=stop_at_convergence,
                lr_list=C_lr_list,
                lr_dec_epochs=C_lr_dec_epochs,
                node_num=node_num,
            )
        elif algo == "CRR":
            theta, theta_opt, F_opt = copt.C_RR(
                logis_model,
                lr,
                epoch,
                model_para,
                bz,
                C_lr_dec,
                lr_staged,
                train_log_path,
                f"{algo}_bz{bz}_lr_staged_check"
                if lr_staged
                else f"{algo}_bz{bz}_lr{lr:.3f}_check",
                save_every,
                error_lr,
                stop_at_converge=stop_at_convergence,
                lr_list=C_lr_list,
                lr_dec_epochs=C_lr_dec_epochs,
                node_num=node_num,
            )

        F_loss = logis_model.F_val(np.array(theta))
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_loss.npy",
            F_loss,
        )

        np.save(
            f"{exp_save_path}/{algo}_opt_theta_epoch{epoch}_bz{bz}_lr{lr:.6f}.npy",
            theta_opt,
        )

        res_F = error_lr.cost_gap_path(theta, gap_type="theta")
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_theta1.npy",
            res_F,
        )
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_theta2.npy",
            res_F,
        )
        res_F_F = error_lr.cost_gap_path(theta, gap_type="F")
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_F.npy", res_F_F
        )
        res_F_grad = error_lr.cost_gap_path(theta, gap_type="grad")
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_grad1.npy",
            res_F_grad,
        )
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_grad2.npy",
            res_F_grad,
        )
        res_F_concensus = error_lr.cost_gap_path(theta, gap_type="consensus")
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_consensus.npy",
            res_F_concensus,
        )

        if save_theta_path:
            np.save(
                f"{exp_save_path}/{algo}_theta_epoch{epoch}_bz{bz}_lr{lr:.6f}.npy",
                theta,
            )

    if not os.path.exists(f"{exp_save_path}/convergence_{algo}_theta1_{exp_name}.pdf"):
        plot_figure_path(
            exp_save_path,
            [
                f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_theta1.npy"
                for idx, (epoch, bz, lr) in enumerate(params)
            ],
            line_formats,
            legends,
            f"{exp_save_path}/convergence_{algo}_theta1_{exp_name}.pdf",
            plot_every,
            mark_every,
            plot_first,
            use_smoother,
        )
    if not os.path.exists(f"{exp_save_path}/convergence_{algo}_F_{exp_name}.pdf"):
        plot_figure_path(
            exp_save_path,
            [
                f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_F.npy"
                for idx, (epoch, bz, lr) in enumerate(params)
            ],
            line_formats,
            legends,
            f"{exp_save_path}/convergence_{algo}_F_{exp_name}.pdf",
            plot_every,
            mark_every,
            plot_first,
            use_smoother,
        )
    if not os.path.exists(f"{exp_save_path}/convergence_{algo}_grad1_{exp_name}.pdf"):
        plot_figure_path(
            exp_save_path,
            [
                f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_grad1.npy"
                for idx, (epoch, bz, lr) in enumerate(params)
            ],
            line_formats,
            legends,
            f"{exp_save_path}/convergence_{algo}_grad1_{exp_name}.pdf",
            plot_every,
            mark_every,
            plot_first,
            use_smoother,
        )
    if not os.path.exists(
        f"{exp_save_path}/convergence_{algo}_consensus_{exp_name}.pdf"
    ):
        plot_figure_path(
            exp_save_path,
            [
                f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_consensus.npy"
                for idx, (epoch, bz, lr) in enumerate(params)
            ],
            line_formats,
            legends,
            f"{exp_save_path}/convergence_{algo}_consensus_{exp_name}.pdf",
            plot_every,
            mark_every,
            plot_first,
            use_smoother,
        )

    exp_names = [f"central_{algo}/{name}" for name in exp_names]
    return exp_names, legends


def decentralized_algo(
    logis_model,
    model_para,
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
    error_lr,
    line_formats,
    plot_every,
    mark_every,
    plot_first,
    train_load,
    load_init_theta,
    init_theta_path,
    save_theta_path,
    stop_at_convergence,
    algo,
    gap_type,
    use_smoother,
    comm_every_epoch,
):
    exp_save_path = f"{exp_log_path}/{algo}"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    if load_init_theta:
        _, model_para_cp = load_state(init_theta_path, "", type="init")
        model_para_cp = np.array([model_para_cp for i in range(logis_model.n)])
    else:
        model_para_cp = cp.deepcopy(model_para)

    lr_staged = D_lr_list is not None
    if lr_staged:
        assert D_lr is None, "No learning rate when staged learning rate is specified"

    params = []
    if lr_staged:
        for idx, bz in enumerate(D_batch_size):
            for cr in communication_rounds:
                params.append((DEPOCH_base[idx], bz, 0, cr))
    else:
        for idx, bz in enumerate(D_batch_size):
            for lr in D_lr:
                for cr in communication_rounds:
                    params.append((DEPOCH_base[idx], bz, lr, cr))

    exp_names = []
    legends = []
    for idx, (epoch, bz, lr, cr) in enumerate(params):
        exp_names.append(
            f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_{gap_type}.npy"
        )
        if lr_staged:
            legends.append(f"{algo}: bz = {bz}, ur = {cr}, lr = {D_lr_list}")
        else:
            legends.append(f"{algo}: bz = {bz}, ur = {cr}, lr = {lr}")
        model_para = model_para_cp
        print(f"\n{'-'*50}")
        print(
            f"Running {algo} with epoch = {epoch}, batch size = {bz}, lr = {lr}, ur = {cr}"
        )
        print(f"{'-'*50}")

        if os.path.exists(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_{gap_type}.npy"
        ):
            print(
                f"Already exists {exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_{gap_type}.npy"
            )
            continue
        if train_load:
            model_para = load_optimal(
                exp_save_path, f"{algo}_opt_theta_bz{bz}_lr{lr:.6f}_ur{cr}.npy"
            )

        if algo == "DSGD":
            theta_D = dopt.D_SGD(
                logis_model,
                communication_matrix,
                lr,
                int(epoch),
                model_para,
                bz,
                cr,
                D_lr_dec,
                lr_staged,
                grad_track,
                train_log_path,
                f"{algo}_bz{bz}_ur{cr}_lr{lr}",
                save_every,
                error_lr,
                stop_at_converge=stop_at_convergence,
                comm_type=comm_type,
                lr_list=D_lr_list,
                lr_dec_epochs=D_lr_dec_epochs,
            )
        elif algo == "DRR":
            theta_D = dopt.D_RR(
                logis_model,
                communication_matrix,
                lr,
                int(epoch),
                model_para,
                bz,
                cr,
                D_lr_dec,
                lr_staged,
                grad_track,
                train_log_path,
                f"{algo}_bz{bz}_ur{cr}_lr{lr}",
                save_every,
                error_lr,
                stop_at_converge=stop_at_convergence,
                comm_type=comm_type,
                lr_list=D_lr_list,
                lr_dec_epochs=D_lr_dec_epochs,
                exact_diff=exact_diff,
                comm_every_epoch=comm_every_epoch,
            )

        F_loss = logis_model.F_val(np.array(theta_D))
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_loss.npy",
            F_loss,
        )

        res_F_D = error_lr.cost_gap_path(theta_D, gap_type="theta")
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_theta1.npy",
            res_F_D,
        )
        res_F_D = error_lr.cost_gap_path(
            np.sum(theta_D, axis=1) / logis_model.n, gap_type="theta"
        )
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_theta2.npy",
            res_F_D,
        )

        res_F_D_F = error_lr.cost_gap_path(
            np.sum(theta_D, axis=1) / logis_model.n, gap_type="F"
        )
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_F.npy",
            res_F_D_F,
        )
        res_F_D_grad = error_lr.cost_gap_path(
            np.sum(theta_D, axis=1) / logis_model.n, gap_type="grad"
        )
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_grad1.npy",
            res_F_D_grad,
        )
        res_F_D_grad = error_lr.cost_gap_path(theta_D, gap_type="grad")
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_grad2.npy",
            res_F_D_grad,
        )
        res_F_D_concensus = error_lr.cost_gap_path(theta_D, gap_type="consensus")
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_consensus.npy",
            res_F_D_concensus,
        )

        if save_theta_path:
            np.save(
                f"{exp_save_path}/{algo}_theta_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}.npy",
                theta_D,
            )

    if not os.path.exists(f"{exp_save_path}/convergence_{algo}_theta1_{exp_name}.pdf"):
        plot_figure_path(
            exp_save_path,
            [
                f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_theta1.npy"
                for idx, (epoch, bz, lr, cr) in enumerate(params)
            ],
            line_formats,
            legends,
            f"{exp_save_path}/convergence_{algo}_theta1_{exp_name}.pdf",
            plot_every,
            mark_every,
            plot_first,
            use_smoother,
        )
    if not os.path.exists(f"{exp_save_path}/convergence_{algo}_theta2_{exp_name}.pdf"):
        plot_figure_path(
            exp_save_path,
            [
                f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_theta2.npy"
                for idx, (epoch, bz, lr, cr) in enumerate(params)
            ],
            line_formats,
            legends,
            f"{exp_save_path}/convergence_{algo}_theta2_{exp_name}.pdf",
            plot_every,
            mark_every,
            plot_first,
            use_smoother,
        )
    if not os.path.exists(f"{exp_save_path}/convergence_{algo}_F_{exp_name}.pdf"):
        plot_figure_path(
            exp_save_path,
            [
                f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_F.npy"
                for idx, (epoch, bz, lr, cr) in enumerate(params)
            ],
            line_formats,
            legends,
            f"{exp_save_path}/convergence_{algo}_F_{exp_name}.pdf",
            plot_every,
            mark_every,
            plot_first,
            use_smoother,
        )
    if not os.path.exists(f"{exp_save_path}/convergence_{algo}_grad1_{exp_name}.pdf"):
        plot_figure_path(
            exp_save_path,
            [
                f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_grad1.npy"
                for idx, (epoch, bz, lr, cr) in enumerate(params)
            ],
            line_formats,
            legends,
            f"{exp_save_path}/convergence_{algo}_grad1_{exp_name}.pdf",
            plot_every,
            mark_every,
            plot_first,
            use_smoother,
        )
    if not os.path.exists(f"{exp_save_path}/convergence_{algo}_grad2_{exp_name}.pdf"):
        plot_figure_path(
            exp_save_path,
            [
                f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_grad2.npy"
                for idx, (epoch, bz, lr, cr) in enumerate(params)
            ],
            line_formats,
            legends,
            f"{exp_save_path}/convergence_{algo}_grad2_{exp_name}.pdf",
            plot_every,
            mark_every,
            plot_first,
            use_smoother,
        )
    if not os.path.exists(
        f"{exp_save_path}/convergence_{algo}_consensus_{exp_name}.pdf"
    ):
        plot_figure_path(
            exp_save_path,
            [
                f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_consensus.npy"
                for idx, (epoch, bz, lr, cr) in enumerate(params)
            ],
            line_formats,
            legends,
            f"{exp_save_path}/convergence_{algo}_consensus_{exp_name}.pdf",
            plot_every,
            mark_every,
            plot_first,
            use_smoother,
        )

    exp_names = [f"{algo}/{name}" for name in exp_names]
    return exp_names, legends
