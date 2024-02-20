########################################################################################################################
####---------------------------------------------------Utilities----------------------------------------------------####
########################################################################################################################

## Used to pre-set networkx-class properties

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from analysis import error
import os
from graph import *
import glob


def monitor(name, current, total, start_time):
    if (current + 1) % (total / 10) == 0:
        print(name + " %d%% completed" % int(100 * (current + 1) / total), flush=True)
        print(f"Time Span: {time.time() - start_time}", flush=True)
        return time.time()
    else:
        return start_time


def nx_options():
    options = {
        "node_color": "skyblue",
        "node_size": 10,
        "edge_color": "grey",
        "width": 0.5,
        "arrows": False,
        "node_shape": "o",
    }
    return options


def save_npy(npys, root_path, exp_name):
    print("saving experiment results...")
    for i, array in enumerate(npys):
        np.save(f"{root_path}/{exp_name[i]}.npy", array)


def plot_figure_path(
    exp_save_path,
    exp_names,
    formats,
    legend,
    save_path,
    plot_every,
    mark_every,
    plot_first,
    smooth=False,
):
    print("plotting the figure...", flush=True)
    plt.clf()
    font = FontProperties()
    font.set_size(18)
    font2 = FontProperties()
    font2.set_size(10)
    plt.figure(3)

    for i, name in enumerate(exp_names):
        print(f"plotting {exp_save_path}/{name}...", flush=True)
        line = np.load(f"{exp_save_path}/{name}")
        if "consensus" in name:
            line = line[1:]  # remove the first element, since it is zero
        if smooth:
            line = smoother(line, window_len=5)

        if plot_first == -1:
            plot_first = len(line)
        xaxis = np.linspace(0, len(line) - 1, num=len(line), dtype=int)
        yaxis = [
            abs(point) for point in line[:plot_first:plot_every]
        ]  # the F_val could be negative
        if i >= len(formats):
            plt.plot(xaxis[:plot_first:plot_every], yaxis, markevery=mark_every)
        else:
            plt.plot(
                xaxis[:plot_first:plot_every], yaxis, formats[i], markevery=mark_every
            )

    plt.legend(legend, prop=font2)
    plt.grid(True)
    plt.yscale("log")
    plt.yticks([1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11])
    plt.tick_params(labelsize="large", width=3)
    plt.title("MNIST", fontproperties=font)
    plt.xlabel("Epochs", fontproperties=font)
    plt.ylabel("Optimality Gap", fontproperties=font)
    plt.savefig(save_path, format="pdf", dpi=4000, bbox_inches="tight")
    print("figure plotted...")


def plot_figure_data(data, formats, legend, save_path, plot_every):
    print("plotting the figure...", flush=True)
    plt.clf()
    mark_every = 1
    font = FontProperties()
    font.set_size(18)
    font2 = FontProperties()
    font2.set_size(10)
    plt.figure(3)

    for i, line in enumerate(data):
        xaxis = np.linspace(0, len(line) - 1, num=len(line), dtype=int)
        yaxis = [
            abs(point) for point in line[::plot_every]
        ]  # the F_val could be negative
        if i >= len(formats):
            plt.plot(xaxis[::plot_every], yaxis, markevery=mark_every)
        else:
            plt.plot(xaxis[::plot_every], yaxis, formats[i], markevery=mark_every)

    plt.legend(legend, prop=font2)
    plt.grid(True)
    plt.yscale("log")
    plt.yticks([1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11])
    plt.tick_params(labelsize="large", width=3)
    plt.title("MNIST", fontproperties=font)
    plt.xlabel("Epochs", fontproperties=font)
    plt.ylabel("Optimality Gap", fontproperties=font)
    plt.savefig(save_path, format="pdf", dpi=4000, bbox_inches="tight")
    print("figure plotted...")


def save_state(theta, save_path, exp_name):
    print("saving experiment results...")
    np.save(f"{save_path}/{exp_name}_theta", theta)


def load_state(save_path, exp_name, type="optimal"):
    print("loading experiment results...")
    if type == "path":
        theta = np.load(f"{save_path}/{exp_name}_theta_path.npy")
        return (
            theta,
            theta[-1],
        )  # return the state sequence and the last state (optimum)
    elif type == "optimal":
        theta = np.load(f"{save_path}/{exp_name}_theta_optimal.npy")
        return None, theta  # return the last state (optimum)
    elif type == "init":
        theta = np.load(save_path)
        return None, theta


def load_optimal(save_path, exp_name):
    return np.load(f"{save_path}/{exp_name}")


def loadPathAndPlot(save_path, exp_names, error_lr, plot_every):
    load_thetas = []
    print("loading theta training results...")
    for i, name in enumerate(exp_names):
        load_thetas.append(np.load(f"{save_path}/{name}_theta.npy"))
    gaps = [error_lr.cost_gap_path(theta) for theta in load_thetas]

    print("plotting error gap results...")
    plot_figure(
        gaps,
        ["-vb", "-^m", "-dy", "-sr", "-1k", "-2g", "-3C", "-4w"],
        [f"optimum{i}" for i in range(len(exp_names))],
        f"{save_path}/cost_gap_all.pdf",
        plot_every,
    )


def loadGapAndPlot(save_path, exp_names, legands, plot_every, fig_name):
    gaps = []
    print("loading error gap results...")
    for i, name in enumerate(exp_names):
        gaps.append(np.load(f"{save_path}/{name}.npy"))

    # gaps = [gap[:5000] for gap in gaps]

    print("plotting error gap results...")
    plot_figure(
        gaps,
        [
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
        ],
        legands,
        f"{save_path}/{fig_name}.pdf",
        plot_every,
    )


def load_optimum(pr, save_path, exp_name):
    theta, theta_opt = load_state(save_path, exp_name)
    error_lr = error(pr, theta[-1], pr.F_val(theta[-1]))
    return error_lr


def initDir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def convert_to_doubly_stochastic(matrix, max_iterations=100, tolerance=1e-6):
    """
    The Sinkhorn-Knopp algorithm.

    Converts the given matrix to a doubly stochastic matrix.
    The conversion is done in-place.

    @param matrix: The matrix to be converted.
    @param max_iterations: The maximum number of iterations to perform.
    @param tolerance: The tolerance to check convergence.
    @return: The given matrix, converted to doubly stochastic.
    """
    n = matrix.shape[0]  # Assuming matrix is a square matrix

    # Normalize rows and columns iteratively
    for _ in range(max_iterations):
        prev_matrix = matrix.copy()

        # Normalize rows
        row_sums = np.sum(matrix, axis=1)
        matrix /= row_sums[:, np.newaxis]

        # Normalize columns
        col_sums = np.sum(matrix, axis=0)
        matrix /= col_sums

        # Check convergence
        max_diff = np.max(np.abs(matrix - prev_matrix))
        if max_diff < tolerance:
            break

    return matrix


def spectral_norm(comm_matrix):
    if comm_matrix is None:
        return 1

    # ones matrix
    ones = np.ones(comm_matrix.shape)
    matrix = comm_matrix - ones / comm_matrix.shape[0]

    # Compute the matrix A^T * A
    ata = np.matmul(matrix.T, matrix)

    # Calculate the eigenvalues of A^T * A
    eigenvalues, _ = np.linalg.eig(ata)

    # Find the maximum eigenvalue
    max_eigenvalue = np.max(eigenvalues)

    # Calculate the spectral norm as the square root of the maximum eigenvalue
    spectral_norm = np.sqrt(max_eigenvalue)

    return spectral_norm


def print_matrix(matrix, name):
    if matrix is None:
        print("Solo graph")
        return
    print(f"{name} Matrix:")
    for row in matrix:
        for element in row:
            if element < 0:
                print(f"{element:.3f}", end=" ")
            else:
                print(f" {element:.3f}", end=" ")
        print()
    print()


def is_primitive(matrix, max_iterations=10000):
    n = matrix.shape[0]  # Assuming matrix is a square matrix
    power_counter = 1

    while power_counter <= max_iterations:
        power_matrix = np.linalg.matrix_power(matrix, power_counter)
        if np.any(power_matrix <= 0):
            return False
        elif np.all(power_matrix > 0):
            return True
        power_counter += 1

    return False


def convergence_analysis():
    lr_constant = False
    epoch = 1000

    m = {
        "centralized": 12000,
        "distributed": 75,
    }
    n = {
        "centralized": 1,
        "distributed": 16,
    }

    net_lambda = {
        "grid": 0.8629819803720482,
        "exp": 0.5000000000000003,
        "geo": 0.43760846131515274,
    }

    algos = ["SGD", "CRR", "DSGD", "DRR", "DSGT", "GT-RR"]

    if lr_constant:
        # Constant learning rate
        lr = 1 / 8000

        for algo in algos:
            if algo == "SGD":
                error_floor = lr / n["centralized"]
                print(f"{algo:<5}: Error floor: {error_floor}")

            elif algo == "CRR":
                error_floor = lr**2 * m["centralized"]
                print(f"{algo:<5}: Error floor: {error_floor}")

            elif algo == "DSGD":
                for net, lambd in net_lambda.items():
                    error_floor = lr / n["distributed"] + lr**2 / (1 - lambd) ** 2
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

            elif algo == "DRR":
                for net, lambd in net_lambda.items():
                    error_floor = lr**2 * m["distributed"] / (1 - lambd) ** 3
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

            elif algo == "DSGT":
                for net, lambd in net_lambda.items():
                    error_floor = (
                        lr / n["distributed"]
                        + lr**2 / (1 - lambd)
                        + lr**4 / (n["distributed"] * (1 - lambd) ** 4)
                    )
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

            elif algo == "GT-RR":
                for net, lambd in net_lambda.items():
                    error_floor = (
                        lr**2 * m["distributed"] / (1 - lambd)
                        + lr**4 * m["distributed"] ** 4 / (1 - lambd) ** 2
                    )
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

    else:
        # Decreasing learning rate
        lr = lambda t: 1 / (50 * t + 400)

        for algo in algos:
            if algo == "SGD":
                error_floor = 1 / (m["centralized"] * n["centralized"] * epoch)
                print(f"{algo:<5}: Error floor: {error_floor}")

            elif algo == "CRR":
                error_floor = np.log(epoch) / (m["centralized"] * epoch**2)
                print(f"{algo:<5}: Error floor: {error_floor}")

            elif algo == "DSGD":
                for net, lambd in net_lambda.items():
                    error_floor = 1 / (
                        m["distributed"] * n["distributed"] * epoch
                    ) + 1 / ((1 - lambd) ** 2 * m["distributed"] ** 2 * epoch**2)
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

            elif algo == "DRR":
                for net, lambd in net_lambda.items():
                    error_floor = 1 / ((1 - lambd) ** 3 * m["distributed"] * epoch**2)
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

            elif algo == "DSGT":
                for net, lambd in net_lambda.items():
                    error_floor = 1 / (
                        m["distributed"] * n["distributed"] * epoch
                    ) + 1 / ((1 - lambd) ** 3 * m["distributed"] ** 2 * epoch**2)
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

            elif algo == "GT-RR":
                for net, lambd in net_lambda.items():
                    error_floor = 1 / (
                        (1 - lambd) * m["distributed"] * epoch**2
                    ) + 1 / ((1 - lambd) ** 2 * epoch**4)
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")


def try_geo(save_path):
    start = time.time()

    np.random.seed(0)
    nodes_list = [8, 16, 32]
    matrices_lambda = []
    search_space = int(1e5)

    print("generating matrices...", flush=True)
    for node_num in nodes_list:
        print(f"node_num = {node_num}", flush=True)
        mat = []
        for i in range(search_space):
            undir_graph = Geometric_graph(node_num).undirected(0.8)

            communication_matrix = Weight_matrix(undir_graph).row_stochastic()
            communication_matrix = convert_to_doubly_stochastic(
                communication_matrix, int(1e4), 1e-7
            )

            norm = spectral_norm(communication_matrix)
            mat.append((node_num, communication_matrix, norm))

            if i % (search_space / 10) == 0:
                print(f"{i / search_space * 100}% completed", flush=True)

        matrices_lambda.append(mat)
    print("time elapsed: ", time.time() - start, flush=True)

    print("searching for min distance... (1)", flush=True)
    min_dist = 1e10
    min_dist_tup = None
    min_dist_list = []
    for idx1, mat1 in enumerate(matrices_lambda[0]):
        for idx2, mat2 in enumerate(matrices_lambda[1]):
            dist = abs(mat1[2] - mat2[2])
            if dist < min_dist:
                min_dist = dist
                min_dist_tup = (mat1, mat2)
                min_dist_list.append((mat1, mat2))
        if idx1 % (search_space / 10) == 0:
            print(f"{idx1 / search_space * 100}% completed", flush=True)

    print(f"min_dist = {min_dist} (1)", flush=True)
    print(f"lambdas = {min_dist_tup[0][2]}, {min_dist_tup[1][2]}", flush=True)
    print("time elapsed: ", time.time() - start, flush=True)

    print("searching for min distance... (2)", flush=True)
    min_dist = 1e10
    min_dist_tup = None
    min_dist_list_2 = []
    for idx3, mat3 in enumerate(matrices_lambda[2]):
        dist1 = abs(mat3[2] - min_dist_list[-1][0][2])
        dist2 = abs(mat3[2] - min_dist_list[-1][1][2])
        dist = min(dist1, dist2)
        if dist < min_dist:
            min_dist = dist
            min_dist_tup = (*min_dist_list[-1], mat3)
            min_dist_list_2.append(min_dist_tup)
        if idx3 % (search_space / 10) == 0:
            print(f"{idx3 / search_space * 100}% completed", flush=True)

    print(f"min_dist = {min_dist} (2)", flush=True)
    print(
        f"lambdas = {min_dist_tup[0][2]}, {min_dist_tup[1][2]}, {min_dist_tup[2][2]}",
        flush=True,
    )
    print("time elapsed: ", time.time() - start, flush=True)

    # save the closest 100 matrices
    print("saving matrices...", flush=True)
    if not os.path.exists(f"{save_path}/geo"):
        os.mkdir(f"{save_path}/geo")

    for idx, tup in enumerate(min_dist_list_2[-min(100, len(min_dist_list_2)) :]):
        np.save(f"{save_path}/geo/geo_{idx}_node{nodes_list[0]}", tup[0][1])
        np.save(f"{save_path}/geo/geo_{idx}_node{nodes_list[1]}", tup[1][1])
        np.save(f"{save_path}/geo/geo_{idx}_node{nodes_list[2]}", tup[2][1])

    # save the closest 100 lambdas
    print("saving lambdas...", flush=True)
    for idx, tup in enumerate(min_dist_list_2[-min(100, len(min_dist_list_2)) :]):
        np.save(f"{save_path}/geo/geo_{idx}_node{nodes_list[0]}_lambda", tup[0][2])
        np.save(f"{save_path}/geo/geo_{idx}_node{nodes_list[1]}_lambda", tup[1][2])
        np.save(f"{save_path}/geo/geo_{idx}_node{nodes_list[2]}_lambda", tup[2][2])

    print(f"Time elapsed: {time.time() - start}", flush=True)
    print("done", flush=True)


def init_comm_matrix(node_num, graph, load_path=None):
    """
    This function initializes the communication matrix

    :param node_num: number of nodes
    :param graph: type of graph (exponential, grid, geometric, fully_connected, erdos_renyi)
    :param load_path: path to load the communication matrix
    """
    if graph == "solo":
        return None

    if load_path is None:
        if graph == "exponential":
            undir_graph = Exponential_graph(
                node_num
            ).undirected()  # generate the undirected graph
        elif graph == "grid":
            undir_graph = Grid_graph(int(math.sqrt(node_num))).undirected()
        elif graph == "geometric":
            undir_graph = Geometric_graph(node_num).undirected(0.8)
        elif graph == "ring":
            undir_graph = Ring_graph(node_num).undirected()
        elif graph == "fully_connected":
            undir_graph = Fully_connected_graph(node_num).undirected()
        elif graph == "erdos_renyi":
            undir_graph = Erdos_Renyi_graph(node_num, 0.1).undirected()
        else:
            raise ValueError(
                "graph must be exponential, grid, geometric, fully_connected, or erdos_renyi"
            )

        if graph in ["exponential", "grid", "geometric", "ring", "fully_connected"]:
            communication_matrix = Weight_matrix(undir_graph).row_stochastic()
            communication_matrix = convert_to_doubly_stochastic(
                communication_matrix, int(1e4), 1e-7
            )
        elif graph in ["erdos_renyi"]:
            communication_matrix = Weight_matrix(undir_graph).metroplis_weights()

    else:
        communication_matrix = np.load(load_path)
        print(f"loaded communication matrix from {load_path}")

    return communication_matrix


def is_doubly_stochastic(matrix):
    """
    Checks if the given matrix is doubly stochastic.

    @param matrix: The matrix to be checked.
    @return: True if the matrix is doubly stochastic, False otherwise.
    """
    n = matrix.shape[0]  # Assuming matrix is a square matrix

    # Check if all row sums are 1
    row_sums = np.sum(matrix, axis=1)
    if not np.allclose(row_sums, np.ones(n)):
        print("row_sums: ", row_sums)
        return False

    # Check if all column sums are 1
    col_sums = np.sum(matrix, axis=0)
    if not np.allclose(col_sums, np.ones(n)):
        print("col_sums: ", col_sums)
        return False

    return True


def fix_lambda_transformation(comm_matrix, lambd, max_iterations=2, tolerance=1e-6):
    """
    This function transforms the communication matrix to a new communication matrix with a given lambda

    :param comm_matrix: the communication matrix
    :param lambd: the given lambda
    :return: the new communication matrix
    """
    # ones matrix
    ones = np.ones(comm_matrix.shape)
    matrix = comm_matrix - ones / comm_matrix.shape[0]

    # Calculate the eigenvalues of A^T * A
    eigenvalues, eigenvectors = np.linalg.eig(comm_matrix)
    # print_matrix(comm_matrix, "comm_matrix")
    # print("eigenvalues: ", eigenvalues)
    # print_matrix(eigenvectors, "eigenvectors")

    # generate new eigenvalues
    for i in range(max_iterations):
        new_eigenvalues = np.zeros(comm_matrix.shape[0])
        new_eigenvalues[0] = 1
        new_eigenvalues[1] = lambd
        for j in range(2, comm_matrix.shape[0]):
            if eigenvalues[j] > 0:
                bound = min(eigenvalues[j], lambd)
                new_eigenvalues[j] = np.random.uniform(0, bound)
            else:
                bound = max(eigenvalues[j], -lambd)
                new_eigenvalues[j] = np.random.uniform(bound, 0)

        # construct new matrix
        new_matrix = np.matmul(
            np.matmul(eigenvectors, np.diag(new_eigenvalues)), eigenvectors.T
        )
        # print(f"new_matrix is primitive after {i} iterations")
        # print_matrix(new_matrix, "new_matrix")
        # print("is_doubly_stochastic(new_matrix): ", is_doubly_stochastic(new_matrix))
        # print("is_primitive(new_matrix): ", is_primitive(new_matrix))
        # print()

        if is_primitive(new_matrix):
            print(f"new_matrix is primitive after {i} iterations")
            print_matrix(new_matrix, "new_matrix")
            print(
                "is_doubly_stochastic(new_matrix): ", is_doubly_stochastic(new_matrix)
            )
            print("is_primitive(new_matrix): ", is_primitive(new_matrix))
            print()

            return True, new_matrix

    return False, None


def model_converged(model, theta, threshold=1e-1, gap_type="theta"):
    """
    This function checks if the gradients have converged

    :param gradients: the gradients
    :return: True if the gradients have converged, False otherwise
    """
    if len(theta) < 20:
        return False

    tail = theta[-20:]
    if gap_type == "theta":
        diffs = [np.linalg.norm(tail[i] - tail[i + 1]) for i in range(len(tail) - 1)]
        if np.mean(diffs) < threshold:
            return True
    elif gap_type == "F":
        F_vals = [model.F_val(t) for t in tail]
        diffs = [abs(F_vals[i] - F_vals[i + 1]) for i in range(len(F_vals) - 1)]
        if np.mean(diffs) < threshold:
            return True
    else:
        raise ValueError("gap_type must be theta or F")

    return False


def smoother(x, window_len=11, window="flat"):
    """
    moving average smoothing
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ["flat"]:
        raise ValueError("Window is on of 'flat'")

    w = np.ones(window_len, "d") / window_len
    y = np.convolve(w, x, mode="valid")
    # TODO: what if distributed algorithms?

    return y


def gen_gap_names(exp_name_all, gap_type="theta"):
    """
    This function generates the names of the gap files
    """
    exp_names = []
    for exp_name in exp_name_all:
        items = exp_name.split("_")
        items[-1] = f"{gap_type}.npy"
        exp_names.append("_".join(items))

    return exp_names


def avg_gap(info_log_path, trial_num):
    """
    This function calculates the average gap
    """
    # pickle file
    import pickle

    with open(f"{info_log_path}/info_log.pkl", "rb") as f:
        pickle_file = pickle.load(f)

    gap_types = ["F", "theta1", "theta2", "grad1", "grad2", "consensus"]

    for type_idx, gap_type in enumerate(gap_types):
        for exp in pickle_file[f"trial1"][
            type_idx
        ]:  # exp is the name of the experiment. Different trails have the same exp name
            print(f"processing {exp}...", flush=True)
            gaps = []
            for trial in range(trial_num):
                gaps.append(np.load(f"{info_log_path}/trial{trial+1}/{exp}"))
            # average over all trials
            avg_gap = np.mean(gaps, axis=0)
            exp_name = exp.split("/")[-1]
            np.save(f"{info_log_path}/avg_{exp_name}", avg_gap)


def plot_avg(avg_path, gap_type="theta", lr_list=[1 / 50, 1 / 250, 1 / 1000]):
    # list files with .npy
    import glob
    import re
    from matplotlib.font_manager import FontProperties

    def custom_sort_key(f):
        order = {"SGD": 1, "CRR": 2, "DSGD": 3, "DRR": 4}
        prefix = re.match(r"\D+", f).group()  # Extract the non-numeric prefix
        return order.get(prefix, 5)

    font = FontProperties()
    font.set_size(18)
    font2 = FontProperties()
    font2.set_size(10)
    plt.figure(3)

    files = glob.glob(f"{avg_path}/*.npy")
    files.sort(key=custom_sort_key)
    gap_files = [file for file in files if gap_type in file]
    files = gap_files

    num_of_algos = 2
    bzs = []
    legends = []
    file_set_list = []
    for i in range(int(len(files) / num_of_algos)):
        bzs.append([])
        legends.append([])
        file_set_list.append([])

    num_of_bzs = len(bzs)
    for i, file in enumerate(files):
        algo_idx = i % num_of_bzs
        print(f"processing {file} idx {algo_idx}...", flush=True)

        if gap_type == "consensus":
            if "SGD" in file and "DSGD" not in file:
                os.remove(file)
                continue
            if "CRR" in file:
                os.remove(file)
                continue
        # get legend
        exp_name = file.split("/")[-1]
        items = exp_name.split("_")
        algo = items[1]
        bz = items[4][2:]
        lr = items[5][2:]
        if lr == "0.000000":
            lr = lr_list
        if len(items) == 8:
            cr = items[6][2:]
            legends[algo_idx].append(f"{algo} (bz={bz}, lr={lr}, cr={cr})")
        else:
            legends[algo_idx].append(f"{algo} (bz={bz}, lr={lr})")

        bzs[algo_idx].append(int(bz))
        file_set_list[algo_idx].append(file)

    # plot the figure
    for i, file_set in enumerate(file_set_list):
        plot_figure_path(
            avg_path,
            [file.split("/")[-1] for file in file_set],
            [
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
            ],
            legends[i],
            f"{avg_path}/avg_{gap_type}_bz{sorted(bzs[i])[0]}.pdf",
            1,
            10,
            -1,
            smooth=False,
        )

        # remove the avg files
        for file in file_set:
            os.remove(file)
            print(f"removed {file}")

        print("--------")


def printTime():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def remove_files():
    import shutil

    base_path = "/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/multi_trials"
    exp_list = [
        "exp2_nonconvex_grid",
        "exp3_nonconvex_exp",
        "exp4_convex_ring",
        "exp5_convex_grid",
        "exp6_convex_exp",
    ]
    trial_num = 5
    algo_list = [
        # "central_SGD",
        # "central_CRR",
        "DSGD",
        "DRR",
    ]

    for exp_idx, exp_name in enumerate(exp_list):
        for trial_idx in range(trial_num):
            for algo_idx, algo in enumerate(algo_list):
                dir_to_remove = (
                    f"{base_path}/{exp_name}/trial{trial_idx+1}/{algo}/training"
                )

                # print(dir_to_remove, " -> ", os.path.isdir(dir_to_remove))
                if os.path.exists(dir_to_remove) and os.path.isdir(dir_to_remove):
                    shutil.rmtree(dir_to_remove)
                    print(
                        f"Directory {dir_to_remove} and its contents removed successfully."
                    )
                else:
                    print(f"Directory {dir_to_remove} does not exist.")


def plot_by_config(
    node_num,
    epoch,
    bz,
    lr,
    lr_list,
    cr,
    gap_type,
    GT_path,
    base_path,
    save_path,
    plot_every=1,
    mark_every=10,
    plot_first=-1,
    skip_algos=[],
    prev_legends=None,
    format_offset=0,
    clear_fig=False,
):
    # list trials in directory
    files = glob.glob(f"{base_path}/trial*")
    trial_num = len(files)
    algo_list = [
        # ("central_SGD", "SGD"),
        ("central_CRR", "CRR"),
        # ("DSGD", "DSGD"),
        ("DRR", "DRR"),
        # ("GTRR", "GTRR"),
    ]
    # algo_abbrev = ["SGD", "CRR", "DSGD", "DRR", ] # "GTRR"
    gap_type_dict = {
        "nonconvex": ["grad1", "grad2", "consensus"],
        "convex": ["theta1", "theta2", "consensus"],
    }
    formats = [  # list of line formats for plotting
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
    legends = []

    print("plotting the figure...", flush=True)
    if clear_fig:
        plt.clf()

    font = FontProperties()
    font.set_size(18)
    font2 = FontProperties()
    font2.set_size(10)
    # plt.rc('text', usetex=True)
    plt.figure(3)

    for algo_idx, algo_pair in enumerate(algo_list):
        algo = algo_pair[0]
        algo_abbrev = algo_pair[1]

        if algo in skip_algos:
            continue

        gap_list = []
        if algo in ["central_SGD", "central_CRR"]:
            legends.append(f"{algo[-3:]}")
            # legends.append(f"{algo} (bz={bz*node_num}, lr={lr_list})")
            exp_name = (
                f"{algo_abbrev}_gap_epoch{epoch}_bz{bz*node_num}_lr{lr:.6f}_{gap_type}"
            )
        elif algo in ["DSGD", "DRR", "GTRR"]:
            legends.append(f"{algo} (cr={cr})")
            # legends.append(f"{algo} (bz={bz}, lr={lr_list}, cr={cr})")
            exp_name = (
                f"{algo_abbrev}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_{gap_type}"
            )

        for trial_idx in range(trial_num):
            load_path = f"{base_path}/trial{trial_idx+1}/{algo}/{exp_name}.npy"
            if algo == "GTRR":
                load_path = f"{GT_path}/trial{trial_idx+1}/DRR/DRR_{exp_name[5:]}.npy"
            gap_list.append(np.load(load_path))

        avg_gap = np.mean(gap_list, axis=0)
        if gap_type == "consensus":
            avg_gap = avg_gap[1:]
        xaxis = np.linspace(0, len(avg_gap) - 1, num=len(avg_gap), dtype=int)
        yaxis = [abs(point) for point in avg_gap]  # the F_val could be negative

        if plot_first == -1:
            plot_first = len(avg_gap)
        plt.plot(
            xaxis[:plot_first:plot_every],
            yaxis,
            formats[algo_idx+format_offset],
            markevery=mark_every,
        )

    if prev_legends is not None:
        prev_legends.extend(legends)
        legends = prev_legends
        print("legends: ", legends)

    plt.legend(legends, prop=font2)
    plt.grid(True)
    # plt.yscale("log")
    # plt.yticks([10, 0])
    plt.tick_params(labelsize="large", width=3)
    plt.title("CIFAR10", fontproperties=font)
    plt.xlabel("Epochs", fontproperties=font)
    plt.ylabel("Loss", fontproperties=font)
    plt.savefig(save_path, format="pdf", dpi=4000, bbox_inches="tight")
    print("figure plotted...")

    return legends


def plot_loss_at_epoch(
    at_epoch,
    node_num,
    epoch,
    bz,
    lr_list,
    cr,
    base_path,
    save_path,
    plot_every=1,
    mark_every=10,
    plot_first=-1,
    clear_fig=True,
):
    base_path_origion = base_path
    algo_list = [
        ("central_CRR", "CRR"),
        ("DRR", "DRR"),
    ]

    formats = [  # list of line formats for plotting
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
    legends = []

    print("plotting the figure...", flush=True)
    if clear_fig:
        plt.clf()

    font = FontProperties()
    font.set_size(18)
    font2 = FontProperties()
    font2.set_size(10)
    plt.figure(3)

    networks = [
        ("solo", "solo"),
        ("ring", "ring"),
        ("exponential", "exp"),
    ]

    curve_to_plot = []
    for i in range(len(networks) * len(algo_list)):
        curve_to_plot.append([])

    for net_idx, net_pair in enumerate(networks):
        base_path = f"{base_path_origion}/{net_pair[1]}"
        print(base_path)
        files = glob.glob(f"{base_path}/trial*")
        trial_num = len(files)

        for algo_idx, algo_pair in enumerate(algo_list):
            algo = algo_pair[0]
            algo_abbrev = algo_pair[1]
            if algo in ["central_SGD", "central_CRR"]:
                legends.append(f"{net_pair[1]} {algo[-3:]}")
            elif algo in ["DSGD", "DRR", "GTRR"]:
                legends.append(f"{net_pair[1]} {algo} (cr={cr})")

            for lr in lr_list:
                gap_list = []
                if algo in ["central_SGD", "central_CRR"]:
                    exp_name = f"{algo_abbrev}_gap_epoch{epoch}_bz{bz*node_num}_lr{lr:.6f}_loss"
                elif algo in ["DSGD", "DRR", "GTRR"]:
                    exp_name = (
                        f"{algo_abbrev}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_loss"
                    )

                for trial_idx in range(trial_num):
                    load_path = f"{base_path}/trial{trial_idx+1}/{algo}/{exp_name}.npy"
                    gap_list.append(np.load(load_path))

                avg_gap = np.mean(gap_list, axis=0)
                if algo == "DRR":
                    avg_gap = np.mean(avg_gap, axis=-1)
                print(f"{net_pair[1]} {algo_pair[1]} {lr} {avg_gap.shape} {net_idx} {algo_idx}")
                curve_to_plot[net_idx*len(algo_list)+algo_idx].append(avg_gap[at_epoch])

    curve_to_plot = np.array(curve_to_plot)
    print(f"curve_to_plot shape {curve_to_plot.shape}")
    print(curve_to_plot)

    if plot_first == -1:
        plot_first = len(curve_to_plot[0])

    for curve_idx in range(curve_to_plot.shape[0]):
        print(curve_idx)
        plt.plot(
            lr_list,
            curve_to_plot[curve_idx],
            formats[curve_idx],
            markevery=mark_every,
        )

    plt.legend(legends, prop=font2)
    plt.grid(True)
    plt.xscale("log")
    plt.xticks(lr_list)
    plt.tick_params(labelsize="large", width=3)
    plt.title("CIFAR10", fontproperties=font)
    plt.xlabel("Epochs", fontproperties=font)
    plt.ylabel("Loss", fontproperties=font)
    plt.savefig(save_path, format="pdf", dpi=4000, bbox_inches="tight")
    print("figure plotted...")

    pass


if __name__ == "__main__":
    gap_type_list = [
        # ("grad1", "grad1_pre_avg"),
        # ("grad2", "grad2_post_avg"),
        # ("theta1", "theta1_post_avg"),
        # ("theta2", "theta2_pre_avg"),
        # ("consensus", "consensus"),
        ("loss", "loss"),
    ]

    legends = None
    for gap_type, save_name in gap_type_list:
        for bz in [10]:
            for cr in [1]:
                for lr in [0.0001, 0.001, 0.001, 0.01, 0.1, 1]:
                    skip_algos = []
                    if cr != 1:
                        skip_algos.extend(["central_SGD", "central_CRR", "DSGD"])

                    legends = plot_by_config(
                        node_num=16,
                        epoch=30,
                        bz=bz,
                        lr=lr,
                        lr_list=0.001000,
                        cr=cr,
                        gap_type=gap_type,
                        GT_path=None,
                        base_path="/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/lr_speedup_charac/solo",
                        save_path=f"/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/lr_speedup_charac/solo/plots/{save_name}_bz{bz}_cr{cr}_lr{lr}_solo.pdf",
                        skip_algos=skip_algos,
                        prev_legends=legends,
                        format_offset=len(legends) if legends is not None else 0,
                        clear_fig=True,
                    )
                    legends = None

    for at_epoch in range(0, 30, 5):
        plot_loss_at_epoch(
            at_epoch,
            node_num=16,
            epoch=30,
            bz=10,
            lr_list=[0.0001, 0.001, 0.001, 0.01, 0.1, 1],
            cr=1,
            base_path="/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/lr_speedup_charac",
            save_path=f"/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/lr_speedup_charac/solo/plots/{save_name}_bz{bz}_cr{cr}_atEpoch{at_epoch}_solo.pdf",
            clear_fig=True,
        )
