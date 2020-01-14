import pickle
from itertools import product
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from functools import reduce

if __name__ == "__main__":

    video_root = Path("/Users/xwang169/Downloads/videos")
    video_path = Path(os.path.join(video_root, "16PlasticLinitis1.mpg"))
    result_path = Path(str(video_path)[:-4])

    """Finding combination orders"""
    skips = np.arange(10, 21, 3).tolist()  # [10] 21
    winsizes = np.arange(10, 21, 3).tolist()  # [10] 21
    r_thresholds = np.arange(0.2, 0.8, 0.05).tolist()  #
    products = []
    for i, j, k in product(skips, r_thresholds, winsizes):
        products.append([i, j, k])
    """Add in EX"""
    r_thresholds = np.arange(0.9, 1.5, 0.1).tolist()  #
    for i, j, k in product(skips, r_thresholds, winsizes):
        products.append([i, j, k])  # My product was at Index No.284

    """Loading"""
    difs, std_1s, std_2s = [], [], []
    with open("difs.dat", "rb") as f:
        difs = np.array(pickle.load(f))
    with open("std_1s.dat", "rb") as f1:
        std_1s = np.array(pickle.load(f1))
    with open("std_2s.dat", "rb") as f2:
        std_2s = np.array(pickle.load(f2))
    """Loading EX"""
    with open("difs_EX.dat", "rb") as f:
        difs_EX = np.array(pickle.load(f))
        difs = np.concatenate((difs, difs_EX))
    with open("std_1s_EX.dat", "rb") as f1:
        std_1s_EX = np.array(pickle.load(f1))
        std_1s = np.concatenate((std_1s, std_1s_EX))
    with open("std_2s_EX.dat", "rb") as f2:
        std_2s_EX = np.array(pickle.load(f2))
        std_2s = np.concatenate((std_2s, std_2s_EX))

    comparisons = 3
    topn = 160  # 2: dif vs. std_sum | 3: dif vs. std_1 vs. std_2
    """
        Calculate sums to find an overall rating of a set of hyper-parameter
        Find matching most dif and least std_sum for best combo
    """
    if comparisons == 2:
        std_sums = std_1s + std_2s
        std_sums_sort = sorted(std_sums)
        difs_sort = sorted(difs, reverse=True)
        std_sums_topn, difs_topn = [], []
        for i in range(topn):
            std_sums_topn.append(np.where(std_sums == std_sums_sort[i]))
            difs_topn.append(np.where(difs == difs_sort[i]))
        intersects = np.intersect1d(std_sums_topn, difs_topn)
        best_combos = np.array(products)[intersects]

        """Details of best_combos"""
        overall_max = max(max(difs), max(std_sums))
        overall_min = min(min(difs), min(std_sums))
        file1 = open("Comp_2_ways_top{}.txt".format(topn), "w")
        for j in intersects:
            std_sum = std_sums[j]
            std_sum_rank = np.where(std_sums_sort == std_sum)
            dif = difs[j]
            dif_rank = np.where(difs_sort == dif)
            result = "Combo {}: std_sum: {:.2f} with rank: {} | dif: {:.2f} with rank: {} | " \
                     "Hyper-parameters: skips: {}, winsize: {}, residual_threshold: {:.2f}\n".\
                format(j, std_sum, std_sum_rank[0][0], dif, dif_rank[0][0],
                       products[j][0], products[j][2], products[j][1])
            plt.vlines(x=j, ymin=dif, ymax=overall_max, colors='gold', linestyles='dotted')
            plt.vlines(x=j, ymin=overall_min, ymax=std_sum, colors='purple', linestyles='dotted')
            print(result)
            file1.writelines(result)
        file1.close()

        """Plotting"""
        plt.plot(np.arange(len(difs)), difs)
        plt.plot(np.arange(len(std_sums)), std_sums)
        # for i, j in zip(difs_topn, std_sums_topn):
        #     plt.vlines(x=i, ymin=difs[i], ymax=max(max(difs), max(std_sums)), colors='purple', linestyles='dotted')
        #     plt.vlines(x=j, ymin=0, ymax=std_sums[j], colors='green', linestyles='dotted')
        plt.legend(['difs', 'sums'], loc='upper left')
        ax1 = plt.gca()
        ax1.set_xlabel(r"Iterations")
        plt.savefig(str(result_path / "Comp_2_ways.png"))
        plt.show()
        plt.close()

    if comparisons == 3:
        std_1s_sort = sorted(std_1s)
        std_2s_sort = sorted(std_2s)
        difs_sort = sorted(difs, reverse=True)
        std_1s_topn, std_2s_topn, difs_topn = [], [], []
        for i in range(topn):
            std_1s_topn.append(np.where(std_1s == std_1s_sort[i]))
            std_2s_topn.append(np.where(std_2s == std_2s_sort[i]))
            difs_topn.append(np.where(difs == difs_sort[i]))
        intersects = reduce(np.intersect1d, (std_1s_topn, std_2s_topn, difs_topn))
        best_combos = np.array(products)[intersects]

        """Details of best_combos"""
        overall_max = max(max(difs), max(std_1s), max(std_2s))
        overall_min = min(min(difs), min(std_1s), min(std_2s))
        file2 = open("Comp_3_ways_top{}.txt".format(topn), "w")
        for j in intersects:
            std_1 = std_1s[j]
            std_1_rank = np.where(std_1s_sort == std_1)
            std_2 = std_2s[j]
            std_2_rank = np.where(std_2s_sort == std_2)
            dif = difs[j]
            dif_rank = np.where(difs_sort == dif)
            result = "Combo {}: std_1: {:.2f} with rank: {} | std_2: {:.2f} with rank: {} | dif: {:.2f} with rank: {}" \
                     " | Hyper-parameters: skips: {}, winsize: {}, residual_threshold: {:.2f}\n".\
                format(j, std_1, std_1_rank[0][0], std_2, std_2_rank[0][0], dif, dif_rank[0][0],
                       products[j][0], products[j][2], products[j][1])
            plt.vlines(x=j, ymin=dif, ymax=overall_max, colors='gold', linestyles='dotted')
            plt.vlines(x=j, ymin=overall_min, ymax=std_1, colors='purple', linestyles='dotted')
            plt.vlines(x=j, ymin=std_2, ymax=overall_max/2+overall_min/2, colors='red', linestyles='dotted')
            print(result)
            file2.writelines(result)
        file2.close()

        """Plotting"""
        plt.plot(np.arange(len(difs)), difs)
        plt.plot(np.arange(len(std_1s)), std_1s)
        plt.plot(np.arange(len(std_2s)), std_2s)
        # for i, j, k in zip(difs_topn, std_1s_topn, std_2s_topn):
        #     plt.vlines(x=i, ymin=difs[i], ymax=overall_max, colors='gold', linestyles='dotted')
        #     plt.vlines(x=j, ymin=overall_min, ymax=std_1s[j], colors='purple', linestyles='dotted')
        #     plt.vlines(x=j, ymin=std_2s[j], ymax=overall_max/2+overall_min/2, colors='red', linestyles='dotted')
        plt.legend(['difs', 'std_1s', 'std_2s'], loc='upper left')
        ax1 = plt.gca()
        ax1.set_xlabel(r"Iterations")
        plt.savefig(str(result_path / "Comp_3_ways.png"))
        plt.show()
        plt.close()
