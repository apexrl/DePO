import numpy as np
from scipy import stats

import os
import pandas as pd


def get_data(file_path, column="avg"):
    df = pd.read_csv(file_path)
    data = df[column]

    return data


def t_test(baseline, your_method):
    var_homo = stats.levene(baseline, your_method)
    flag = var_homo.pvalue > 0.05
    res = stats.ttest_ind(baseline, your_method, equal_var=flag)

    return res


baseline_name = ["BCO", "GAIfO", "GAIfO-DP"]  #
baselines = ["bco", "gailfo", "gailfo-dp"]  #
method_name = ["DPO (w.o. PG)", "DPO"]
methods = ["sl-lfo", "gail-lfo"]
envs = ["invdoublependulum", "hopper", "walker", "halfcheetah", "ant"]

if __name__ == "__main__":

    num_envs = len(envs)

    for i in range(num_envs):
        file_path = "../final_performance/{}/{}/res.csv".format(methods[0], envs[i])
        method_1 = get_data(file_path)
        file_path = "../final_performance/{}/{}/res.csv".format(methods[1], envs[i])
        method_2 = get_data(file_path)
        for idx, baseline in enumerate(baselines):
            file_path = "../final_performance/{}/{}/res.csv".format(baseline, envs[i])
            baseline_data = get_data(file_path)

            res_1 = t_test(baseline_data, method_1)
            res_2 = t_test(baseline_data, method_2)

            # print('env {} against {}:\t{}-{:.3},\t{}-{:.3}\n'.format(envs[i], baseline_name[idx], method_name[0], res_1.pvalue, method_name[1], res_2.pvalue))
            print(
                "env {} against {}: \t{}-{:.3}\n".format(
                    envs[i], baseline_name[idx], method_name[1], res_2.pvalue
                )
            )
