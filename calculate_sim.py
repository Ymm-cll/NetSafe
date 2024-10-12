# from scipy.stats import kendalltau
#
# ranking1 = {}
# ranking2 = {}
# ranking1["fact"] = [1, 3, 4, 5, 2]
# ranking1["csqa"] = [1, 2, 4, 5, 3]
# ranking1["gsm8k"] = [4, 3, 5, 2, 1]
#
# ranking2["ne"] = [5, 3, 4, 2, 1]
# ranking2["ec"] = [[5, 3, 2, 1, 4], [5, 4, 2, 1, 3]]
# ranking2["apv"] = [[5, 3, 1, 2, 4], [4, 3, 1, 2, 5]]
# tau, p_value = kendalltau(ranking1, ranking2)
#
# print(f"Kendall's Tau: {tau}")
# print(f"P-value: {p_value}")
from scipy.stats import kendalltau
import numpy as np

ranking1 = {
    "fact": [1, 3, 4, 5, 2],
    "csqa": [1, 2, 4, 5, 3],
    "gsm8k": [4, 3, 5, 2, 1]
}

ranking2 = {
    "ne": [5, 3, 4, 2, 1],
    "ec": [[5, 3, 2, 1, 4], [5, 4, 2, 1, 3]],
    "apv": [[1, 3, 5, 4, 2], [2, 3, 5, 4, 1]]
}


def calculate_kendall_tau(ranking1, ranking2):
    results = {}

    for key1, ranks1 in ranking1.items():
        if key1 not in results:
            results[key1] = []
        for key2, ranks2 in ranking2.items():
            if isinstance(ranks2[0], list):
                tau_values = []
                for rank in ranks2:
                    tau, _ = kendalltau(ranks1, rank)
                    tau_values.append(tau)
                average_tau = np.mean(tau_values)
                results[key1].append((key2, average_tau))
            else:
                tau, _ = kendalltau(ranks1, ranks2)
                results[key1].append((key2, tau))

    return results


kendall_results = calculate_kendall_tau(ranking1, ranking2)

for key, values in kendall_results.items():
    print(f"{key}:")
    for rank_key, tau_value in values:
        print(f"  {rank_key}: {tau_value:.4f}")
