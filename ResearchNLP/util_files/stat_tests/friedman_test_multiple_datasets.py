def freedom_degree(alg_num, dataset_num):
    return (alg_num - 1) * (dataset_num - 1)


def friedman_chi_squared(freedom_deg, alg_num, avg_ranks):

    a = 12 * freedom_deg / float(alg_num * (alg_num + 1))
    b = sum(map(lambda a: a**2, avg_ranks)) - alg_num * ((alg_num + 1) ** 2) / 4.0
    return a*b


def friedman_significance(n, k, chi_sq):
    return ((n-1)*chi_sq) / (n*(k-1) - chi_sq)

import Orange
import matplotlib.pyplot as plt
names = ["first", "third", "second", "fourth" ]
avranks =  [1.9, 3.2, 2.8, 3.3 ]
cd = Orange.evaluation.scoring.compute_CD(avranks, 30) #tested on 30 datasets
Orange.evaluation.scoring.graph_ranks('/tmp/blabla.png', avranks, names, cd=cd, width=6, textspace=1.5)
plt.show()