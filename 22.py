from typing import List, Any
from plotly.graph_objs import Scatter
import numpy as np
import array as arr
import math
import plotly.graph_objs as go
from more_itertools import powerset
from visualisation import Visualization

# прошу прощения, на данный момент моих знаний не хватает, чтобы выполнить эту работу,
# но обещаю заниматься усерднее, чтобы выполнить следующее задание(я понимаю что нужно зделать, но знаний как не хватает)
# дано
N = 10
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error


# тут обучение
def learning(design_matrix, t):
    return np.linalg.pinv(design_matrix.T @ design_matrix) @ design_matrix.T @ t


functions = [lambda x: np.sin(x), lambda x: np.cos(x), lambda x: np.log(x + 1e-7), lambda x: np.exp(x),
             lambda x: np.sqrt(x), lambda x: x, lambda x: x ** 2, lambda x: x ** 3]
indexes = [0, 1, 2, 3, 4, 5, 6, 7]
func_names = ["sin(x)", "cos(x)", "log(x + 1e-7)", "exp(x)", "sqrt(x)", "x", "x^2", "x^3"]
sets = list(powerset(indexes))[1:93]
#?
def M_F(x, ind):
    F = np.ones((1, len(x)))
    for i in ind:
        F = np.append(F, [functions[i](x)], axis=0)
    return F



    ind_prm = np.random.permutation(np.arange(N))
    train_ind = ind_prm[:int(tr * N)]
    valid_ind = ind_prm[int(tr * N):int((val + tr) * N)]
    test_ind = ind_prm[int((val + tr) * N):]
    x_train, t_train, x_valid, t_valid, x_test, t_test = x[train_ind], t[train_ind], x[valid_ind], t[valid_ind], x[
        test_ind], t[test_ind]


tr = 0.8, val = 0.1


# ошибка
def calculate_error(t, W, design_matrix):
    return (1 / 2) * sum((t - (W @ design_matrix.T)) ** 2)


visualisation = Visualization()
visualisation.models_error_scatter_plot(min_error_valid, min_error_test, names, 'title', show=True, save=True,
                                        name="2",
                                        path2save="C:/Users/26067/PycharmProjects/ML_HW2")
