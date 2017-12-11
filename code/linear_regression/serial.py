import sys
sys.path.insert(0,'..')

import csv
import math
import random
import seq_gen
import data_loader
import numpy as np
import matplotlib.pyplot as plt


def target_func(X, y, my_beta):
	tmp_vec = X.dot(my_beta) - y
	tmp_norm = np.linalg.norm(tmp_vec, ord=2)
	return tmp_norm * tmp_norm / 2


def optimal_solve(X, y, n):
	X_trans = X.transpose()
	tmp = X_trans.dot(X)
	tmp_inv = np.linalg.inv(tmp)
	opt = tmp_inv.dot(X_trans)
	opt = opt.dot(y)
	return target_func(X, y, opt)


def solve_sequential():
	X, y, n, dim = data_loader.get_data()
	X_trans = X.transpose()
	iteration_cnt = 10000

	opt_loss = optimal_solve(X, y, n)

	bt_beta = 0.5
	bt_alpha = 0.1

	current_beta = np.zeros(dim + 1)

	for i in range(iteration_cnt):
		gradient = X.dot(current_beta) - y
		gradient = X_trans.dot(gradient)

		gradient_norm = np.linalg.norm(gradient, ord=2)
		gradient_norm_sq = gradient_norm ** 2

		step_size = 1
		# back track line search
		while target_func(X, y, current_beta - step_size * gradient) > \
		      target_func(X, y, current_beta) - bt_alpha * step_size * gradient_norm_sq:
			step_size *= bt_beta
		
		current_beta = current_beta - step_size * gradient

		diff = target_func(X, y, current_beta) - opt_loss
		diff /= n
		print(math.log(diff))

solve_sequential()
