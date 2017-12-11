import sys
sys.path.insert(0,'..')

import csv
import math
import random
import seq_gen
import data_loader
import numpy as np
import matplotlib.pyplot as plt


def func_g(X, y, my_beta):
	diff_vec = X.dot(my_beta) - y
	tmp_norm = np.linalg.norm(diff_vec, ord=2)
	return tmp_norm * tmp_norm / 2


def target_func(X, y, lasso_lam, my_beta):
	assert(lasso_lam > 0)
	return func_g(X, y, my_beta) + lasso_lam * np.linalg.norm(my_beta, ord=1)


def soft_thres_op(lam, beta):
	dim = len(beta)
	tmp = np.zeros(dim)
	for i in range(dim):
		if beta[i] > lam:
			tmp[i] = beta[i] - lam
		elif beta[i] < - lam:
			tmp[i] = beta[i] + lam
		else:
			tmp[i] = 0
	return tmp


def solve_sequential():
	X, y, n, dim = data_loader.get_data()
	lasso_lam = 1

	opt_val = float('inf')

	X_trans = X.transpose()
	iteration_cnt = 10000

	current_beta = np.zeros(dim + 1)

	for i in range(iteration_cnt):

		rate = 1
		shrink_beta = 0.5

		while True:
			op = X_trans.dot(y - X.dot(current_beta))
			gradient = -op
			op = rate * op
			op = current_beta + op
			try_beta = soft_thres_op(lasso_lam * rate, op)

			gtbeta = (current_beta - try_beta) / rate

			if func_g(X, y, try_beta) > func_g(X, y, current_beta) - rate * gtbeta.dot(gradient) + ((rate / 2) * (np.linalg.norm(gtbeta, ord=2)) ** 2):
				rate *= shrink_beta
			else:
				current_beta = try_beta
				break
		
		opt_val = min(opt_val, target_func(X, y, lasso_lam, current_beta))
		print(opt_val)

solve_sequential()
