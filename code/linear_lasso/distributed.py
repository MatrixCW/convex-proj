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


def solve_distributed():
	X, y, n, dim = data_loader.get_data()
	X_trans = X.transpose()
	iteration_cnt = 75
	lasso_lam = 1
	opt_val = float('inf')
	shrink_beta = 0.5

	all_staleness = [0, 5, 10, 20, 50]
	colors = ['yellow', 'blue', 'black', 'red', 'green']

	worker_cnt = 4

	# init each worker's data count
	worker_data_X = []
	worker_data_X_trans = []
	worker_data_Y = []
	each_worker_data_cnt = n // worker_cnt
	for i in range(worker_cnt):
		worker_data_X.append(X[i * each_worker_data_cnt : (i + 1) * each_worker_data_cnt])
		worker_data_Y.append(y[i * each_worker_data_cnt : (i + 1) * each_worker_data_cnt])
		worker_data_X_trans.append(worker_data_X[i].transpose())

	for staleness in all_staleness:
		event_seq = seq_gen.gen_event_sequence(worker_cnt, staleness, iteration_cnt)
		worker_param = {i : np.zeros(dim + 1) for i in range(worker_cnt)}
		current_server_beta = np.zeros(dim + 1)

		current_loss = []

		for event in event_seq:
			event_type, worker_num = event.split("_")
			worker_num = int(worker_num)
			assert(event_type in ["push", "pull"])

			if event_type == "push":
				local_X = worker_data_X[worker_num]
				local_X_trans = worker_data_X_trans[worker_num]
				local_y = worker_data_Y[worker_num]
				local_beta = worker_param[worker_num]

				neg_grad = local_X_trans.dot(local_y - local_X.dot(local_beta))
				gradient = -neg_grad  # worker compute this value and push to server

				learn_rate = 1

				while True:
					op = learn_rate * neg_grad
					op = current_server_beta + op  # when server do this computation, it uses its version of param
					try_param = soft_thres_op(lasso_lam * learn_rate, op)

					gtbeta = (current_server_beta - try_param) / learn_rate

					if func_g(X, y, try_param) > func_g(X, y, current_server_beta) - learn_rate * gtbeta.dot(gradient) + ((learn_rate / 2) * (np.linalg.norm(gtbeta, ord=2)) ** 2):
						learn_rate *= shrink_beta
					else:
						current_server_beta = try_param

						diff = target_func(X, y, lasso_lam, current_server_beta) - 4948.57512794
						diff /= n
						current_loss.append(math.log(diff))
						print(math.log(diff))
						break
			else:
				worker_param[worker_num] = np.copy(current_server_beta)
		loss_to_plot = [current_loss[4 * i] for i in range(len(current_loss) // 4)]
		plt.plot([i for i in range(len(current_loss))], list(current_loss), color=colors[all_staleness.index(staleness)], label=('Staleness: ' + str(staleness)))
	plt.title('Linear regression with L1 Regularization')
	plt.legend()
	plt.show()

solve_distributed()
