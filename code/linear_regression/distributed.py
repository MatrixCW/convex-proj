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


def optimal_solve(X, y):
	X_trans = X.transpose()
	tmp = X_trans.dot(X)
	tmp_inv = np.linalg.inv(tmp)
	opt = tmp_inv.dot(X_trans)
	opt = opt.dot(y)
	return target_func(X, y, opt)


def solve_distributed():
	X, y, n, dim = data_loader.get_data()
	X_trans = X.transpose()
	iteration_cnt = 200

	opt_loss = optimal_solve(X, y)

	all_staleness = [5, 10, 50, 100]
	colors = ['yellow', 'blue', 'black', 'red']

	worker_cnt = 4

	bt_beta = 0.5
	bt_alpha = 0.1

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

				gradient = local_X.dot(local_beta) - local_y
				gradient = local_X_trans.dot(gradient)

				gradient_norm = np.linalg.norm(gradient, ord=2)
				gradient_norm_sq = gradient_norm ** 2

				step_size = 1
				# back track line search
				while target_func(X, y, current_server_beta - step_size * gradient) > \
				      target_func(X, y, current_server_beta) - bt_alpha * step_size * gradient_norm_sq:
					step_size *= bt_beta
				current_server_beta -= step_size * gradient

				diff = target_func(X, y, current_server_beta) - opt_loss
				diff /= n
				current_loss.append(math.log(diff))

				print(math.log(diff))

			else:
				worker_param[worker_num] = np.copy(current_server_beta)
		plt.plot([i for i in range(len(current_loss))], list(current_loss), color=colors[all_staleness.index(staleness)])
	plt.show()

def solve_distributed_fixed():
	X, y, n, dim = data_loader.get_data()
	X_trans = X.transpose()
	iteration_cnt = 1000

	# step size 0.000001 , staleness = 5 converges, staleness = 100 diverges
	step_size = 0.0000005

	opt_loss = optimal_solve(X, y)

	all_staleness = [5, 10, 20, 100]
	colors = ['yellow', 'blue', 'black', 'red']
	worker_cnt = 4

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

				gradient = local_X.dot(local_beta) - local_y
				gradient = local_X_trans.dot(gradient)

				current_server_beta -= step_size * gradient
				
				diff = target_func(X, y, current_server_beta) - opt_loss
				diff /= n

				current_loss.append(math.log(diff))
				print(math.log(diff))

			else:
				worker_param[worker_num] = np.copy(current_server_beta)

		plt.plot([i for i in range(len(current_loss))], list(current_loss), color=colors[all_staleness.index(staleness)])
	plt.show()


solve_distributed()
