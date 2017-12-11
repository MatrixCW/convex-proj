import numpy as np
import math
import random
import matplotlib.pyplot as plt

N = 10000
DIM = 5
NOISE_SCALE = 1
LASSO_LAM = 1

WORKER_CNT = 4
ITER_CNT = 200

A = np.random.rand(DIM + 1) * 10

data_X = []
data_y = []

for i in range(N):
	tmp_x = np.random.rand(DIM + 1) * 10
	tmp_x[-1] = 1.0
	tmp_y = A.dot(tmp_x)
	# noise = np.random.normal(scale=NOISE_SCALE)
	# tmp_y += noise
	data_X.append(tmp_x)
	data_y.append(tmp_y)

X = np.zeros((N, DIM + 1))

for i in range(N):
	for j in range(DIM + 1):
		X[i][j] = data_X[i][j]

y = np.zeros(N)
for i in range(N):
	y[i] = data_y[i]


def gen_event_sequence(worker_cnt, staleness, iter_cnt):
	over_all_seq = []
	worker_event = {}
	worker_iter_num = {}
	
	for i in range(worker_cnt):
		push_i = "push_" + str(i)
		pull_i = "pull_" + str(i)
		worker_event[i] = [push_i, pull_i]
		worker_iter_num[i] = 0
	
	while len(over_all_seq) < 2 * iter_cnt * worker_cnt:
		chosen_worker = random.randint(0, worker_cnt - 1)
		chosen_worker_event = worker_event[chosen_worker]
		assert(chosen_worker_event)

		if chosen_worker_event[0].startswith("push"):

			would_be_iter_num = worker_iter_num[chosen_worker] + 1
			all_iter_nums = worker_iter_num.values()
			stalest_iter_num = min(all_iter_nums)

			if worker_iter_num[chosen_worker] == stalest_iter_num:
				over_all_seq.append(chosen_worker_event.pop(0))
				worker_iter_num[chosen_worker] += 1

			elif would_be_iter_num - stalest_iter_num > staleness:
				continue

			else:
				over_all_seq.append(chosen_worker_event.pop(0))
				worker_iter_num[chosen_worker] += 1
		else:
			assert(chosen_worker_event[0].startswith("pull"))
			over_all_seq.append(chosen_worker_event.pop(0))
			push_w = "push_" + str(chosen_worker)
			pull_w = "pull_" + str(chosen_worker)
			worker_event[chosen_worker] = [push_w, pull_w]
	
	return over_all_seq

def loss_func(my_beta):
	global X
	global y
	global N
	global LASSO_LAM
	loss = 0
	for i in range(N):
		diff = (y[i] - X[i].dot(my_beta)) ** 2
		diff = 0.5 * diff
		loss += diff
	return loss + LASSO_LAM * np.linalg.norm(my_beta, ord=1)

def soft_thres_op(lam, beta):
	global DIM
	tmp = np.zeros(DIM + 1)
	for i in range(DIM + 1):
		if beta[i] > lam:
			tmp[i] = beta[i] - lam
		elif beta[i] < - lam:
			tmp[i] = beta[i] + lam
		else:
			tmp[i] = 0
	return tmp


def g_x(my_beta):
	global X
	global y
	global N
	global LASSO_LAM
	val = 0
	for i in range(N):
		val += 0.5 * ((y[i] - X[i].dot(my_beta)) ** 2)
	return val

def solve_sequential():
	global DIM
	global X
	global y
	global LASSO_LAM
	current_beta= np.zeros(DIM + 1)


	X_trans = X.transpose()

	for i in range(1000):

		rate = 1
		shrink_beta = 0.5

		while True:

			op = X_trans.dot(y - X.dot(current_beta))
			gradient = -op
			op = rate * op
			op = current_beta + op
			try_beta = soft_thres_op(LASSO_LAM * rate, op)

			gtbeta = (current_beta - try_beta) / rate

			if g_x(try_beta) > g_x(current_beta) - rate * gtbeta.dot(gradient) + ((rate / 2) * (np.linalg.norm(gtbeta, ord=2)) ** 2):
				rate *= shrink_beta
			else:
				current_beta = try_beta
				break
		print(loss_func(current_beta))
		print(current_beta)
	
# solve_sequential()

staleness_try = [5, 10, 50, 100]
colors = ['yellow', 'blue', 'black', 'red']

worker_data_X = []
worker_data_Y = []
each_worker_data_cnt = N // WORKER_CNT
for i in range(WORKER_CNT):
	worker_data_X.append(X[i * each_worker_data_cnt : (i + 1) * each_worker_data_cnt])
	worker_data_Y.append(y[i * each_worker_data_cnt : (i + 1) * each_worker_data_cnt])

shrink_beta = 0.5

for staleness in staleness_try:
	print(staleness)
	event_seq = gen_event_sequence(WORKER_CNT, staleness, ITER_CNT)
	worker_param = {i : np.zeros(DIM + 1) for i in range(WORKER_CNT)}
	current_server_param = np.zeros(DIM + 1)

	current_loss = []

	for event in event_seq:
		# print(event)
		event_type, worker_num = event.split("_")
		worker_num = int(worker_num)
		assert(event_type in ["push", "pull"])

		if event_type == "push":
			X_data_of_worker = worker_data_X[worker_num]
			y_data_of_worker = worker_data_Y[worker_num]
			param_to_use = worker_param[worker_num]

			rate = 1
			X_data_trans = X_data_of_worker.transpose()

			while True:
				op = X_data_trans.dot(y_data_of_worker - X_data_of_worker.dot(param_to_use))
				gradient = -op  # worker compute this value and push to server
				op = rate * op
				op = current_server_param + op  # when server do this computation, it uses its version of param
				try_param = soft_thres_op(LASSO_LAM * rate, op)

				gtbeta = (current_server_param - try_param) / rate

				if g_x(try_param) > g_x(current_server_param) - rate * gtbeta.dot(gradient) + ((rate / 2) * (np.linalg.norm(gtbeta, ord=2)) ** 2):
					rate *= shrink_beta
				else:
					current_server_param = try_param

					local_loss = loss_func(current_server_param)
					print(local_loss)
					# current_loss.append(math.log(loss_func(current_server_param)))
					break
		else:
			worker_param[worker_num] = np.copy(current_server_param)
	
	plt.plot([i for i in range(len(current_loss))], list(current_loss), color=colors[staleness_try.index(staleness)])

plt.show()
