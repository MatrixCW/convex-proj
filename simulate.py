import numpy as np
import math
import random
import matplotlib.pyplot as plt

N = 10000
DIM = 5
NOISE_SCALE = 5
LEARN_RATE = 0.001

WORKER_CNT = 4
ITER_CNT = 200

A = np.random.rand(DIM + 1) * 10

data_X = []
data_y = []

for i in range(N):
	tmp_x = np.random.rand(DIM + 1) * 10
	tmp_x[-1] = 1.0
	tmp_y = A.dot(tmp_x)
	noise = np.random.normal(scale=NOISE_SCALE)
	tmp_y += noise
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

def loss_func(my_A):
	global X
	global y
	global N
	loss = 0
	for i in range(N):
		loss += (y[i] - X[i].dot(my_A)) ** 2
	loss /= N
	return loss

# staleness_try = [5, 10, 15, 20, 25]
staleness_try = [5, 10, 20, 50, 100]
colors = ['yellow', 'blue', 'black', 'red', 'blue']

worker_data_X = []
worker_data_Y = []
each_worker_data_cnt = N // WORKER_CNT
for i in range(WORKER_CNT):
	worker_data_X.append(X[i * each_worker_data_cnt : (i + 1) * each_worker_data_cnt])
	worker_data_Y.append(y[i * each_worker_data_cnt : (i + 1) * each_worker_data_cnt])

for staleness in staleness_try:
	print(staleness)
	event_seq = gen_event_sequence(WORKER_CNT, staleness, ITER_CNT)
	worker_param = {i : np.zeros(DIM + 1) for i in range(WORKER_CNT)}
	current_server_param = np.zeros(DIM + 1)

	current_loss = []

	for event in event_seq:
		event_type, worker_num = event.split("_")
		worker_num = int(worker_num)
		assert(event_type in ["push", "pull"])

		if event_type == "push":
			X_data_of_worker = worker_data_X[worker_num]
			y_data_of_worker = worker_data_Y[worker_num]
			gradient = np.zeros(DIM + 1)
			param_to_use = worker_param[worker_num]
			for i in range(each_worker_data_cnt):
				gradient = gradient + (-y_data_of_worker[i] + X_data_of_worker[i].dot(param_to_use)) * X_data_of_worker[i]
			gradient = (2.0 / each_worker_data_cnt) * gradient
			current_server_param = current_server_param - LEARN_RATE * gradient
			current_loss.append(math.log(loss_func(current_server_param)))
		else:
			worker_param[worker_num] = np.copy(current_server_param)
	
	plt.plot([i for i in range(len(current_loss))], list(current_loss), color=colors[staleness_try.index(staleness)])

plt.show()

