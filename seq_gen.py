import numpy as np
import math
import random
import matplotlib.pyplot as plt

WORK_CNT = 3
STALENESS = 2
ITER_CNT = 10

over_all_seq = []

worker_event = {}

worker_version = {}

global_version = 0

for i in range(WORK_CNT):
	push_i = "push_" + str(i)
	pull_i = "pull_" + str(i)
	worker_event[i] = [push_i, pull_i]
	worker_version[i] = 0

while len(over_all_seq) < ITER_CNT * WORK_CNT:
	# print("len", len(over_all_seq))
	# print(worker_event)
	# print("global", global_version)
	# print(worker_version)
	chosen_worker = random.randint(0, WORK_CNT - 1)
	chosen_worker_event = worker_event[chosen_worker]
	assert(chosen_worker_event)

	if chosen_worker_event[0].startswith("push"):

		would_be_version = global_version + 1
		current_versions = worker_version.values()
		oldest_version = min(current_versions)

		if worker_version[chosen_worker] == oldest_version:
			over_all_seq.append(chosen_worker_event.pop(0))
			global_version += 1

		elif would_be_version - oldest_version > STALENESS * WORK_CNT:
			continue

		else:
			over_all_seq.append(chosen_worker_event.pop(0))
			global_version += 1
	else:
		assert(chosen_worker_event[0].startswith("pull"))
		over_all_seq.append(chosen_worker_event.pop(0))
		worker_version[chosen_worker] = global_version
		push_w = "push_" + str(chosen_worker)
		pull_w = "pull_" + str(chosen_worker)
		worker_event[chosen_worker] = [push_w, pull_w]

for ev in over_all_seq:
	print(ev)





