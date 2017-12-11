import random

def gen_event_sequence(worker_cnt, staleness, iter_cnt):
	over_all_seq = []
	worker_event = {}
	worker_iter_num = {}
	
	for i in range(worker_cnt):
		push_i = "push_" + str(i)
		pull_i = "pull_" + str(i)
		worker_event[i] = [push_i, pull_i]
		worker_iter_num[i] = 0
	
	random.seed()
	
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
	
