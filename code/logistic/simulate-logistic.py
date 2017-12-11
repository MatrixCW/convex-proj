import numpy as np
import os
import matplotlib.pyplot as plt
import mnist_data
import random


def sigmoid(w, xs):
    assert isinstance(w, np.matrix)
    assert isinstance(xs, np.matrix)

    res = 1.0 / (1 + np.exp(- xs * w))
    assert np.all(res != 0.0) and np.all(res != 1.0)
    return res


def objective(w, xs, ys):
    assert isinstance(w, np.matrix)
    assert isinstance(xs, np.matrix)
    assert isinstance(ys, np.matrix)

    return np.mean(
        - np.multiply(ys, np.log(sigmoid(w, xs)))
        - np.multiply((1.0 - ys), np.log(1.0 - sigmoid(w, xs)))
    )


def error_rate(w, xs, ys):
    assert isinstance(w, np.matrix)
    assert isinstance(xs, np.matrix)
    assert isinstance(ys, np.matrix)

    return np.mean(np.abs(ys - sigmoid(w, xs)) > 0.5)


def gradient(w, xs, ys):
    assert isinstance(w, np.matrix)
    assert isinstance(xs, np.matrix)
    assert isinstance(ys, np.matrix)

    return np.mean(np.multiply(sigmoid(w, xs) - ys, xs), axis=0).T


def minibatch(xs, ys, nbatches, ibatch):
    m, _ = xs.shape
    ibeg = m // nbatches * ibatch
    iend = min(m // nbatches * (ibatch + 1), m)
    return xs[ibeg:iend, :], ys[ibeg:iend, :]


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
        assert chosen_worker_event

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
            assert (chosen_worker_event[0].startswith("pull"))
            over_all_seq.append(chosen_worker_event.pop(0))
            push_w = "push_" + str(chosen_worker)
            pull_w = "pull_" + str(chosen_worker)
            worker_event[chosen_worker] = [push_w, pull_w]

    return over_all_seq


path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mnist_data')
xs, ys = mnist_data.read(dataset='training', path=path)
xs = np.matrix(xs).astype('float') / 255.0
ys = np.matrix(ys).T
ys = (ys != 0).astype('float')

idxs0, _ = np.where(ys == 0.0)
idxs1, _ = np.where(ys == 1.0)
np.random.shuffle(idxs1)
idxs1 = idxs1[:len(idxs0)]
xs0 = xs[idxs0, :]
xs1 = xs[idxs1, :]
ys0 = ys[idxs0, :]
ys1 = ys[idxs1, :]

xs = np.append(xs0, xs1, axis=0)
ys = np.append(ys0, ys1, axis=0)

m, d = xs.shape
w_init = np.matrix(np.zeros((d, 1)))

staleness_try = [0, 5, 10, 20, 50]
# staleness_try = [5, 10000]
colors = ['yellow', 'blue', 'black', 'red', 'green']
nworkers = 16
niters = 20
rate = 0.01

batches = [minibatch(xs, ys, nworkers, i) for i in range(nworkers)]

for staleness in staleness_try:
    print('staleness =', staleness)
    event_seq = gen_event_sequence(nworkers, staleness, niters)
    w_workers = [w_init for _ in range(nworkers)]
    w_server = w_init
    objs_server = []

    for event in event_seq:
        event_type, worker_num = event.split("_")
        worker_num = int(worker_num)
        assert (event_type in ["push", "pull"])

        if event_type == "push":
            xs_batch, ys_batch = batches[worker_num]
            w_curr = w_workers[worker_num]
            grad = gradient(w_curr, xs_batch, ys_batch)
            w_server = w_server - rate * grad / nworkers
            objs_server.append(objective(w_server, xs, ys))
        else:
            assert event_type == "pull"
            w_workers[worker_num] = w_server

    plt.plot(range(len(objs_server)), np.log(objs_server), color=colors[staleness_try.index(staleness)], label=('Staleness: ' + str(staleness)))

plt.title('Learning rate: ' + str(rate))
plt.legend()
plt.show()
