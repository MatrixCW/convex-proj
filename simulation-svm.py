import hw2_data
import numpy as np
import random
import matplotlib.pyplot as plt


def loss(w, xs, signs):
    assert isinstance(w, np.matrix)
    assert isinstance(xs, np.matrix)
    assert isinstance(ys, np.matrix)

    return np.maximum(0, 1 - np.multiply(signs, xs * w))


def gradients(ws, xs, signss, signed_xss, nlabels):
    assert isinstance(ws[0], np.matrix)
    assert isinstance(xs, np.matrix)
    assert isinstance(signss[0], np.matrix)

    grads = [np.matrix(np.zeros((d, 1))) for _ in range(nlabels)]  # usage: grads[label]
    for j in range(nlabels):
        w = ws[j]

        ls = loss(w, xs, signss[j])
        grads[j] = w + lambda_ * (-2) * np.mean(np.multiply(ls, signed_xss[j]))
    return grads


def objective(ws, xs, signss, nlabels):
    f = 0.0
    for j in range(nlabels):
        w = ws[j]
        ls = loss(w, xs, signss[j])
        f += 0.5 * float(w.T * w) + lambda_ * np.mean(np.multiply(ls, ls))
    return f


def minibatch(xs, nbatches, ibatch):
    m, _ = xs.shape
    ibeg = m // nbatches * ibatch
    iend = min(m // nbatches * (ibatch + 1), m)
    return xs[ibeg:iend, :]


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


xs, ys = hw2_data.load()
idxs0, _ = np.where(ys == 1)
idxs1, _ = np.where(ys == 2)
xs0 = xs[idxs0, :]
xs1 = xs[idxs1, :]
ys0 = ys[idxs0, :]
ys1 = ys[idxs1, :]
xs = np.append(xs0, xs1, axis=0)
ys = np.append(ys0, ys1, axis=0)
m, d = xs.shape
nlabels = np.max(ys)
idxs = np.array(range(m))
np.random.shuffle(idxs)
xs = xs[idxs, :]
ys = ys[idxs, :]
nworkers = 16

# signss[label] = 1{y_j == j}
signss = [2 * (ys == i + 1) - 1 for i in range(nlabels)]

# signed_xss[label] = 1{y_j == j} x_i
signed_xss = [np.multiply(signss[i], xs) for i in range(nlabels)]

# signss_batches[worker][label]
signss_batches = [[minibatch(signs, nworkers, i) for signs in signss] for i in range(nworkers)]

# signed_xss_batches[label][worker]
signed_xss_batches = [[minibatch(signed_xs, nworkers, i) for signed_xs in signed_xss] for i in range(nworkers)]

# xs_batches[worker]
xs_batches = [minibatch(xs, nworkers, i) for i in range(nworkers)]

# ys_batches[worker]
ys_batches = [minibatch(ys, nworkers, i) for i in range(nworkers)]

# for lambda_ in [0.1, 1.0, 30.0, 50.0]:
# for lambda_ in [0.1]:
#     ws = [np.matrix(np.zeros((d, 1))) for _ in range(nlabels)]  # usage: ws[label]
#     fs = []
#     for iiter in range(50):
#         print('iteration', iiter)
#
#         batch = 0
#         xs_batch = xs_batches[batch]
#         signss_batch = signss_batches[batch]
#         signed_xss_batch = signed_xss_batches[batch]
#
#         grads = gradients(ws, xs_batch, signss_batch, signed_xss_batch, nlabels)
#         for j in range(nlabels):
#             ws[j] = ws[j] - 0.001 * grads[j]
#
#         f = objective(ws, xs, signss, nlabels)
#         print(f)
#         fs.append(f)
#
#     plt.plot(fs, label='Objective Function')
#
# plt.show()

staleness_try = [0, 10000]
colors = ['yellow', 'blue', 'black', 'red', 'green']
niters = 20
rate = 0.1
lambda_ = 0.1

for staleness in staleness_try:
    print('staleness =', staleness)
    event_seq = gen_event_sequence(nworkers, staleness, niters)

    ws_workers = [[np.matrix(np.zeros((d, 1))) for _ in range(nlabels)] for _ in range(nworkers)]
    ws_server = [np.matrix(np.zeros((d, 1))) for _ in range(nlabels)]
    objs_server = []

    for event in event_seq:
        event_type, worker_num = event.split("_")
        worker_num = int(worker_num)
        assert (event_type in ["push", "pull"])

        if event_type == "push":
            ws_curr = ws_workers[worker_num]
            xs_batch = xs_batches[worker_num]
            signss_batch = signss_batches[worker_num]
            signed_xss_batch = signed_xss_batches[worker_num]

            grads = gradients(ws_curr, xs_batch, signss_batch, signed_xss_batch, nlabels)
            ws_server = [ws_curr[j] - rate * grads[j] for j in range(nlabels)]

            objs_server.append(objective(ws_server, xs, signss, nlabels))
        else:
            assert event_type == "pull"
            ws_workers[worker_num] = ws_server

    plt.plot(range(len(objs_server)), np.log(objs_server), color=colors[staleness_try.index(staleness)])

plt.show()
