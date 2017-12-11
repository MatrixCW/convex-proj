import numpy
from threading import Condition
from parameter_server import consistency
import threading


__worker_pipes = {}
__poisson_lam = 2  # expect average communication time 0.02 sec
__send_buffer = None
__sender = None

__push_cv = Condition()
__wait_cv = Condition()


def sender_func(worker_id):

    push_pipe, pull_pipe = __worker_pipes[worker_id]

    while True:
        time_to_sleep = max(numpy.random.poisson(lam=__poisson_lam), 1.0)
        time_to_sleep /= 100.0

        __push_cv.acquire()
        while not __send_buffer:
            __push_cv.wait()
        item_to_send = __send_buffer.pop(0)
        __push_cv.release()
        __push_cv.notify()

        push_pipe.send(item_to_send)

        if len(item_to_send) == 0:
            return


def register_worker(worker_id, push_pipe, pull_pipe):
    __worker_pipes[worker_id] = (push_pipe, pull_pipe)


def push_params(param):
    __push_cv.acquire()
    __send_buffer.append(param)
    __push_cv.release()
    __push_cv.notify()


def pull_params(keys, worker_id):
    pull_pipe = __worker_pipes[worker_id][1]
    pull_pipe.send(keys)
    return pull_pipe.recv()


def sync():
    current_model = consistency.get_consistency_model()
    __push_cv.acquire()
    while len(__send_buffer) > current_model:
        __push_cv.wait()
    __push_cv.release()


def init_sender(worker_id):
    global __send_buffer
    global __sender
    assert(__send_buffer is None)
    __send_buffer = []
    assert(__sender is None)
    __sender = threading.Thread(target=sender_func, args = (worker_id,))
    __sender.start()


def worker_done(worker_id):
    _, pull_pipe = __worker_pipes[worker_id]
    push_params([])
    pull_pipe.send([])
    __sender.join()
