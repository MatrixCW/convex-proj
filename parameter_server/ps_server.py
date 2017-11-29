from parameter_server import param_store
import threading

__custom_funcs = []
__worker_ids = set()

# pipes the server listen to to get
# parameter updates from workers
__worker_push_pipes = []

# pipes the server listen to to push
# parameters to workers as requested
__worker_pull_pipes = []

__learning_rate = 1.0


def set_learning_rate(rate):
    __learning_rate = rate


def add_custom_funcs(func):
    assert(func and callable(func))
    __custom_funcs.append(func)


def add_worker(worker_id, push_pipe, pull_pipe):
    assert(worker_id not in __worker_ids)
    assert(push_pipe)
    assert(pull_pipe)
    __worker_push_pipes.append(push_pipe)
    __worker_pull_pipes.append(pull_pipe)

# server's background thread, it will loop and
# poll each worker_pull_pipe, and take actions
# if there is any data
def server_bg_func():
    active_worker_cnt = len(__worker_ids)
    while True:
        for pipe in __worker_pull_pipes:
            if pipe.closed:
                continue
            elif not pipe.poll():  # pipe is empty, nothing to read
                continue
            else:
                data = pipe.recv()
                if len(data) == 0:
                    pipe.close()
                    active_worker_cnt -= 1
                    if active_worker_cnt == 0:
                        return
                else:
                    ret = {}
                    for tup in data:
                        key, r_start, r_end = tup
                        ret[key] = param_store.get_param(key, r_start, r_end)
                    pipe.send(ret)


def server_loop():
    worker_cnt = len(__worker_ids)
    t = threading.Thread(target=server_bg_func, args = ())
    t.start()
    while True:
        for pipe in __worker_push_pipes:
            if pipe.closed:
                continue
            elif not pipe.poll():  # pipe is empty, nothing to read
                continue
            else:
                data = pipe.recv()
                if len(data) == 0:
                    pipe.close()
                    worker_cnt -= 1
                    if worker_cnt == 0:
                        t.join()
                        return
                else:
                    for tup in data:
                        key, r_start, r_end, new_vals = tup
                        param_store.update_param(key, r_start, r_end, new_vals, __learning_rate)
        if __custom_funcs:
            for func in __custom_funcs:
                func()
    t.join()
