from multiprocessing import Process, Pipe
from parameter_server import ps_server
from parameter_server import postoffice
from parameter_server import consistency

def run_ps(worker_main, consistency_model, worker_cnt, rate):

    consistency.set_consistency_model(consistency_model)
    ps_server.set_learning_rate(rate)
    workers = []

    next_worker_id = 1
    for i in range(worker_cnt):
        s_push_pipe, w_push_pipe = Pipe()
        s_pull_pipe, w_pull_pipe = Pipe()
        ps_server.add_worker(next_worker_id, s_push_pipe, s_pull_pipe)
        postoffice.register_worker(next_worker_id, w_push_pipe, w_pull_pipe)
        workers.append(Process(target=worker_main, args=(next_worker_id,)))
        next_worker_id += 1
    
    server = Process(target=ps_server.server_loop, args=())

    server.start()
    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()
    
    server.join()