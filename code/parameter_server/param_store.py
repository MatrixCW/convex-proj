from threading import Lock

__all_params = {}
__param_name_to_lock = {}


def add_param(key, vals):
    assert(isinstance(key, str))
    __all_params[key] = vals
    __param_name_to_lock[key] = Lock()


def remove_param(key):
    if key in __all_params:
        lock = __param_name_to_lock[key]
        lock.acquire()
        del __all_params[key]
        del __param_name_to_lock[key]
        lock.release()


def update_param(key, range_start, range_end, new_vals, rate):
    assert(key in __all_params)
    lock = __param_name_to_lock[key]
    lock.acquire()
    current_version = __all_params[key]
    for i in range(range_start, range_end):
        current_version[i] += new_vals[i - range_start] * rate
    lock.release()

def get_param(key, range_start, range_end):
    assert(key in __all_params)
    lock = __param_name_to_lock[key]
    lock.acquire()
    ret = __all_params[key][range_start:range_end]
    lock.release()
    return ret
