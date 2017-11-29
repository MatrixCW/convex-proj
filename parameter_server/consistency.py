SEQUENTIAL = 0
EVENTUAL = float("inf")

__current_model = SEQUENTIAL

def set_consistency_model(model):
    assert(model >= 0)
    __current_model = model

def get_consistency_model():
    return __current_model