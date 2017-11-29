# class Work(type):
# 	# we use __init__ rather than __new__ here because we want
# 	# to modify attributes of the class *after* they have been
# 	# created
# 	def __new__(meta, name, bases, dict):
# 		return super().__new__(meta, name, bases, dct)

# 	def __init__(cls, name, bases, dct):
# 		super().__init__(name, bases, dct)

import types


def create_customer_worker(**funcs):
	attributes = dict(**funcs)
	for key, value in attributes.items():
		if isinstance(value, types.FunctionType):
			attributes[key] = staticmethod(value)
	return type('WorkerClass', (object,), attributes)


def test_func():
	for f in get_files():
		print(f)
	return 1


def get_files():
	return [1, 2, 3]


customer_worker = create_customer_worker(get_files=get_files, test=test_func)
a = customer_worker()
print(dir(a))
print(a.test())
