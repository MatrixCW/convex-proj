import math
import random
import csv
import numpy as np
import matplotlib.pyplot as plt

def get_data():
	X = np.loadtxt("../data/X.csv", delimiter=',')
	y = np.loadtxt("../data/y.csv", delimiter=',')
	n, dim = X.shape
	return (X, y, n, dim - 1)
