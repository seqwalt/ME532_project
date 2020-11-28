import numpy as np
import matplotlib.pyplot as plt

A = np.genfromtxt('../data/cardio_disease.csv',delimiter=';',skip_header=1)
print(A.shape)
