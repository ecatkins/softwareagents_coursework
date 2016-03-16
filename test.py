from os import listdir
from os.path import isfile, join
import os
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

mypath = os.getcwd() + '/pickles'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

def movingaverage(steps, window_size):
	moving_average = []
	for i, value in enumerate(steps):
		if i < window_size:
			window = steps[:i+1]
		else:
			window = steps[i-4:i+1]
		mean = np.mean(window)
		moving_average.append(mean)
	return moving_average



for file in onlyfiles:
	open_list = pickle.load(open('pickles/' + file, 'rb'))
	data = open_list[3]
	moving_average = movingaverage(data,5)
	print(moving_average)




	break