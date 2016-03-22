from os import listdir
from os.path import isfile, join
import os
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

mypath = os.getcwd() + '/pickles2'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


def movingaverage(steps, window_size):
	moving_average = []
	for i, value in enumerate(steps):
		if i < window_size:
			window = steps[:i+1]
		else:
			a = window_size -1
			window = steps[i-a:i+1]
		mean = np.mean(window)
		moving_average.append(mean)
	return moving_average



legends_all = []
data_all = []
data_smoothed_all = []
totals = []
lasts = []


for file in onlyfiles:
	open_list = pickle.load(open('pickles2/' + file, 'rb'))
	parameters = open_list[2]
	# if parameters['policy'] == 'exponential20':
	if True:
		print('here')
		legend = open_list[1]
		legends_all.append(legend)
		data = open_list[3]
		data_all.append(data)
		smoothed_data = movingaverage(data,10)
		data_smoothed_all.append(smoothed_data)
		# Average moves per epsiode
		total = sum(data)
		totals.append(total)
		# Last end of episode
		last = data[-1]
		lasts.append(last)
		



# for data in data_all:
# 	plt.plot(data)

for smoothed_data in data_smoothed_all:
	plt.plot(smoothed_data[0:50])

# for smoothed_data in data_all:
# 	plt.plot(smoothed_data)


plt.legend(legends_all)
plt.show()

# width = 0.35
# ind = [i for i in range(len(totals))]
# plt.bar(ind, totals, width)
# plt.xticks([i + width/2. for i in ind], legends_all,rotation=90)

# plt.show()

for index, legend in enumerate(legends_all):
	print(legend)
	print(totals[index])
	print(lasts[index])

