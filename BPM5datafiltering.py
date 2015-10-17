__author__ = 'emil'
import requests
import numpy as np
from matplotlib import pyplot as plt
from collections import  namedtuple

redownload_data = False
t_range = (0, 20)
data_url = "https://dl.dropboxusercontent.com/u/2640195/BPM5_data.txt"
def url2mat(url):
    r = requests.get(url)
    mat = np.fromiter((float(element) for line in r.text.split('\n') if '\t' in line for element in line.split('\t')),
                      float)
    return np.matrix(mat.reshape((len(mat)/3, 3)))
if redownload_data:
    raw_data = url2mat(data_url)
    np.save('rawdat.npy', raw_data)
else:
    raw_data = np.load('rawdat.npy')

cropped_data = raw_data[t_range[0] <= raw_data[:, 0], :]
cropped_data = cropped_data[t_range[1] >= cropped_data[:, 0], :]

def show_plots(data):
    plt.plot(data[:, 0], data[:, 1], 'r')
    plt.plot(data[:, 0], data[:, 2], 'g')
    plt.show()

show_plots(cropped_data)

