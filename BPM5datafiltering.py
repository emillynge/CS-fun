__author__ = 'emil'
import requests
import numpy as np
from matplotlib import pyplot as plt

redownload_data = False

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


print(raw_data)


