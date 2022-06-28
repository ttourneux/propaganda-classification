import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
import pprint

from tabulate import tabulate 

fp = open('scores/scores.json')
data = json.load(fp)


def plot_accuracy(data):
    acc_arr = np.array([ [k,float(v)] for k,v in data.items() if 'acc' in k])
    
    acc_arr = pd.DataFrame(acc_arr, columns=["model","acc"]).sort_values(by='acc')
    acc_arr.iloc[:,1] = acc_arr['acc'].astype(float)
    #print(acc_arr)
    
    
    #sns.barplot(list(acc_arr[:,0]),list(acc_arr[:,1].astype("float32")), order = "increasing")
    #plt.bar(list(acc_arr[:,0]),list(acc_arr[:,1].astype("float32")))
    print(acc_arr)
    plt.bar(x = "model" , height = "acc", data =acc_arr )
    plt.xticks(rotation = 90)
    plt.yticks(np.arange(.5,.75,.05))
    plt.ylim(bottom = .5)
    plt.title("accuracy per model-data pair")
    print(tabulate(acc_arr))
    plt.show()

'''
def plot_accuracy(data):
    acc_arr = np.array([ [k,float(v)] for k,v in data.items() if 'acc' in k])
    plt.bar(list(acc_arr[:,0]),list(acc_arr[:,1].astype("float32")))
    plt.xticks(rotation = 90)
    plt.yticks(np.arange(.5,.75,.05))
    plt.ylim(bottom = .5)
    plt.title("accuracy per model-data pair")
    print(tabulate(acc_arr))
    plt.show()
'''    
    
plot_accuracy(data)
