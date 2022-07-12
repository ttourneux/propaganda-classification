import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
import pprint

from tabulate import tabulate 




def plot_accuracy():
    '''
    input: 
    data_name(string): can take values "twitter" or "all_data" or "non_twitter"
    
    return: 
    NA
    
    to be changed:
    the file path on line 27 could be changed to graph different scores
    
    '''


    fp = open('scores/scores.json')
    data = json.load(fp)
    ## not sure if this^ should go in or out of the function
         ## it just seems unlikely we would want to change the score file path each time we run it.
    
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
    
    
#plot_accuracy()
