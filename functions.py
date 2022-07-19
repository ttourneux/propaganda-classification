#!/usr/bin/env python
# coding: utf-8

# In[1]:
import datasets

import math
import numpy as np
import scipy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
#import sklearn as skl
import io
import pickle
import json
import seaborn as sns

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
from sklearn.dummy import DummyRegressor

from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

from sklearn. preprocessing import scale

from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline

from tabulate import tabulate

# In[2]:

def get_data(data_name):

    '''
    input: 
    data_name(string): can take values "twitter" or "all_data" or "non_twitter" representing which data should be retrieved

    return: 
    data(pd.DataFrame()): this is a dataframe of the specified data 
    
    to be changed: NA 
    
    '''

    DS = datasets.Datasets()

    if data_name == "twitter": 
        
        tweets = DS.tweets_df
        retweets = DS.retweets_df
        twitter = tweets.append(retweets)## all tweets and retweets
        data = twitter 
        
    elif data_name =="non_twitter":
        data = DS.non_twitter_df
        #data = non_twitter
        
    elif data_name == 'all_data':
        
        data = DS.all_data_df
        
    else:
        print("something went wrong")
        
        
    return data




def balance_data(data): 
    '''
    input: 
    data(pd.DataFrame()): data that has different number of rows corresponding to Trump than Hillary

    return: 
    data(pd.DataFrame()): data that has the same number of rows for Trump and Hillary
    
    to be changed: NA 
    
    '''
    dfH = data.loc[data['BCandidate']==0,:]
    print("HC length: ",len(dfH))

    dfT =data.loc[data['BCandidate']==1,:]
    print("DT length: ",len(dfT))
    df_new = data
    A = len(dfT)
    B = len(dfH)
    
    if B>A: ## more hillary data
        ##print('look at function "balance_data"')
        missing_rows = B-A
        row_fraction = missing_rows/A
        print('missing_rows:', missing_rows, 'row_fraction:', row_fraction) 
        for i in range(0,math.floor(row_fraction),1):
            print('data repeat',i)
            df_new = df_new.append(dfT)
        leftover = row_fraction - math.floor(row_fraction)
        df_new = df_new.append(dfT[:math.floor(len(dfT)*leftover)])
        
        
    else:## more trump data
        missing_rows = A-B
        row_fraction = missing_rows/B
        print('missing_rows:', missing_rows, 'row_fraction:', row_fraction) 
        for i in range(0,math.floor(row_fraction),1):
            print('data repeat',i)
            df_new = df_new.append(dfH)
		
        leftover = row_fraction - math.floor(row_fraction)
        df_new = df_new.append(dfH[:math.floor(len(dfH)*leftover)])

    dfT =df_new.loc[df_new['BCandidate']==1,:]## need to recalculate how many trump entries there are
    print("DT length ratio: ",len(dfT)/len(df_new))
    
    data = df_new
    
    return data


# In[3]:

def choose_model(name,granularity = 1):
    '''
    input: 
    name(string): the name of the model that should be returned
    granularity(positive int): a parameter that allows for smaller gridsearches as it is increased

    return: 
    model(model class): model containing inital parameters
    param_grid(dictionary): a dictionary containing the hyperparameters and the values that should be tried
    
    to be changed: NA 
    
    '''
    
    if name =="support vector machine" or name == 'SVM':
        model = SVC(gamma='auto') 
        param_grid = { 
            name + '__'+'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
            name + '__'+'C' : [1/100,1/10,1,10,100,100]}# list(range(1,30, granularity)) 
        
    elif name == 'extra trees':
        model = ExtraTreesClassifier()   
        param_grid = {
            name + '__' + 'max_leaf_nodes' : [10,100,1000,10000],
        
            #name + '__'+'n_estimators' : list(range(1,100,2*granularity)),
            name + '__'+'max_depth': list(range(2,100,2*granularity))
            
            }
        
    elif name == 'random forest':
        model = RandomForestClassifier()
        param_grid={ 
            name + '__' + 'max_leaf_nodes' : [10,100,1000,10000],
        
            #name + '__'+'n_estimators' : list(range(1,100,2*granularity)),
            name + '__'+'max_depth': list(range(2,100,2*granularity))
            
            #name + '__'+'random_state': list(range(10,100,2*granularity))
            }
        ## here max depth and max leaf nodes should also be searched 
    elif name == 'logistic regression':
        model = SGDClassifier(early_stopping = True, max_itter = 1e6)
        
        param_grid = {
            name + '__' +'loss' : ['log_loss', 'hinge'] ,## logistic regression, svm with sgd
            name + '__' +'alpha' : [1/100,1/10,1,10,100,100],
            name + '__' +'learning_rate' : ['optimal','adaptive']
            
            
        }
        ## SGD is used instead of LogisticRegression because I think that is the how we want to optimize
        '''model = LogisticRegression(class_weight = 'balanced',
            solver = 'saga',
            penalty = 'l2')## this solver supports all the different types of 
        param_grid={ 
            #name + '__'+'penalty' : ['l2'],#'l1', 'none' 
             
             ## always use balanced when it is a parameter. 
            name + '__'+'max_itters' : [1e-4,1e-2,1e-6],
 
            
            name + '__'+'C' : [1/100,1/10,1,10,100,100] for x in range(0,30,granularity)]#list(range(1,30, granularity))}## inverse of regularization strength
            #log1.100001 , log(1.2000001)... log(3.10000000....1)  
            #try 1/100 to 1000 by multiples of 10.( orders of magnitude. 
            }'''
        
    elif name == 'neural net' or name == 'NN':
        model = MLPClassifier(max_iter=1e6, learning_rate = 'adaptive')
        param_grid={ 
           ### early stopping parameter so we can use that. 
        
            name + '__'+'solver' : [ 'adam'],## one solver is fine 'lbfgs', 'sgd',
            name + '__'+'alpha' : [1/100,1/10,1,10,100,100],## regularization
            name + '__'+'activation': ['tanh', 'relu']
            #name + '__'+'learning_rate': ['adaptive']# no need for constant.
            } 
        
    else: 
        print("wrong name input")
        print("change granularity?")
        
        
    filename = "untitled_model"
    pickle.dump(model, open("trained_models/"+filename, 'wb'))    
    
    return model, param_grid



def model_training (name, model, X_train,Y_train,param_grid):

    param_grid["sfs__n_features_to_select"] = list(range(1,len(X_train.columns)-1)) 

    sfs = SequentialFeatureSelector(model)
    
    make_pipeline(sfs,model)
    pipe = Pipeline([('sfs',sfs),(name,model)])


    CVM = GridSearchCV(
        pipe, param_grid = param_grid,n_jobs = -1, verbose = True, scoring = 'accuracy')## note, refit = True by default and thus will "return" the best model ie our predict and such works as if we were using a model with the best hyperparameters.

    CVM.fit(X_train,Y_train)
    return CVM


# In[4]:


def graph_best_with_n_features(name,fitted_model, data_name):
    df_cv_scores=pd.DataFrame(fitted_model.cv_results_).sort_values(by='rank_test_score')
    scores = df_cv_scores.loc[:,["mean_test_score","param_sequentialfeatureselector__n_features_to_select"]]
    n_feat_list = scores.loc[:,'param_sequentialfeatureselector__n_features_to_select'].unique()

    feat_n = {}

    for i in n_feat_list:
        #print(i)
        Maximum = scores.loc[scores.loc[:,"param_sequentialfeatureselector__n_features_to_select"]==i,:].max()[0]
        feat_n[i] = [Maximum]
    #feat_n   
    df = pd.DataFrame(feat_n)
    df = df.transpose()
    #print(df)
    #print(list(df.columns))
    df.plot(kind = 'bar')
    plt.xticks(rotation = 0)
    plt.yticks(np.arange(.5,.75,.05))
    plt.ylim(bottom = .5)
    plt.title(data_name+'_'+ name +"accuracy per model-data pair")
    plt.savefig("figures/"+data_name+'_'+ name + '_n_feat.png')
    plt.show()


def save_model(model,name, data_name):
    filename = name+'_GS_'+data_name
    pickle.dump(model, open("trained_models/"+filename, 'wb'))
    return filename


# In[5]:


def scorer(filename, X_test, Y_test ): 
    loaded_model = pickle.load(open("trained_models/"+filename, 'rb'))
    print(filename+"Results on test data")
    print("roc_auc:",roc_auc_score(Y_test,loaded_model.predict(X_test)))
    print("accurarcy:",accuracy_score(Y_test,loaded_model.predict(X_test)))
    print("note 1 is Donald Trump")
    cm = confusion_matrix(Y_test,loaded_model.predict(X_test))
    ConfusionMatrixDisplay(cm).plot()
    
    json_ptr = open("scores/scores.json", 'r')
    scores = json.load(json_ptr)
    scores["roc_auc_"+filename] = roc_auc_score(Y_test,loaded_model.predict(X_test))
    scores["accurarcy_"+filename] = accuracy_score(Y_test,loaded_model.predict(X_test))
    
    json_ptr = open("scores/scores.json", 'w')
    json.dump(scores, json_ptr)## confirm that this is overwriting the old scores(I think it does).
    ## consider pretty printing the json file 
    json_ptr.close()
    return loaded_model


# In[7]:


def graph_MDI_importance(X_train,Y_train,loaded_model,name, data_name): 
    
    if name == "extra trees":
        MOD = ExtraTreesClassifier(n_estimators =loaded_model.best_params_['n_estimators'],random_state = loaded_model.best_params_['random_state'] )
    elif name == "random forest":
        MOD = RandomForestClassifier(n_estimators =loaded_model.best_params_['n_estimators'],
                                                  random_state = loaded_model.best_params_['random_state'],
                                                  max_depth = loaded_model.best_params_['max_depth'])
    else:
        print("there is a problem with name param")
        return
        
    
    MOD.fit(X_train,Y_train)
    ## is there a better way to do this than a bigggg if else   
        
    importances = MOD.feature_importances_
    std = np.std([MOD.feature_importances_ for tree in MOD.estimators_],
                 axis=0)
    indices = np.argsort(importances)
    feature_list = [X_train.columns[indices[f]] for f in range(X_train.shape[1])]  #names of features. ## may have to be X)train
    ff = np.array(feature_list)
                           
    '''
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f) name: %s" % (f + 1, indices[f], importances[indices[f]], ff[indices[f]]))
    '''

    plt.figure()
    plt.rcParams['figure.figsize'] = [16, 6]
    plt.title("Feature importances using Mean Decrease in Impurity")## MDI
    plt.bar(range(X_train.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), ff[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    
    pic_name = name+'_MDI_importance_'+data_name+".png"
    plt.savefig(pic_name, format="png")
    
    plt.show()

    


# In[8]:


def show_permutation_feature_importance(loaded_model,X_train, Y_train, name, data_name):
    if not (name == 'random forest') and not (name == 'extra trees'): 
        print(' importance not available or name param is wrong')
        print('name:', name)
        return
    
    if name == "extra trees":
        MOD = ExtraTreesClassifier(n_estimators =loaded_model.best_params_['n_estimators'],
                                   random_state = loaded_model.best_params_['random_state'] )
    elif name == "random forest":
        MOD = RandomForestClassifier(n_estimators =loaded_model.best_params_['n_estimators'],
                                                  random_state = loaded_model.best_params_['random_state'],
                                                  max_depth = loaded_model.best_params_['max_depth'])
    else:
        print("there is a problem with name param")
        return
        
    
    MOD.fit(X_train,Y_train)
    scoring = ['roc_auc', 'accuracy']#
    r_multi = permutation_importance(
        MOD, X_train, Y_train, n_repeats=30, random_state=0, scoring=scoring)## n_repeats should be bigger

    features= list(X_train.columns)
    roc_auc_importance = []
    roc_auc_std = []
    acc_importance =[]
    acc_std=[]
    for metric in r_multi:
        #print(f"{metric}")
        r = r_multi[metric]
        for i in range(len(features)):
            #if r.importances_mean[i] - 2 * r.importances_std[i] > 0: ## do we want to know this?
            #print(i,f"    {list(X_train.columns)[i]:<8}"
            #      f"{r.importances_mean[i]:.3f}"
            #      f" +/- {r.importances_std[i]:.3f}")
            if metric == 'roc_auc':
                roc_auc_importance.append(r.importances_mean[i])
                roc_auc_std.append(r.importances_std[i])
            if metric == 'accuracy': 
                acc_importance.append(r.importances_mean[i])
                acc_std.append(r.importances_std[i])

    d = {'features': features, 'acc_importance': acc_importance, 'roc_auc_importance': roc_auc_importance}
    acc_importance_df = pd.DataFrame(d)


    plt.figure(figsize=(15, 4))# this adjusts how far away the plots are from eachother

    ax = plt.subplot(111)

    acc_importance_df = acc_importance_df.sort_values(by = 'acc_importance')
    acc_importance_df.plot(x = 'features', y ='acc_importance', kind = 'bar',ax = ax)
    plt.xticks(rotation = 'vertical')
    ax.set_title("Feature importance of"+name+" based on accuracy")
    ax.set_ylabel("permutation importance")


    pic_name = name+'_permutation_importance_'+data_name+".png"
    plt.savefig(pic_name, format="png")
    plt.show()

# In[ ]:
def barchart(data, data_name): 
    names= []
    trump = []
    hillary = []
    for i in data.columns[4:]:
        
        #print(i)
        names.append(i)
        
        dfH = data.loc[data['BCandidate']==0,:]
        n_rows = sum(dfH[i]!=0)
        hillary.append(n_rows)
        #print(n_rows)
        #hillary = [x/len(dfH) for x in hillary]
        #hillary = np.array(hillary)/len(dfH)
    
    
        dfT =data.loc[data['BCandidate']==1,:]
        n_rows = sum(dfT[i]!=0)
        trump.append(n_rows)
        #print(n_rows)
    names.pop(-2)
    trump.pop(-2)
    hillary.pop(-2)
    trump = [x/len(dfT) for x in trump]
    hillary = [x/len(dfH) for x in hillary]
        #trump = np.array(trump)/len(dfT)
    
    
    df = pd.DataFrame({"trump": trump, "hillary":hillary}, index = names)
    df.plot.bar(rot=90, title = "fraction of media where each tactic was used in "+data_name)
    print("data from " +data_name)
    
    return names, trump, hillary 


## 
def boxplotting(data, data_name):
    data = data.replace(0,np.nan)
    df1 =pd.melt(data, id_vars = ["BCandidate", "Content_Category"],value_vars =data.columns[4:],  var_name = "propaganda techinique",value_name = "times used")
    df1['BCandidate'] = df1["BCandidate"].replace(np.nan, 0)
    print(sum(df1["BCandidate"]==0))
    ##print(sum(df1[:,3]))
    sns.boxplot ( data = df1, x ="propaganda techinique", y = "times used", hue = "BCandidate", showmeans = True, meanline = True, showfliers = False)
    plt.xticks(rotation = 90)
    
    plt.title("frequency with which each propaganda technique was used in " +data_name)## among documents where that prop tactic was used.    
    plt.show()

def plot_accuracy(data):
    acc_arr = np.array([ [k,v] for k,v in data.items() if 'acc' in k])
    plt.bar(list(acc_arr[:,0]),list(acc_arr[:,1]),bottom = 0)
    plt.xticks(rotation = 90)
    plt.title("accuracy per model-data pair")
    print(tabulate(acc_arr))
    plt.show()





