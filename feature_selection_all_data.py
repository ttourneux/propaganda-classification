from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SequentialFeatureSelector as SFS
import numpy as np

import functions
import datasets
from sklearn.model_selection import train_test_split


# In[3]:


##def feature_selection_loop():
DS = datasets.Datasets()
all_data = DS.all_data_df
data_name = "all_data_FS"## to be changed

data = all_data
data = functions.balance_data(data)

X = data.loc[:,"Num Prop":"Bandwagon,Reductio_ad_hitlerum",]
Y = data.loc[:,"BCandidate"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = .2,shuffle=True)


features_to_select = []

models = ['extra trees', 'random forest',
              'logistic regression',"support vector machine",
              'neural net','logistic regression']
for name in models:
    model, param_grid = functions.choose_model(name)
    #print('choose model')
    sfs = SFS(estimator = model, scoring = 'accuracy', n_features_to_select= "auto", tol = .010,
              direction = 'forward',n_jobs = -1)
    trained_model = sfs.fit(X_train,Y_train)## the warnings should not affect the code
    ## changing X_train to X_train.values makes a difference...
    ## doccumentation says to use np.arrays <=> use .values
    print('{} optimal number of features: {}'.format(model,sfs.n_features_to_select_))
    print('features are:',X_train.columns[sfs.support_] )
    features_to_select.append((X_train.columns[sfs.support_]))
    
    
    
for i in range(len(features_to_select)):
    new_X_train = X_train.loc[:,list(features_to_select[i])]
    new_X_test = X_test.loc[:,list(features_to_select[i])]
    model, param_grid = functions.choose_model(models[i])
    trained_model = functions.model_training(model,new_X_train, Y_train, param_grid)
    #trained_model = model.fit(new_X_train, Y_train)
    filename = functions.save_model(trained_model,name,data_name)
    loaded_model = functions.scorer(filename, new_X_test, Y_test)
