import functions
import datasets
from sklearn.model_selection import train_test_split


# In[3]:


def all_data_loop(data_name):
    '''
    input: 
    data_name(string): can take values "twitter" or "all_data" or "non_twitter" representing which data the models should be trained on
    
    return: NA
    
    to be changed: 
    MDI feature importance and permuation feature importance can be shown for their respective algorithms by uncommenting lines 43 and 44
    a barplot and boxplot visualizing the data can also be shown by uncommenting lines 56 and 57
    
    '''


    #DS = datasets.Datasets()
    #all_data = DS.all_data_df
    #data_name = "all_data"## to be changed
    
    #data = all_data
    data = functions.get_data(data_name)
    
    data = functions.balance_data(data)
    
    X = data.loc[:,"Num Prop":"Bandwagon,Reductio_ad_hitlerum",]
    Y = data.loc[:,"BCandidate"]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = .2,shuffle=True)
   
    RF_ET_name = ['extra trees', 'random forest']
    for name in RF_ET_name:
        model, param_grid = functions.choose_model(name)
        trained_model = functions.model_training(model,X_train, Y_train, param_grid)
        filename = functions.save_model(trained_model,name,data_name)
        loaded_model = functions.scorer(filename, X_test, Y_test)
        #functions.graph_MDI_importance(X_train,Y_train,loaded_model,name, data_name)
        #functions.show_permutation_feature_importance(loaded_model, X_train, Y_train, name, data_name)
        print(name + ': Done!')

    
    SVM_LO_NN = ["support vector machine",'neural net','logistic regression']#'lasso', ## this cannot be used because it cannon predict by classification
    for name in SVM_LO_NN :
        model, param_grid = functions.choose_model(name)
        trained_model = functions.model_training(model,X_train, Y_train, param_grid)
        filename = functions.save_model(trained_model,name,data_name)
        loaded_model = functions.scorer(filename, X_test, Y_test)
        print(name + ': Done!')
    
    #functions.barchart(data, data_name)
    #functions.boxplotting(data)
    
        
#data_loop("all_data")
