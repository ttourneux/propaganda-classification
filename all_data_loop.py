import functions
import datasets
from sklearn.model_selection import train_test_split


# In[3]:


def all_data_loop():
    DS = datasets.Datasets()
    all_data = DS.all_data_df
    data_name = "all_data"## to be changed
    
    data = all_data
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
    
        
all_data_loop()
