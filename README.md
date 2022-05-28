# README

- Twitter_loop.py
       - this is a script that will: 
       1. load in the the cleaned dataset from datasets.py
       2. select which parts of the data to train on 
           a. choose from "tweets_df","retweets_df","non_twitter_df", or "all_data_df"
       3. the data is balanced so predicting the majority class will yield ~50% accuracy
       4. split the dataset into training and testing dataframes
       5. train the the ML models using functions from functions.py
       6. print/show results of our model tested on the testing portion of our data
       
       
       - Executable like changes that can be made in the script: 
       1. change which dataframes are selected from the datasets.Datasets() object
       2. change the data_name to describe what data is being trained/scored
       3. we can change which models should be trained. 
       
       
       
- functions.py
    - balance_data(data): will make sure that our data is 50% of Clinton examples and 50% Trump examples
        - should print DT length ratio of about 50% 
        - returns a balanced dataframe
        
    - choose_model(name,granularity = 1)
        - name is the name of the model that should be trained 
        - returns the model to be trained and the associated parameter grid that should be searched over. 

    
    - model_training (model, X_train,Y_train,param_grid)
        - model is the model to be trained 
        - param_grid
        ....
    
    
- scores/scores.json
    - this file contains the most recent accuracy and AUC of the model trained
    
- trained_models
    - this is a folder with the trained models that have been saved as pickled objects.
    
    
- Other files are just temporary files that have acted as scratch work for the above files and functions.


