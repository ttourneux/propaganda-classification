#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


## load dataset 


#mainDf = pd.read_csv(r'classifier_training_data (1).csv')
mainDf = pd.read_csv(r'data/inital_BERT_output.csv')


#kaggleDf = pd.read_csv(r'classifier_training_data (2).csv')
kaggleDf = pd.read_csv(r'data/kaggle_BERT_output.csv')


#df.head(5)


# In[3]:

## 0 instances of propaganda where there is no data
kaggleDf = kaggleDf.fillna(0)
mainDf = mainDf.fillna(0)


# In[4]:


kaggleDf = kaggleDf.assign(Content_Category=lambda Candidate	: 'tweet')
## all data from kaggle is a tweet

# In[5]:


everythingDf = mainDf


# In[6]:


kaggleDf =kaggleDf.assign(
    BCandidate = lambda dataframe : dataframe['Candidate'].map(lambda Candidate: 1 if Candidate == "DT" else 0)
)
kaggleDf
#everythingDf.append(kaggleDf)


# In[7]:


everythingDf =everythingDf.assign(
    BCandidate = lambda dataframe : dataframe['Candidate'].map(lambda Candidate: 1 if Candidate == "DT" else 0)
)## we want to make two numerical categories
everythingDf


# In[8]:


## a vectorized approach to combining data
# loop through every row turning it into a vector 
# if the filename of the vector is the same as the last row in our data, add the column values.
#df = everythingDf.loc[everythingDf["File Name"]== "HC0_500.txt"]


# In[9]:


everythingDf = pd.DataFrame(everythingDf.groupby(["File Name", "BCandidate", "Content_Category"], as_index = False).sum())## we combine instances of propaganda(rows) together by filename (grouping propaganda by complete input file)
#everythingDf["File Name"]
assert(len(everythingDf["File Name"].unique()) == len(everythingDf["File Name"]))

# In[10]:

assert(len(kaggleDf['Sentence'].unique()) == len(kaggleDf['Sentence']))
## make sure each sentance/tweet is grouped together as above.
everythingDf = everythingDf.append(kaggleDf)
## each row is a tweet 


# In[11]:

everythingDf = everythingDf.reindex(columns = ["File Name",
                               "BCandidate",                               
                               "Content_Category",
                               "Num Prop",                 
                               "Loaded_Language",
                               
  "Name_Calling,Labeling",
  "Repetition",
  "Exaggeration,Minimisation",
  "Doubt",
  "Appeal_to_fear-prejudice",
  "Flag-Waving",
  "Causal_Oversimplification",
  "Slogans",
  "Appeal_to_Authority",
  "Black-and-White_Fallacy",
  "Thought-terminating_Cliches",
  "Whataboutism,Straw_Men,Red_Herring",
  "Obfuscation,Intentional_Vagueness,Confusion",
  "Bandwagon,Reductio_ad_hitlerum"])


nonTwitterDf = everythingDf.loc[everythingDf['Content_Category']!='tweet',:]
nonTwitterDf = nonTwitterDf.loc[nonTwitterDf['Content_Category']!='retweet',:]
len(nonTwitterDf)


# In[12]:


everythingDf["File Name"]


# In[13]:


class Datasets: 
  def __init__(self): 
    self.tweets_df =everythingDf.loc[everythingDf['Content_Category']=='tweet',:]
    self.retweets_df =  everythingDf.loc[everythingDf['Content_Category']=='retweet',:]
    self.non_twitter_df = nonTwitterDf
    self.all_data_df = everythingDf
print("done collecting data")

