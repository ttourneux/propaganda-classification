#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


## load dataset 


mainDf = pd.read_csv(r'classifier_training_data (1).csv')

kaggleDf = pd.read_csv(r'classifier_training_data (2).csv')

#df.head(5)


# In[6]:


kaggleDf = kaggleDf.fillna(0)
mainDf = mainDf.fillna(0)


# In[7]:


kaggleDf = kaggleDf.assign(Content_Category=lambda Candidate: 'tweet')


# In[8]:


everythingDf = mainDf.append(kaggleDf)


# In[9]:

everythingDf =everythingDf.assign(
    BCandidate = lambda dataframe : dataframe['Candidate'].map(lambda Candidate: 1 if Candidate == "DT" else 0)
)
everythingDf

# In[11]:


nonTwitterDf = everythingDf.loc[everythingDf['Content_Category']!='tweet',:]
nonTwitterDf = nonTwitterDf.loc[nonTwitterDf['Content_Category']!='retweet',:]
len(nonTwitterDf)


# In[12]:





# In[16]:


class Datasets: 
  def __init__(self): 
    self.tweets_df =everythingDf.loc[everythingDf['Content_Category']=='tweet',:]
    self.retweets_df =  everythingDf.loc[everythingDf['Content_Category']=='retweet',:]
    self.non_twitter_df = nonTwitterDf
    self.all_data_df = everythingDf

