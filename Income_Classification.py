#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('D:\Data Science\DS\Datasets\Income Classification\income_evaluation.csv')


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.isnull().any()


# In[6]:


df = df.rename(columns={'age': 'age',' workclass': 'workclass',
                         ' fnlwgt': 'final_weight',
                         ' education': 'education',
                         ' education-num': 'education_num',
                         ' marital-status': 'marital_status',
                         ' occupation': 'occupation',
                         ' relationship': 'relationship',
                         ' race': 'race',
                         ' sex': 'sex',
                         ' capital-gain': 'capital_gain',
                         ' capital-loss': 'capital_loss',
                         ' hours-per-week': 'hrs_per_week',
                         ' native-country': 'native_country',
                         ' income': 'income'
                        })


# In[7]:


df.columns


# #Since “income” is our target variables, we want it to be numeric for ease of calculation. I’m going to create new variables derived from 'income'

# In[9]:


df['income'].values


# In[10]:


df['income'].unique()
df['income_encoded'] = [1 if value == ' >50K' else 0 for value in df['income'].values]
df['income_encoded'].unique()
# Let's check some descriptive statistics
df.describe()

#Observations from the above statistics:

1.In the dataset the mean and median age is similar, I guess it will be a normal distribution, we will check it later using visualizations.
2.The variables of capital gain and loss are suspect. All observations greater than 0 are in the 4th quartile.
3.In the “hrs_per_week” columns, the min is 1 and the max is 99, which is not common in real life. We will have to investigate this later.
4.Only about a quarter of the population can earn more than 50,000 a year.
# # Income Classification

# #Let’s see how each profession plays out by comparing the number of people earning over 50K.
# 
# We’ll look at the total number of workers for each area and the total number of people earning over 50K in each

# In[25]:


df[df['income'] == '>50K']['occupation'].value_counts().head(3)
pd.crosstab(df['occupation'],df['income']).plot(kind='barh',stacked=True,figsize=(10,10))


# # Observation:
# 
# 1.The 3 main occupations in total number are the professional speciality, home repair, executive management.
# 
# 2.The top 3 occupations in terms of a total number of people earning more than 50K (in order) are Executive, Occupational
# Specialties and Handicraft Sales and Repairs (with a close margin).
# 
# 3.Senior executives have the highest percentage of people earning more than 50,000 people: 48%.

# In[ ]:




