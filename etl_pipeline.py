# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv('Data/messages.csv')

# load categories dataset
categories = pd.read_csv('Data/categories.csv')

# merge datasets
df = pd.merge(messages, categories, on='id')

# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';', expand=True)

# select the first row of the categories dataframe
row = categories.iloc[0,:]

#Extracting the number of categories and keeping just the name
category_colnames = row.apply(lambda x: x[:-2])
print(category_colnames)

# rename the columns of `categories`
categories.columns = category_colnames

#transforming the data to binomial
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])


# drop the original categories column from `df`
df=df.drop(columns=['categories'])

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df,categories],axis=1)

# check number of duplicates
ind_duplicates = df.duplicated('message')
print(ind_duplicates.sum())

# drop duplicates
df = df.drop_duplicates('message')

# check number of duplicates
ind_duplicates = df.duplicated('message')
print(ind_duplicates.sum())

#Saving as sql table to the working directory
engine = create_engine('sqlite:///DisastersProject.db')
df.to_sql('Messages_Categories', engine, index=False)