import pandas as pd
import numpy as np
import re
import os

from sklearn.feature_extraction.text import CountVectorizer

file_paths = []
file_names = []

def file_folder_split(dir):
    data = []
    for x in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, x)) and ("csv" in x):
            data.append(os.path.join(dir, x))
            file_names.append(x)
            #print("File ==> {0}".format(x))
        if os.path.isdir(os.path.join(dir, x)):
            #print("Folder ==> {0}".format(x))
            file_folder_split(os.path.join(dir, x))
    
    if data:
        file_paths.append(data)

file_folder_split('./articles_korean_with_date')

data_index = 0
for data_list in file_paths:
    partial_articles = []
    for data in data_list:
        try:
            articles = pd.read_csv(data, encoding = 'cp949', converters={'ART_ID': str})
        except UnicodeDecodeError:
            articles = pd.read_csv(data, encoding = 'utf8', converters={'ART_ID': str})
        
        partial_articles.append(articles)
        data_index += 1
    
    all_articles = partial_articles[0]
    for i in range(1, len(partial_articles)):
        all_articles = pd.concat([all_articles, partial_articles[i]])
    
    try:
        all_articles.to_csv('all_articles_korean.csv', index = False, encoding = 'cp949')
    except UnicodeEncodeError:
        all_articles.to_csv('all_articles_korean.csv', index = False, encoding = 'utf8')

    all_id = all_articles['ART_ID']
    del all_articles['ART_ID']
    all_id = all_id.reset_index()

    vec = CountVectorizer()
    text_vec = vec.fit_transform(all_articles.ART_CONTENT.values.astype(str))
    count_vec_df = pd.DataFrame(text_vec.todense(), columns = vec.get_feature_names())
    yearly_vec = pd.concat([all_id, count_vec_df], axis = 1)
    
    try:
        yearly_vec.to_csv('all_articles_korean_vec.csv', index = False, encoding = 'cp949')
    except UnicodeEncodeError:
        yearly_vec.to_csv('all_articles_korean_vec.csv', index = False, encoding = 'utf8')