import pandas as pd
import numpy as np
import re
import os

from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Okt
okt = Okt()

temp = []
temp2 = []
def file_folder_split_temp(dir):
    data = []
    for x in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, x)) and ("csv" in x):
            data.append(os.path.join(dir, x))
            temp2.append(x)
        if os.path.isdir(os.path.join(dir, x)):
            file_folder_split_temp(os.path.join(dir, x))
    
    if data:
        temp.append(data)
file_folder_split_temp('articles_korean_with_date')

data_index = 0
for data_list in temp:
    partial_articles = []
    year = data_list[0].split('\\')[1]
    print(year)
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
        all_articles.to_csv('all_articles_korean_' + str(year) + '.csv', index = False, encoding = 'cp949')
    except UnicodeEncodeError:
        all_articles.to_csv('all_articles_korean_' + str(year) + '.csv', index = False, encoding = 'utf8')

    all_id = all_articles['ART_ID']
    month = int(all_articles['MONTH'].mean())

    vec = CountVectorizer()
    text_vec = vec.fit_transform(all_articles.ART_CONTENT_CLEAN.values.astype(str))
    count_vec_df = pd.DataFrame(text_vec.todense(), columns = vec.get_feature_names())
    all_id = all_id.reset_index(drop=True)
    yearly_vec = pd.concat([all_id, count_vec_df], axis = 1)
    
    try:
        yearly_vec.to_csv('all_articles_korean_vec_' + str(year) + "_" + str(month) + '.csv', index = False, encoding = 'cp949')
    except UnicodeEncodeError:
        yearly_vec.to_csv('all_articles_korean_vec_' + str(year) + "_" + str(month) + '.csv', index = False, encoding = 'utf8')