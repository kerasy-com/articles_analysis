import pandas as pd
import numpy as np
import re
import os

from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Okt
okt = Okt()

file_paths = []
file_names = []

def file_folder_split(dir):
    for x in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, x)) and "csv" in x:
            file_paths.append(os.path.join(dir, x))
            file_names.append(x)
        if os.path.isdir(os.path.join(dir, x)):
            file_folder_split(os.path.join(dir, x))
file_folder_split('./articles_korean_with_date')

data_index = 0
for data in file_paths:
    print(data)
    try:
        articles = pd.read_csv(data, encoding = 'cp949', converters={'ART_ID': str})
    except UnicodeDecodeError:
        articles = pd.read_csv(data, encoding = 'utf8', converters={'ART_ID': str})
    articles.ART_ID = "a" + articles.ART_ID
    articles = articles[['ART_ID', 'ART_CONTENT', 'MONTH']].dropna(axis = 0)
    articles = articles.reset_index(drop = True)

    id = articles[['ART_ID', 'MONTH']]
    text = articles['ART_CONTENT'].values.tolist()

    # 1. For each sentence, imported the tagger(in this case, Twitter) and preprocessed.
    sentences_tag = []
    for sentence in text:
        morph = okt.pos(sentence, stem = True)
        sentences_tag.append(morph)

    # 2. Put in words that are noun/verb/adjective
    noun_adj_list = []
    for sentence1 in sentences_tag:
        words = ""
        for word, tag in sentence1:
            if tag in ['Noun','Adjective', 'Verb']:
                words += word + ", "
        noun_adj_list.append(words)

    noun_adj_df = pd.DataFrame(noun_adj_list, columns = ['ART_CONTENT_CLEAN'])
    # 3. Concated the ARTICLES_ID and saved the file.
    count_vec_df_concat = pd.concat([id, noun_adj_df], axis = 1)

    try:
        count_vec_df_concat.to_csv(data, index = False, encoding = 'cp949')
    except UnicodeEncodeError:
        count_vec_df_concat.to_csv(data, index = False, encoding = 'utf8')
    data_index += 1