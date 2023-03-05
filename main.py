#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing required modules
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import LineTokenizer
import glob
import re
import os
import pandas as pd
import docx2txt
import textract
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
from docx.api import Document
# from tabula import read_pdf
# from tabulate import tabulate
import numpy as np
from fastapi import FastAPI, HTTPException
from transformers import BertModel, BertTokenizer
from mangum import Mangum
nltk.download("stopwords")
nltk.download('punkt')


#pandas max columns and rows
pd.set_option('display.max_colwidth', None)
pd.set_option("display.max_rows", None)


# In[2]:


# change the File directory
file_list = glob.glob(os.path.join(os.getcwd(), "Source rules or OCD rulebook.csv"))
corpus = []
files = []

for file_path in file_list:
    with open(file_path, encoding="latin-1") as f_input:
        corpus.append(f_input.read())
        files.append(''.join([n for n in os.path.basename(file_path)]))

main_df = pd.DataFrame({'Document name':files, 'Document text':corpus})
main_df



# In[3]:


# change the File directory
file_list2 = glob.glob(os.path.join(os.getcwd(), "C:/Users/Omer/Desktop/Target Rules or EA rulebook.csv"))
corpus2 = []
files2 = []

for file_path2 in file_list2:
    with open(file_path2, encoding="latin-1") as f_input2:
        corpus2.append(f_input2.read())
        files2.append(''.join([n for n in os.path.basename(file_path2)]))

main_df2 = pd.DataFrame({'Document name':files2, 'Document text':corpus2})
main_df2


# In[4]:


# importing the stopwords using nltk library.
from nltk.corpus import stopwords
stop = stopwords.words('english')

custom_stopwords = ["ï","»","¿","â€“","â€™","a)","â","¿","–","–","b)","c)","d)","e)",":","(",")","â€˜","-",'must','used','using'
                   'near']
# punctation=[":","(",")"]
stop.extend(custom_stopwords)
# stop.extend(punctation)


# In[5]:


main_df['Clean_documents_rules'] = main_df['Document text'].astype(str).str.lower()
#main_df['Clean_documents_rules']= main_df['Clean_documents_rules'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
main_df


# In[6]:


def token_sent(text):   
    sent_tokens=LineTokenizer(blanklines='keep').tokenize(text)
    return sent_tokens


# In[7]:


main_df['Sentence_Tokenize_rules']=main_df['Clean_documents_rules'].apply(token_sent) 
main_df['Word_Tokenize_rules']=main_df['Clean_documents_rules'].apply(word_tokenize) 
main_df


# In[8]:


token_doc_name = dict(zip(main_df['Document name'], main_df['Sentence_Tokenize_rules']))
token_doc_name


# In[ ]:





# In[9]:


#Create an embedding for all the sentences in the documents
from sentence_transformers import SentenceTransformer

# all the embedding used for semantic search

#model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
#model = SentenceTransformer('ddobokki/electra-small-nli-sts')

#For Semantic textual similarity for small search queries
#model = SentenceTransformer('sentence-transformers/msmarco-distilroberta-base-v2')
model = SentenceTransformer('sentence-transformers/stsb-distilbert-base')
# sentence_embeddings = model.encode(sent_tokens)


# In[10]:


# finding the cosine similarity
def cosine_sim(embeddings1,embeddings2):
    """Cosine similarity metric function to calculate the distance between the two vectors."""
    cossim=( np.dot(embeddings1,embeddings2) )/ (np.linalg.norm(embeddings1)*np.linalg.norm(embeddings2))
    if np.isnan(np.sum(cossim)):
        return 0
    return cossim


# In[11]:


from itertools import chain
docs_sent_tokens=list(chain.from_iterable(main_df['Sentence_Tokenize_rules']))
#docs_name=main_df['Document name']


# In[12]:


app = FastAPI()
handler = Mangum(app)

@app.get("/vectorize")
async def vectorize(search_sentence: str):
    # Tokenize the sentence
    search_sentence_embeddings = (model.encode(search_sentence))
    results=[]

    # #set the threshold value to get the similairity result accordingly
    threshold=0.6

    # #embedding all the documents and find the similarity between search text and all the tokenize sentences
    # for name,docs_sent_token in zip(main_df['Document name'], docs_sent_tokens):
    for doc_name, docs_sent_tokens in token_doc_name.items():
        name=doc_name.title()
        for docs_sent_token in docs_sent_tokens:
            sentence_embeddings = model.encode(docs_sent_token)
            sim_score1 = cosine_sim(search_sentence_embeddings, sentence_embeddings)
            if sim_score1 > threshold:
                results.append((
                docs_sent_token,
                sim_score1,
                name
                ))
    #printing the top 10 matching result in dataframe format
    df=pd.DataFrame(results, columns=['Matching Sentence','Similarity Score','Document name'])

    # sorting in descending order based on the similarity score
    df.sort_values("Similarity Score", ascending = False, inplace = True)

    #change the value of n to see more results
    df.head(n=10)

    return df.to_dict()


# In[13]:


