#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, BertEmbeddings, CharacterEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from typing import List
import pandas as pd
import os
import random
import xml.etree.cElementTree as ET
from sklearn.preprocessing import LabelEncoder
import nltk.data
from nltk import sent_tokenize
from flair.data_fetcher import NLPTaskDataFetcher
from pathlib import Path


# In[2]:


#!pip install flair


# In[4]:


import torch


data_folder = Path('FLAIR_data\\')

# load corpus containing training, test, and dev data
corpus: TaggedCorpus = NLPTaskDataFetcher.load_classification_corpus(data_folder,
                                                                     test_file='test.txt',
                                                                     dev_file='dev.txt',
                                                                     train_file='train.txt')


# In[121]:


# create label dictionary
label_dict = corpus.make_label_dictionary()


# In[122]:


# initialize embeddings to stack later
#bert_embedding = BertEmbeddings()
#char_embedding = CharacterEmbeddings()

# create StackedEmbedding object
#stacked_embeddings = StackedEmbeddings(embeddings=[bert_embedding, char_embedding])


# In[123]:


stacked_embeddings = [
    BertEmbeddings(),
    CharacterEmbeddings()
]


# In[124]:


#embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)


# In[ ]:





# In[125]:


# initialize document embedding by passing list of word embeddings
document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(stacked_embeddings,
                                                                     hidden_size=512,
                                                                     reproject_words=True,
                                                                     reproject_words_dimension=256,
                                                                     )


# In[126]:


# create text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)


# In[127]:


# initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)


# In[ ]:


trainer.train('FLAIR_data\\BERT',
              learning_rate=0.1,
              mini_batch_size=16,
              anneal_factor=0.5,
              patience=5,
              max_epochs=10)


# In[ ]:
