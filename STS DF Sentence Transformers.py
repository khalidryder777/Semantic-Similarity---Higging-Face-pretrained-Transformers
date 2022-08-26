#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[14]:


df = pd.read_csv('ego_text.csv', encoding='cp1252')
#df.info()


# In[4]:


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
x = []

for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,1][i])
    sentences.append(df.iloc[:,2][i])
   
    embeddings = model.encode(sentences)
    score = cosine_similarity(embeddings[[0]],embeddings[[1]])#[0]
    
    s = score.tolist()       
    x.append(s[0][0])  
#print(x)


# In[5]:


ES_sim1 = x


# In[6]:


y = []

for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,3][i])
    sentences.append(df.iloc[:,4][i])
    
    embeddings = model.encode(sentences)
    score = cosine_similarity(embeddings[[0]],embeddings[[1]])#[0]
    
    s = score.tolist()       
    y.append(s[0][0])  
    
#print(y)


# In[7]:


EP_sim1 = y


# In[8]:


z = []

for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,5][i])
    sentences.append(df.iloc[:,6][i])
    
    embeddings = model.encode(sentences)
    score = cosine_similarity(embeddings[[0]],embeddings[[1]])#[0]

    s = score.tolist()       
    z.append(s[0][0])  
    
#print(z)


# In[9]:


AS_sim1 = z


# In[10]:


a = []

for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,7][i])
    sentences.append(df.iloc[:,8][i])


    embeddings = model.encode(sentences)
    score = cosine_similarity(embeddings[[0]],embeddings[[1]])#[0]
    
    s = score.tolist()       
    a.append(s[0][0])  
    
#print(a)


# In[11]:


AP_sim1 = a


# In[12]:


data1 = pd.DataFrame(list(zip(ES_sim1, EP_sim1, AS_sim1, AP_sim1)),
               columns =['ES_sim1', 'EP_sim1', 'AS_sim1', 'AP_sim1'])


# In[15]:


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

b = []

for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,1][i])
    sentences.append(df.iloc[:,2][i])

    embeddings = model.encode(sentences)
    score = cosine_similarity(embeddings[[0]],embeddings[[1]])
    
    s = score.tolist()       
    b.append(s[0][0])  
    
#print(b)


# In[16]:


ES_sim2 = b


# In[17]:


c = []

for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,3][i])
    sentences.append(df.iloc[:,4][i])
    
    embeddings = model.encode(sentences)

    score = cosine_similarity(embeddings[[0]],embeddings[[1]])
    
    s = score.tolist()       
    c.append(s[0][0])  
    
#print(c)


# In[18]:


EP_sim2 = c


# In[19]:


d = []

for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,5][i])
    sentences.append(df.iloc[:,6][i])

    embeddings = model.encode(sentences)

    score = cosine_similarity(embeddings[[0]],embeddings[[1]])
    #print(score)
    s = score.tolist()       
    d.append(s[0][0])  
    
#print(d)


# In[20]:


AS_sim2 = d


# In[21]:


e = []

for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,7][i])
    sentences.append(df.iloc[:,8][i])

    
    embeddings = model.encode(sentences)

    score = cosine_similarity(embeddings[[0]],embeddings[[1]])#[0]
    
    s = score.tolist()       
    e.append(s[0][0])  
    
#print(e)


# In[22]:


AP_sim2 = e


# In[23]:


OP_data = pd.DataFrame(list(zip(ES_sim1, EP_sim1, AS_sim1, AP_sim1, ES_sim2, EP_sim2, AS_sim2, AP_sim2)),
               columns =['ES_sim1', 'EP_sim1', 'AS_sim1', 'AP_sim1', 'ES_sim2', 'EP_sim2', 'AS_sim2', 'AP_sim2'])


# In[24]:


result = pd.concat([df, OP_data], axis=1)


# In[25]:


result.to_csv('df_sim', index=False)


# In[ ]:





# In[ ]:




