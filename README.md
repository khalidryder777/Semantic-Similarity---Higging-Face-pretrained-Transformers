# Semantic-Similarity---Hugging-Face-pretrained-Transformers
We tested the veracity of 6 promising Semantic Similarity Sentence Transformer models provided by Hugging Face with respect to ‘the human ratings paper and datasets’ 
In our project we calculated the similarity between two documents where one of the document is an elaboration of certain scene/s and could contain multiple sentences and the other document is the abstraction/distillation of the same scene/s.
```
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
```
