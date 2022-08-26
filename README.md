# Semantic-Similarity-Using-Hugging-Face-Transformers
We tested the veracity of 6 promising Semantic Similarity Sentence Transformer models provided by Hugging Face with respect to ‘the human ratings paper and datasets’ 
In our project we calculated the similarity between two documents where one of the document is an elaboration of certain scene/s and could contain multiple sentences and the other document is the abstraction/distillation of the same scene/s.

## 1. Download/install required libraries and modules:
We used Jupyter Notebook from Anaconda distribution here. Please type the following in a notebook in Jupyter Notebook to install the libraries. Alternatively, the libraries can also be installed using the Conda terminal.
```python
pip install numpy
pip install pandas
pip install -U sentence-transformers
pip install -U scikit-learn
```

## 2.Import the required libraries and dependencies:
```python
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
```
## 3. Instantiate the Transformer object:
```python
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
```

## 4. Loop throught the dataframe to create a list with sentence pairs:
```python
for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,1][i])
    sentences.append(df.iloc[:,2][i])
```
## 5. Encoding the sentence pair using our Transformer object
```python
    embeddings = model.encode(sentences)
```
## 6. Calculating Cosine Similarity from sklearn library:
We also converted our result into a list and did list unpacking to make it more readable.
Copy the list values x into a suitable variable
```python
    score = cosine_similarity(embeddings[[0]],embeddings[[1]])
    
    s = score.tolist()       
    x.append(s[0][0])  
    ES_sim1 = x
```
## 7. Do the same thing for the remaining 3 sentence pairs:
```python
y = []
for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,3][i])
    sentences.append(df.iloc[:,4][i])
    
    embeddings = model.encode(sentences)
    score = cosine_similarity(embeddings[[0]],embeddings[[1]])#[0]
    
    s = score.tolist()       
    y.append(s[0][0])  
EP_sim1 = y

z = []
for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,5][i])
    sentences.append(df.iloc[:,6][i])
    
    embeddings = model.encode(sentences)
    score = cosine_similarity(embeddings[[0]],embeddings[[1]])#[0]

    s = score.tolist()       
    z.append(s[0][0])  
AS_sim1 = z

a = []
for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,7][i])
    sentences.append(df.iloc[:,8][i])

    embeddings = model.encode(sentences)
    score = cosine_similarity(embeddings[[0]],embeddings[[1]])#[0]
    
    s = score.tolist()       
    a.append(s[0][0])  
AP_sim1 = a
```

## 8. Repeat from step 3 for the second model we're going to use:
```python
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
ES_sim2 = b

c = []
for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,3][i])
    sentences.append(df.iloc[:,4][i])
    
    embeddings = model.encode(sentences)
    score = cosine_similarity(embeddings[[0]],embeddings[[1]])
    
    s = score.tolist()       
    c.append(s[0][0])  
EP_sim2 = c

d = []
for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,5][i])
    sentences.append(df.iloc[:,6][i])

    embeddings = model.encode(sentences)
    score = cosine_similarity(embeddings[[0]],embeddings[[1]])
   
    s = score.tolist()       
    d.append(s[0][0])  
AS_sim2 = d

e = []
for i in range(len(df)):
    sentences = []
    sentences.append(df.iloc[:,7][i])
    sentences.append(df.iloc[:,8][i])

    embeddings = model.encode(sentences)
    score = cosine_similarity(embeddings[[0]],embeddings[[1]])#[0]
    
    s = score.tolist()       
    e.append(s[0][0])  
AP_sim2 = e
```

## 9. Now, create a dataframe encompassing all the pair results of both the models:
```python
OP_data = pd.DataFrame(list(zip(ES_sim1, EP_sim1, AS_sim1, AP_sim1, ES_sim2, EP_sim2, AS_sim2, AP_sim2)),
               columns =['ES_sim1', 'EP_sim1', 'AS_sim1', 'AP_sim1', 'ES_sim2', 'EP_sim2', 'AS_sim2', 'AP_sim2'])

```

## 10. Merge your original data with the respective results (similarity scores):
## Finally, convert the dataframe to CSV file
```python
result = pd.concat([df, OP_data], axis=1)
result.to_csv('df_sim', index=False)
```
