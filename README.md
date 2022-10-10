# Semantic-Text-Similarity-Using-Hugging-Face-Sentence-Transformers
We tested the veracity of 6 promising Semantic Similarity Sentence Transformer models provided by Hugging Face with respect to ‘the human ratings paper and datasets’ 
In our project we calculated the similarity between two documents where one of the document is an elaboration of certain scene/s and could contain multiple sentences and the other document is the abstraction/distillation of the same scene/s.

## So why SBERT?
Before we answer why SBERT/transformers, let’s try to understand what our requirements are:
1. We want to compute the semantic similarities between two documents meaningfully.
2. The documents can be of varying lengths, from a short sentence to a paragraph containing a couple of sentences.

### There were two ways we could go about this:
#### a. Measuring Similarity with classical non-contextual algorithms like:
    i) Bag of Words (Count Vectorizer and TFIDF Vectorizer)
    ii) Jaccard Similarity
   The above algorithms assume that similar texts have many words common in them, but it’s obviously not the case all of the time.
   One way we can tackle this problem is by using pre-trained word embeddings using methods such as Word2Vec, Glove, etc
   This allows better performance but the limitation of the above methods is that the actual words are used in similarity calculation without considering
   the context in which the words appear.
   Each word gets the same embedding vector irrespective of the context of the rest of the sentence in which it appears
   Thus, modern contextual algorithms are better suited for our task.

#### b. Measuring Similarity with classical contextual algorithms like
    i) BERT
    ii) SBERT
   BERT became the state-of-the-art language model by utilizing a self-supervised pre-training task called Masked Language Modeling where some words are
   kept hidden randomly and the model is trained to predict the missing words by providing it with words before and after the missing word. Doing this
   training process over a massive corpus of text data allows BERT to learn the semantic relationships between words in the language. Apart from this SBERT
   also uses Self-attention where the attention mechanism is applied between a word and all of the other words in its own context
   BERT produces very accurate similarity scores but it’s not scalable as finding the most similar pair in a collection of 10,000 sentences requires about
   50 million inference computations which roughly takes nearly 65 hours.
   SBERT solves the problem of inference computation by producing sentence embeddings instead of doing inference computation for very sentence pair
   comparison. SBERTdoes this by processing one sentence at a time and the apply mean pooling (BERT outputs token embeddings consisting of 512768-
   dimensional vectors. The mean pooling function compresses that data into a single 768-dimensional vector) on the final output layer to produce a sentence
   embedding. 
   Moreover, SBERT is fine-tuned on sentence pairs dataset/s using siamese architecture which can be thought of as running two identical BERTs in parallel
   that share the exact same network weights or tied weights


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
