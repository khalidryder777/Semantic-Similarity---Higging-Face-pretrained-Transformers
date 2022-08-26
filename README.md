# Semantic-Similarity---Hugging-Face-pretrained-Transformers
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
We also converted our result into a list and did list unpacking to make it more readable
```python
    score = cosine_similarity(embeddings[[0]],embeddings[[1]])
    
    s = score.tolist()       
    x.append(s[0][0])  
```
## 7. 

