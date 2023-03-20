model1 = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
model2 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model3 = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
model4 = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model5 = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
model6 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model7 = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')


### We are using model3 and model2 respectively

    model3: paraphrase-MiniLM-L6-v2 Rank 6
    model2: all-MiniLM-L6-v2 Rank 4

###  3 best performing models based on complete dataset are:

    model7: all-MiniLM-L12-v2 Rank 1
    model4: paraphrase-multilingual-MiniLM-L12-v2 Rank 2
    model5: all-distilroberta-v1 Rank 3
