from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import randomized_svd
import pandas as pd
import numpy as np
myvocabulary = ['life', 'learning']
corpus = {1: "The game of life is a game of everlasting learning", 2: "The unexamined life is not worth living", 3: "Never stop learning"}
tfidf = TfidfVectorizer(vocabulary = myvocabulary, ngram_range = (1,3))
tfs = tfidf.fit_transform(corpus.values())

feature_names = tfidf.get_feature_names()
corpus_index = [n for n in corpus]
df = pd.DataFrame(tfs.T.todense(), index=feature_names, columns=corpus_index)
print(df)
a = [[2,4,5,0,0], [5,3,2,0,0], [4,1,7,0,0], [5,2,5,0,0], [0,2,0,3,4], [0,0,0,6,4], [0,3,0,2,5], [0,1,0,8,8]]
U, Sigma, VT = randomized_svd(np.matrix(a),
                              n_components=15,
                              n_iter=5,
                              random_state=None)

print(U)
sigma_padded = []
for i in range(len(Sigma)):
    sigma_padded.append([])
    for j in range(len(Sigma)):
        if i == j:
            sigma_padded[i].append(Sigma[i])
        else:
            sigma_padded[i].append(0)

[print(row) for row in sigma_padded]
[print(row) for row in a]
print(sigma_padded)
#print(len(Sigma))
print(VT)

#TF-IDF - nejakou
#Matici z TF-IDF
#SVD
#Zmensena SVD
#Aplikace vektoru na SVD