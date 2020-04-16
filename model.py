# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:00:27 2020

@author: ARKADIP GHOSH
"""
import pandas as pd
import numpy as np
import pickle


movies_df = pd.read_csv('movies.csv',usecols=['movieId','title'],dtype={'movieId': 'int32', 'title': 'str'})
rating_df=pd.read_csv('ratings.csv',usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

movies_df.head()
movies_df.shape

rating_df.head()

df = pd.merge(rating_df,movies_df,on='movieId')
df.head()

movie_features_df=df.pivot_table(index='title',columns='userId',values='rating').fillna(0)
movie_features_df.head()

movie_features_df.shape


from scipy.sparse import csr_matrix
movie_features_df_matrix = csr_matrix(movie_features_df.values)

movie_features_df_matrix .shape
movie_features_df_matrix 


from sklearn.neighbors import NearestNeighbors


model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_features_df_matrix)

movie_features_df.shape

'''query_index = np.random.choice(movie_features_df.shape[0])

distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)


for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))
'''
pickle.dump(model_knn, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

pickle.dump(movie_features_df, open('model1.pkl','wb'))
model1 = pickle.load(open('model1.pkl','rb'))
#model1.head()
#model1.iloc[query_index,:]


model1.index[0]

model1.shape
    

query_index=str('Freedom for Us (1931)')
for i in range(len(model1)) : 
  if(model1.index[i]==query_index) :
      break
print(i)


query_index = np.random.choice(model1.shape[0])
query_index=9718
#(model1.iloc[query_index,:].values.reshape(1, -1)).shape
distances, indices = model.kneighbors(model1.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)

len(distances.flatten())

for i in range(0, 6):
    if i == 0:
        print('Recommendations for {0}:\n'.format(model1.index[query_index]))
    else:
        print('{0}: {1}'.format(i, model1.index[indices.flatten()[i]]))










