from __future__ import print_function
import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from sklearn.cluster import KMeans
import os  # for os.path.basename
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import ward, dendrogram
import plotly.figure_factory as ff


# nltk.download('stopwords')


stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")


# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens



excel_data = pd.read_excel('100(all).xlsx', engine='openpyxl')

texts = excel_data['File structure']
titles = excel_data['Framework']


totalvocab_stemmed = []
totalvocab_tokenized = []

for i in texts:
    # print(i)
    allwords_stemmed = tokenize_and_stem(i)  # for each item in 'texts', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)



vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


# define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

# fit the vectorizer to texts
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)


print(tfidf_matrix)
# for x in tfidf_matrix:
#     print(x)



terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)


"""K Means Clustering"""

num_clusters = 3
km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

# print(clusters)
# joblib.dump(km,  'doc_cluster.pkl')
#
# km = joblib.load('doc_cluster.pkl')

# clusters = km.labels_.tolist()

projects = {'title': titles, 'file_structure': texts, 'cluster': clusters }

frame = pd.DataFrame(projects, index = [clusters] , columns = ['title', 'cluster'])

# number of films per cluster
# print(frame['cluster'].value_counts())



# print("Top terms per cluster:")
# print()
# # sort cluster centers by proximity to centroid
# order_centroids = km.cluster_centers_.argsort()[:, ::-1]
#
# for i in range(num_clusters):
#     print("Cluster %d words:" % i, end='')
#
#     for ind in order_centroids[i, :6]:  # replace 6 with n words per cluster
#         print(' %s' % vocab_frame.iloc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
#     print()  # add whitespace
#     print()  # add whitespace
#
#     print("Cluster %d titles:" % i, end='')
#     for title in frame.iloc[i]['title'].values.tolist():
#         print(' %s,' % title, end='')
#     print()  # add whitespace
#     print()  # add whitespace
#
# print()
# print()
#

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print()
print()



"""Visualizing document clusters"""

# set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#F9FF33', 3: '#3FFF33', 4: '#3383FF'}

# set up cluster names using a dict
cluster_names = {0: '1th', 1: '2th', 2: '3th', 3: '4th',
                 4: '5th'}


# create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

for index, row in df.iterrows():
    print(str(row['title']) + '  ---  ' + str(row['label']))


# group by cluster
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(17, 9))  # set size
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,

            mec='none')
    ax.set_aspect('auto')
    ax.tick_params( \
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params( \
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1)  # show legend with only 1 point

# add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['title'], size=8)

plt.show()  # show the plot



"""Hierarchical document"""

# define the linkage_matrix using ward clustering pre-computed distances
# linkage_matrix = ward(dist)
#
# fig, ax = plt.subplots(figsize=(150, 200))
# ax = dendrogram(linkage_matrix, orientation="right", labels=titles)
#
# plt.tick_params(\
#     axis= 'x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are off
#     top='off',         # ticks along the top edge are off
#     labelbottom='off')
#
# # show plot with tight layout
# plt.tight_layout()
#
# # uncomment below to save figure
# plt.savefig('ward_clusters.png', dpi=200)


# fig = ff.create_dendrogram(films, color_threshold=1.5)
# fig.update_layout(width=800, height=500)
# fig.show()