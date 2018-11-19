import xlrd
import jieba
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = xlrd.open_workbook(r'./P0087_VIN  1000 list-Warranty_claims.xlsx')
table_cause = data.sheet_by_name('OpenText')
comments_cause_list = []
comments_comments_list = []
comments_correction_list = []
#data processing .........
for i in range(1,300):
    comments_cause = table_cause.cell_value(i,10)
    comments_comments = table_cause.cell_value(i,11)
    comments_correction = table_cause.cell_value(i,13) 
    if len(comments_cause) != 0:
        comments_cause = comments_cause.lower().split(' ')
        if len(comments_cause) > 4:
            comments_cause = comments_cause[4:]
            if comments_cause[0][0] == 'T':
                comments_cause_list.append(comments_cause[1:])
            else:
                comments_cause_list.append(comments_cause)
        else:
            comments_cause_list.append(comments_cause)
    if len(comments_comments) != 0:
        comments_comments_list.append(comments_comments)
    if len(comments_correction) != 0:
        comments_correction_list.append(comments_correction)
# model building and training ........
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(comments_cause_list)]
model = Doc2Vec(documents, vector_size=6, window=2, min_count=0, workers=4)
# model testing after training ......
vector_list = []
vector_to_idx = {}
for item in comments_cause_list:
    vector = model.infer_vector(item)
    vector_list.append(vector)
    #vector_to_idx['_'.join(vector)] = item
# implementing K-means...
X = np.array(vector_list)
kmeans = KMeans(n_clusters = 10, random_state = 0).fit(X)
#for item in X:
for i in range(len(kmeans.labels_)):
    print(kmeans.labels_[i], ' '.join(comments_cause_list[i]))
print('done')
