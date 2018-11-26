import xlrd
import jieba
import gensim
import nltk
import pdb
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import sent_tokenize, word_tokenize


class NLPModel(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def fileReader(self):
        self.data = xlrd.open_workbook(self.file_path)
        self.table_cause = self.data.sheet_by_name('OpenText')
        self.cause_list = []
        self.comment_list = []
        self.correction_list = []
        # loop the excel to get all data
        for i in range(1,381):
            self.cause = self.table_cause.cell_value(i,10)
            self.comment = self.table_cause.cell_value(i,11)
            self.correction = self.table_cause.cell_value(i,13)
            if self.text2Token(self.cause): 
                self.cause_list.append(self.text2Token(self.cause))
            if self.text2Token(self.comment):
                self.comment_list.append(self.text2Token(self.comment))
            if self.text2Token(self.correction):
                self.correction_list.append(self.text2Token(self.correction))
        return self.cause_list, self.comment_list, self.comment_list

    def text2Token(self, text):
        token_list = []
        sent_tok = sent_tokenize(text)
        for sent in sent_tok:
            if word_tokenize(sent) != []:
                token_list += word_tokenize(sent)
        return token_list
    
    def wordCollector(self, text_list):
        word_dict = {}
        for sentence in text_list:
            for word in sentence:
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        return word_dict

    def wor2vecEmbedding(self,text_list):
        print('embedding word by using google pretrained model ... ')
        model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True, unicode_errors='ignore')
        print('complete load google pretrain model !!!')
        word2vec_dict = {}
        for word, vector in zip(model.vocab, model.vectors):
            word2vec_dict[word] = vector
        vector_list = []
        idx = 0
        for sentence in text_list:
            vector = 0
            for word in sentence:
                try:
                    vector += np.array(word2vec_dict[word])
                except KeyError:
                    continue
            vector_list.append((vector/len(sentence)).tolist())
        return vector_list

    def TFIDFEmbedding(self, text_list):
        print('embedding word by using TFIDF mdoel ... ')
        tfidf_word = TFIDF(min_df = 0,max_features=None,strip_accents='unicode',analyzer='word',ngram_range=(1,3),use_idf=1,smooth_idf=1,sublinear_tf=False,stop_words='english')
        tfidf_char = TFIDF(min_df = 0,max_features=None,strip_accents='unicode',analyzer='word',ngram_range=(1,3),use_idf=1,smooth_idf=1,sublinear_tf=False,stop_words='english')
        tfidf_word.fit(text_list)

        vector_list = tfidf.transform(text_list)
        return vector_list
    
    def KMeansClustering(self, vector_list):
        kmeans = KMeans(n_clusters = 10, random_state = 0).fit(vector_list)
        for i in range(len(kmeans.labels_)):
            print(kmeans.labels_[i], ' '.join(self.cause_list[i]))
        print('done')

if __name__ == '__main__':
    input_path = './P0087_VIN_1000_list-Warranty_claims.xlsx'
    warranty_data = NLPModel(input_path)
    cause_list, comment_list, correction_list = warranty_data.fileReader()
    vector_list = warranty_data.wor2vecEmbedding(cause_list)
    warranty_data.KMeansClustering(vector_list)