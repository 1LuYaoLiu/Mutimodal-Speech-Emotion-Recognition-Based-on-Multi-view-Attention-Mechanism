import os
import csv
import time
import datetime
import random
import json

import warnings
from collections import Counter
from bs4 import BeautifulSoup
from math import sqrt

import gensim
import pandas as pd
import numpy as np
import pickle

'''配置数据处理相关参数'''


class TrainingConfig(object):
    epoches = 4
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001


class ModelConfig(object):
    embeddingSize = 200

    hiddenSizes = [256, 128]  # LSTM结构的神经元个数

    dropoutKeepProb = 0.5
    l2RegLambda = 0.0

class Config(object):
    sequenceLength = 371  # 取所有语句的最长长度
    batchSize = 128
    dataSource = r'H:/MSP_IMPROV/trans_norm.csv'
    numClasses = 4  # 分类类别数量

    training = TrainingConfig()
    model = ModelConfig()

config = Config()

'''数据预处理的类，生成word2vec特征数据集'''
class data_set(object):
    def __init__(self, config):
        self.config = config
        self._sequenceLength = config.sequenceLength
        self._dataSource = config.dataSource
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize

        self.wordEmbedding = None

    def _readData(self, filePath):
        """
        从csv文件中读取数据集;
        去掉标点，转小写
        """
        df = pd.read_csv(filePath)
        review = df["trans"].tolist()
        review = [line.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',','').replace(
            '.', '').replace('?', '').replace('(', '').replace(')', '') for line in review]

        reviews = [line.strip().split() for line in review]
        reviews = [[w.lower() for w in word] for word in reviews]

        return reviews

    def cleanReview(self, words):
        # 数据处理函数
        newSubject =words.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',','').replace(
            '.', '').replace('?', '').replace('(', '').replace(')', '')

        return newSubject

    def _wordToIndex(self, reviews, word2idx):
        """
        将词转换成索引
        """
        reviewIds = [[word2idx.get(item, word2idx["UNK"]) for item in review] for review in reviews]
        return reviewIds

    def _genVocabulary(self, reviews):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """

        allWords = [word for review in reviews for word in review]

        # 去掉停用词
        # subWords = [word for word in allWords if word not in self.stopWordDict]

        wordCount = Counter(allWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >= 5]

        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding

        word2idx = dict(zip(vocab, list(range(len(vocab)))))

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("./data/wordJson/MSP_word2idx.json", "w", encoding="utf-8") as f:
            json.dump(word2idx, f)

        return word2idx

    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """

        wordVec = gensim.models.KeyedVectors.load_word2vec_format("./word2Vec.bin", binary=True)
        vocab = []
        wordEmbedding = []

        # 添加 "pad" 和 "UNK",
        vocab.append("PAD")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))

        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")

        return vocab, np.array(wordEmbedding)

    def _get_feature_data(self, x, word2idx):
        '''
        生成用于训练的特征数据
        '''
        reviews = []
        for review in x:
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]]*(self._sequenceLength-len(review)))

        return reviews

    def _pre_train_word2vec(self,sentences):
        '''
        训练模型，词向量的长度设置为200， 迭代次数为8，采用skip-gram模型，模型保存为bin格式
        '''
        model = gensim.models.Word2Vec(sentences, size=200, sg=1, iter=8)
        model.wv.save_word2vec_format("./word2Vec" + ".bin", binary=True)





data = data_set(config)
# 初始化数据集
reviews = data._readData(config.dataSource)
data._pre_train_word2vec(reviews)

# 初始化词向量矩阵
word2idx = data._genVocabulary(reviews)
# 将句子数值化
reviewsIds = data._wordToIndex(reviews, word2idx)
# 加padding至128（最大序列长度）
reviewsIds = data._get_feature_data(reviewsIds, word2idx)

df = pd.read_csv(Config.dataSource)
trans_ids = {}
for i in range(len(reviewsIds)):
    trans_ids[df.iloc[i,0]] = reviewsIds[i]

with open('./MSP_wordEmbedding.plk', 'wb') as f:
    pickle.dump(data.wordEmbedding, f)

with open('./MSP_trans_ids.plk', 'wb') as f:
    pickle.dump(trans_ids, f)









