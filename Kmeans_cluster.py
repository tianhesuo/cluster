#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Description: 使用KMEANS方法进行聚类分析
Author: Tian Hesuo
Date: 2021-12-27 14:22:06
LastEditTime: 2021-12-29 15:46:45
'''
'''

　　┏┓　　　┏┓+ +
　┏┛┻━━━┛┻┓ + +
　┃　　　　　　　┃ 　
　┃　　　━　　　┃ ++ + + +
 ████━████ ┃+
　┃　　　　　　　┃ +
　┃　　　┻　　　┃
　┃　　　　　　　┃ + +
　┗━┓　　　┏━┛
　　　┃　　　┃　　　　　　　　　　　
　　　┃　　　┃ + + + +
　　　┃　　　┃
　　　┃　　　┃ +  神兽保佑
　　　┃　　　┃    代码无bug　　
　　　┃　　　┃　　+　　　　　　　　　
　　　┃　 　　┗━━━┓ + +
　　　┃ 　　　　　　　┣┓
　　　┃ 　　　　　　　┏┛
　　　┗┓┓┏━┳┓┏┛ + + + +
　　　　┃┫┫　┃┫┫
　　　　┗┻┛　┗┻┛+ + + +

'''

import json
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score, silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def stopwords(stopwords_path):
    CUSTOM_STOP_WORDS = ["\u3000", "\n", " "]
    # 加载停用词
    with open(stopwords_path, mode="r", encoding="utf8") as f:
        sw = f.read().splitlines()
        for word in CUSTOM_STOP_WORDS:
            sw.append(word)
    return sw

# 通过所选文本构建语料库
def build_corpus(corpus_path, stopwords):      
    corpus = []
    with open(file=corpus_path, mode="r", encoding="utf8") as f:
        corpus_data = json.loads(f.readlines()[0])
    for text in corpus_data[:1000]:
        text = " ".join(jieba.lcut(text["text_combine"]))
        corpus.append(text)
    vectorizer = CountVectorizer(stop_words=stopwords, max_df=0.8)
    vectorizer_fit = vectorizer.fit_transform(corpus)
    word_dict = {}
    for index, word in enumerate(vectorizer.get_feature_names()):
        word_dict[index] = word
    print("关键词的数量为{}".format(len(word_dict)))
    
    # print(vectorizer.vocabulary_)
    # print(vectorizer_fit.toarray())
    tfidf_ = TfidfTransformer()
    tfidf_fit = tfidf_.fit_transform(vectorizer_fit)
    return tfidf_fit, word_dict
    

def Kmeans_cluster(tfidf_fit):
    # kmeans.labels_   # 每个样本所属的簇
    # kmeans.inertia_  # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    # (kmeans.cluster_centers_.argsort()[:, ::-1]) # 找出关键词
    # 找出聚类的中心
    nums = []
    silhouette_score_ = []
    inertia_score_ = []
    pred_ = []
    for k in range(6, 15):
        kmeans = KMeans(n_clusters=k, random_state=9, n_jobs=6)
        pred = kmeans.fit_predict(tfidf_fit)
        
        silhouette = silhouette_score(tfidf_fit, pred)
        print("silhouette_score {} for cluster {}".format(silhouette, k))
        inertia = kmeans.inertia_
        print("inertia_score {} for cluster {}".format(inertia, k))
        
        nums.append(k)
        silhouette_score_.append(silhouette)
        inertia_score_.append(inertia)
        pred_.append(pred)

    #转成字典方便查找
    dict_silhouette = dict(zip(nums, silhouette_score_))
    dict_inertia_score = dict(zip(nums,inertia_score_))
    dict_pred = dict(zip(nums, pred_))
    score_dict = []
    score_dict.append(dict_silhouette)
    score_dict.append(dict_inertia_score)
    score_dict.append(dict_pred)
    
    with open("data/kmean/score.json", mode = "w", encoding="utf8") as f:
        f.write(json.dumps(score_dict, ensure_ascii=False))
    
    plt.figure()  
    plt.plot(nums, silhouette_score_)  
    plt.xlabel("kmeans-k")  
    plt.ylabel("silhouette_score")  
    plt.title("matrix")  
    plt.savefig("outputs/silhouette_score.png")
    plt.show()

    plt.figure()  
    plt.plot(nums,inertia_score_)  
    plt.xlabel("kmeans-k")  
    plt.ylabel("inertia_score(sum of squared)")  
    plt.title("matrix")  
    plt.savefig("outputs/inertia_score.png")
    plt.show()


if __name__ == "__main__":
    corpus_path = "data/meta_data/gw_cluster.json"
    stopwords_path = "/home/tianhesuo/work/knowledge_graph/cluster_algorithm/data/meta_data/stopwords.txt"
    sw = stopwords(stopwords_path)
    tfidf_fit, word_dict = build_corpus(corpus_path, sw)
    Kmeans_cluster(tfidf_fit)
    
