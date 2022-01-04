#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Description: 使用KMEANS方法进行聚类分析
Author: Tian Hesuo
Date: 2021-12-27 14:22:06
LastEditTime: 2022-01-04 10:03:34
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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import jieba
import jieba.posseg as pseg
jieba.enable_paddle()
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

def get_extra_stopwords(text):
    extra_stopwords = []  # 人名列表
    word_list = jieba.lcut(text)
    for word in word_list:
        if len(word)==1:  # 不加判断会爆
            continue
        words = pseg.lcut(word, use_paddle=True)  # paddle模式
        print(list(words))
        word, flag = list(words)[0]
        if flag=='LOC' or flag == "PER":  # 这里写成LOC是地名
            extra_stopwords.append(word)
    extra_stopwords = list(set(extra_stopwords))
    return extra_stopwords


# 通过所选文本构建语料库
def build_corpus(corpus_path, stopwords):      
    corpus = []
    title_list = []
    with open(file=corpus_path, mode="r", encoding="utf8") as f:
        corpus_data = [json.loads(text) for text in f.readlines()]
        
    # 随机选取
    # random_data = []
    # for i in random.sample(range(0, 87984), 10000):
    #     random_data.append(corpus_data[i])
    # corpus_data = random_data
    
    jieba.load_userdict("data/userdict.txt")

    extra_stopwords = []
    for text in corpus_data:
        cluster_text = " ".join(jieba.lcut(text["cluster_text"]))
        title_list.append(text["title"])
        corpus.append(cluster_text)
        extra_stopwords.append(get_extra_stopwords(text["cluster_text"]))
    # 额外补充地名、人名作为停用词
    stopwords.extend(extra_stopwords)
        
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
    return tfidf_fit, word_dict, title_list
    
def Kmeans_cluster(tfidf_fit, word_dict, title_list, cluster_words_path):
    # kmeans.labels_   # 每个样本所属的簇
    # kmeans.inertia_  # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    # (kmeans.cluster_centers_.argsort()[:, ::-1]) # 找出关键词
    # 找出聚类的中心
    
    nums = []
    inertia_score_ = []
    pred_ = []
    kmeans_reslut = []

    f_clusterwords = open(cluster_words_path, mode="w", encoding="utf8")
    for k in range(10, 15):
        kmeans = KMeans(n_clusters=k, random_state=9, n_jobs=6)
        pred = kmeans.fit_predict(tfidf_fit)
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]  
        # silhouette = silhouette_score(tfidf_fit, pred)
        # print("silhouette_score {} for cluster {}".format(silhouette, k))
        inertia = kmeans.inertia_
        print("inertia_score {} for cluster {}".format(inertia, k))
        
        nums.append(k)
        # silhouette_score_.append(silhouette)
        inertia_score_.append(inertia)
        pred_.append(pred.tolist())
        
        # 添加聚类的词语
        f_clusterwords.write("聚类数量为"+str(k)+".................")
        f_clusterwords.write("\n")
        cluster = 0
        for ind in order_centroids: # 每个聚类选 50 个词
            words=[]
            for index in ind[:20]:
                words.append(word_dict[index])
            f_clusterwords.write(str(cluster)+'\t'+','.join(words)+'\n')
            cluster+=1
        f_clusterwords.write("\n")
    f_clusterwords.close()
    
    # 转成字典方便查找
    # dict_silhouette_score = dict(zip(nums, silhouette_score_))
    dict_inertia_score = dict(zip(nums,inertia_score_))
    kmeans_reslut = []
    for i, pre in enumerate(pred_):
        cluster_mark = i + 10
        kmeans_dict = {cluster_mark:[]}     # {cluster_mark:[{0:[]}, {1:[]}]}
        for j in range(cluster_mark):
            kmeans_dict[cluster_mark].append({j:[]})
        for k, text in enumerate(pre):
            kmeans_dict[cluster_mark][text][text].append(title_list[k])
        # kmeans_dict["dict_silhouette_score"] = dict_silhouette_score
        kmeans_dict["dict_inertia_score"] = dict_inertia_score
        kmeans_reslut.append(kmeans_dict)
    
    with open("outputs/Kmeans/Kmeans_result.json", mode = "w", encoding="utf8") as f:
        for text in kmeans_reslut:
            f.write(json.dumps(text, ensure_ascii=False))
            f.write("\n")
    
    # plt.figure()  
    # plt.plot(nums, silhouette_score_)  
    # plt.xlabel("kmeans-k")  
    # plt.ylabel("silhouette_score")  
    # plt.title("matrix")  
    # plt.savefig("outputs/Kmeans/silhouette_score.png")
    # plt.show()

    plt.figure()  
    plt.plot(nums,inertia_score_)  
    plt.xlabel("kmeans-k")
    plt.ylabel("inertia_score(sum of squared)")  
    plt.title("matrix")  
    plt.savefig("outputs/Kmeans/inertia_score.png")
    plt.show()


if __name__ == "__main__":
    corpus_path = "data/gw_cluster_head.json"
    stopwords_path = "/home/tianhesuo/work/knowledge_graph/cluster_algorithm/data/stopwords.txt"
    cluster_words_path = "outputs/Kmeans/cluster_keywords.txt"
    sw = stopwords(stopwords_path)
    tfidf_fit, word_dict, title_list = build_corpus(corpus_path, sw)
    Kmeans_cluster(tfidf_fit, word_dict, title_list, cluster_words_path)
    
