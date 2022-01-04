#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Description: 
Author: Tian Hesuo
Date: 2021-12-24 14:47:00
LastEditTime: 2022-01-04 14:44:27
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

import jieba
import jieba.posseg as pseg
from numpy.lib.function_base import extract
# jieba.enable_paddle()
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
import random
import os
import re

# 常量设置
THEME_NUMS = 10

def stopwords_func(stopwords_path):
    CUSTOM_STOP_WORDS = ["\u3000", "\n", " "]
    # 加载停用词
    with open(stopwords_path, mode="r", encoding="utf8") as f:
        sw = f.read().splitlines()
        for word in CUSTOM_STOP_WORDS:
            sw.append(word)
    return sw

def get_extra_stopwords(text, _paddle=False):
    extra_stopwords = []  # 人名列表
    word_list = jieba.lcut(text)
    for word in word_list:
        if len(word)==1:  # 不加判断会爆
            continue
        if re.search(r"^\d*$", word):
            extra_stopwords.append(word)
        words = pseg.lcut(word, use_paddle=_paddle)  # paddle模式1 
        word, flag = list(words)[0]
        if _paddle:
            if flag=='nr' or flag == "ns":  # ns：地名， nr：人名
                extra_stopwords.append(word)
        else:
            if flag=='LOC' or flag == "PER":  # LOC：地名， PER：人名
                extra_stopwords.append(word)
    extra_stopwords = list(set(extra_stopwords))
    return extra_stopwords

def load_data(corpus_path, random_status=False, random_nums=10000):
    # 加载原始数据
    print("加载原始数据")
    with open(file=corpus_path, mode="r", encoding="utf8") as f:
        corpus_data = [json.loads(text) for text in f.readlines()]
    corpus_nums = len(corpus_data)
    
    # 随机选取
    if random_status:
        random_data = []
        for i in random.sample(range(0, corpus_nums), random_nums):
            random_data.append(corpus_data[i])
        corpus_data = random_data
    return corpus_data

def LDA_(corpus_path, stopwords_path, save_dirpath, random_status=True, random_nums=10000):
    stopwords = stopwords_func(stopwords_path)
    corpus_data = load_data(corpus_path, random_status=random_status, random_nums=random_nums)
    corpus = []
    title_list = []
    # 对数据进行分词处理
    jieba.load_userdict("data/userdict.txt")

    extra_stopwords = []
    for text in corpus_data:
        cluster_text = " ".join(jieba.lcut(text["cluster_text"]))
        title_list.append(text["title"])
        corpus.append(cluster_text)
        extra_stopwords.extend(get_extra_stopwords(text["cluster_text"]))
    # 额外补充地名、人名作为停用词
    stopwords.extend(extra_stopwords)
    
    countvector = CountVectorizer(stop_words=stopwords, max_df=0.8)
    cntTF = countvector.fit_transform(corpus)
    vocs = countvector.get_feature_names()

    # lda分析
    print("LDA模型训练")
    cpu_core_nums = multiprocessing.cpu_count()
    lda = LatentDirichletAllocation(n_topics=THEME_NUMS, max_iter=300, learning_method="online", learning_offset=50, n_jobs=6, random_state=9)
    docres = lda.fit_transform(cntTF)
    LDA_corpus = np.array(docres)
    print("模型训练完成！")

    LDA_corpus_one = np.zeros([LDA_corpus.shape[0]])
    # 对比所属两个概率的大小，确定属于的类别
    LDA_corpus_one = np.argmax(LDA_corpus, axis=1) # 返回沿轴axis最大值的索引，axis=1代表行；最大索引即表示最可能表示的数字是多少
    # print('每个文档所属类别：', LDA_corpus_one)

    if not os.path.exists(save_dirpath):
        os.makedirs(save_dirpath)
        
    for i in range(THEME_NUMS):
        # 选择相同主题进行存储
        theme_list = []
        for j, num in enumerate(LDA_corpus_one):
            if num == i:
                theme_list.append(corpus_data[j])
        nums_ = "cluster_title_subject_" + str(i) + ".json"
        theme_name = os.path.join(save_dirpath, nums_)
        with open(file=theme_name, mode="w", encoding="utf8") as f:
            f.write(json.dumps(theme_list, ensure_ascii=False))
    save_title_path = os.path.join(save_dirpath, "subject_title_terms.json")
    with open(file=save_title_path, mode="w+", encoding="utf8") as f:
        # 打印每个单词的主题的权重值
        tt_matrix = lda.components_
        for id, tt_m in enumerate(tt_matrix):
            tt_dict = [(name, tt) for name, tt in zip(vocs, tt_m)]
            tt_dict = sorted(tt_dict, key=lambda x: x[1], reverse=True)
            # 打印权重值大于0.6的主题词：
            tt_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 0.6]
            # 打印每个类别前5个主题词：
            tt_dict = tt_dict[:10]
            # print('主题%d:' % (id), tt_dict)
            save_string = '主题'+str(id)+":"+str(tt_dict)
            f.write(json.dumps(save_string, ensure_ascii=False))
            f.write("\n")
            
        f.write("\n")
        f.write("-----主题划分-----")
        f.write("\n")
        
        subject = []
        for i in range(10):
            title = "主题"+str(i)
            subject.append({title:[]})
        for i, text in enumerate(LDA_corpus_one):
            subject[text]["主题"+str(text)].append(title_list[i])
        for text in subject:
            f.write(json.dumps(text, ensure_ascii=False))
            f.write("\n")
            
            

if __name__ == "__main__":
    corpus_path = "data/gw_cluster_head.json"
    stopwords_path = "data/stopwords.txt"
    save_dirpath = "outputs/LDA"
    LDA_(corpus_path, stopwords_path, save_dirpath, random_status=True, random_nums=20000)
