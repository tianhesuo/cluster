#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Description: LDA主题分类模型
Author: Tian Hesuo
Date: 2021-12-24 14:47:00
LastEditTime: 2021-12-28 17:53:58
'''

import jieba
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
import random

# 常量设置
THEME_NUMS = 10
meta_ = "data/meta_data/gw_meta.json"

# 加载原始数据
print("加载原始数据")
with open(file=meta_, mode="r", encoding="utf8") as f:
    meta_data = json.loads(f.readlines()[0])
    meta_nums = len(meta_data)
print(f"原始数据加载完成, 总数目为{meta_nums}")

# ——————————————————————————————————————
# 提取标题
title = []
for text in meta_data:
    text_dict = {"text_combine": text["title"]}
    title.append(text_dict)
meta_data = title
# ——————————————————————————————————————
# # 随机选取
# random_data = []
# for i in random.sample(range(0, 87984), 10000):
#     random_data.append(meta_data[i])
# meta_data = random_data

# 对数据进行分词处理
jieba.load_userdict("data/meta_data/userdict.txt")
process_data = []
for i, data in enumerate(meta_data):
    text = data['text_combine']
    process_data.append(" ".join(jieba.lcut(text)))
print("对数据分词完成！")
    
# 加载停用词
with open("data/meta_data/stopwords.txt", mode="r", encoding="utf8") as f:
    sw = f.read().splitlines()
    
# 构建词频矩阵
corpus = process_data
countvector = CountVectorizer(stop_words=sw)
cntTF = countvector.fit_transform(corpus)
vocs = countvector.get_feature_names()
# print(vocs)

# lda分析
print("LDA模型训练")
cpu_core_nums = multiprocessing.cpu_count()
lda = LatentDirichletAllocation(n_topics=THEME_NUMS, max_iter=100, learning_method="online", learning_offset=50, n_jobs=6, random_state=0)
docres = lda.fit_transform(cntTF)
LDA_corpus = np.array(docres)
print("模型训练完成！")

LDA_corpus_one = np.zeros([LDA_corpus.shape[0]])
# 对比所属两个概率的大小，确定属于的类别
LDA_corpus_one = np.argmax(LDA_corpus, axis=1) # 返回沿轴axis最大值的索引，axis=1代表行；最大索引即表示最可能表示的数字是多少
print('每个文档所属类别：', LDA_corpus_one)

for i in range(THEME_NUMS):
    # 选择相同主题进行存储
    theme_list = []
    for j, num in enumerate(LDA_corpus_one):
        if num == i:
            theme_list.append(meta_data[j])
    theme_name = "data/LDA_data/cluster_title_subject_" + str(i) + ".json"
    with open(file=theme_name, mode="w", encoding="utf8") as f:
        f.write(json.dumps(theme_list, ensure_ascii=False))

with open("data/LDA_data/subject_title__terms.json", mode="w", encoding="utf8") as f:
    # 打印每个单词的主题的权重值
    tt_matrix = lda.components_
    for id, tt_m in enumerate(tt_matrix):
        tt_dict = [(name, tt) for name, tt in zip(vocs, tt_m)]
        tt_dict = sorted(tt_dict, key=lambda x: x[1], reverse=True)
        # 打印权重值大于0.6的主题词：
        tt_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 0.6]
        # 打印每个类别前5个主题词：
        tt_dict = tt_dict[:10]
        print('主题%d:' % (id), tt_dict)
        save_string = '主题'+str(id)+":"+str(tt_dict)
        f.write(json.dumps(save_string, ensure_ascii=False))
        f.write("\n")
