#coding:utf-8

"""
@author: zhouzhenyu
@contact: zhouzy1293@foxmail.com
@time: 2018/9/18 10:03
"""
import numpy as np
import pathlib
from sklearn.preprocessing import normalize
import pandas as pd
from scipy.sparse import csr_matrix
import random

def read_concept_dict():
    concept_dict = {}
    file = 'data/cos-refD/concepts_names.txt'
    with open(file,encoding='utf-8') as f:
        idx_num = 0
        for line in f:
            concept_dict.setdefault(idx_num,line.strip())
            idx_num += 1
    return concept_dict


def read_file(dataset):
    course_file = 'data/' + dataset + '.lsvm'
    if dataset == 'ruc_key':
        prereq_file = 'data/ruc.link'
    else:
        prereq_file = 'data/' + dataset + '.link'
    print(prereq_file)
    # 第一步首先把课程的bag-of-concept文件读取为一个稀疏矩阵
    concept = []
    if not pathlib.Path(course_file).exists():
        X = pd.read_csv('data/' + dataset + '.csv', index_col=0,encoding='ISO-8859-1')
        concept = list(X.columns)
        X = csr_matrix(X)

    else:
        row = []
        word = []
        data = []
        with open(course_file, 'r') as f:
            for i in f:
                tmp = i.strip().split(' ')
                for j in tmp[1:]:
                    j = j.split(':')
                    row.append(int(tmp[0]) - 1)
                    word.append(int(j[0]))
                    data.append(int(j[1]))
        word_index = list(set(word))
        word_index.sort()
        col = list(map(lambda x: word_index.index(x), word))

        X = csr_matrix((data, (row, col)))

    links = np.loadtxt(prereq_file, delimiter=' ')
    if 'ruc' in 'ruc_key':
        return (X, links, concept)
    else:
        return (X, links)


## 数据预处理
def row_normlize(X):
    return (normalize(X, norm='l1', axis=1))

#def tf_idf(X):


## 根据课程间先修关系，来生成(i,j,k)三元组，其中i表示第i门课，j表示课程i的先修，k表示非课程i的先修


def inlinks(x, links):
    for i in links:
        if (sum(i == x) == len(x)):
            return 1
            break
    return 0


def generate_trn(links, n):
    # 生成课程i与课程j的先修关系示性矩阵。
    # return: 一个2维的np.array，[[x,y,1],[x,z,0]]，前者表示存在先修关系，后者便是先修关系不存在。
    trn = np.zeros((n * (n - 1), 3), dtype=np.int)
    k = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j: continue
            trn[k, :2] = [i, j]
            k += 1

    for i in range(len(trn)):
        trn[i, 2] = inlinks(trn[i, :2], links)
    return trn


def generate_triple(trn,sample_size = None):
    # 生成课程的三元数对
    # return: 一个2维的np.array,类似[[1,2,4],[1,2,5]]，其中1为2的先修，4不为2的先修。???
    triple = np.array([[0, 0, 0]])
    pos_prelink = trn[trn[:, 2] == 1,]
    neg_prelink = trn[trn[:, 2] == 0,]
    for i in np.unique(trn[:, 1]):
        pos_pool = pos_prelink[pos_prelink[:, 0] == i, 1]
        # print(pos_pool)
        if not len(pos_pool): continue
        neg_pool = neg_prelink[neg_prelink[:, 0] == i, 1]
        # print(neg_pool)
        v2, v3 = np.meshgrid(pos_pool, neg_pool)
        v2.shape = (np.size(v2), 1)
        v3.shape = (np.size(v3), 1)
        v1 = np.ones_like(v3)
        v1[:, :] = i
        i_triple = np.concatenate([v1, v2, v3], axis=1)
        if sample_size:
            n = len(i_triple)
            sample = []
            for i in range(int(sample_size*n)):
                sample.append(np.random.choice(range(n)))
            i_triple = i_triple[sample]
        triple = np.append(triple, i_triple, axis=0)
    return (triple[1:, :])

