# coding:utf-8

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


def read_file(data_file, link_file, concept_file, file_type='dense'):
    # 第一步首先把课程的bag-of-concept文件读取为一个稀疏矩阵
    concept = []
    if file_type == 'dense':
        X = pd.read_csv(data_file, index_col=0)
        concept = list(X.columns)
        X = csr_matrix(X)
    elif file_type == 'sparse_row_col_val':
        row, col, data = [], [], []
        with open(data_file, 'r') as f:
            for i in f:
                tmp = i.strip().split(',')
                row.append(int(tmp[0]))
                col.append(int(tmp[1]))
                data.append(int(tmp[2]))
        X = csr_matrix((data, (row, col)))
        idxs = []
        with open(concept_file, 'r') as f:
            for i in f:
                name, idx = i.strip().split('::')
                if idx not in idxs:
                    concept.append(name)
                    idxs.append(idx)
    else:
        row, word, data = [], [], []
        with open(data_file, 'r') as f:
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

    links = np.loadtxt(link_file, delimiter=' ')
    return (X, links, concept)

# 数据预处理


def row_normlize(X):
    return (normalize(X, norm='l1', axis=1))

# 根据课程间先修关系，来生成(i,j,k)三元组，其中i表示第i门课，j表示课程i的先修，k表示非课程i的先修


def inlinks(x, links):
    for i in links:
        if (sum(i == x) == len(x)):
            return 1
    return 0


def generate_trn(links, n, undirect=False):
    '''
    生成课程i与课程j的先修关系示性矩阵。
    input: 
        links: [i, j] 表示关系对i是j的先修。注意从0开始标号。
        n: 课程数
    return: 
        一个2维的np.array，[[x,y,1],[x,z,0]]，前者表示x是y的先修，后者表示x不是z的先修。
    '''
    if undirect:
        links_inverse = np.array([[i[1],i[0]] for i in links])
        links = np.concatenate([links,links_inverse])
    trn = np.zeros((n * (n - 1), 3), dtype=np.int)
    k = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            trn[k, :2] = [i, j]
            k += 1

    for i in range(len(trn)):
        trn[i, 2] = inlinks(trn[i, :2], links)
    return trn


def generate_triple(trn, sample_size=None, undirect=False):
    '''
    生成课程的三元数对(i,j,k)
    input:
        trn: 2维的np.array，generate_trn的输出结果。
        sample_size: 是否要做抽样
    return:
        一个2维的np.array,类似[[1,2,4],[1,2,5]]，其中1为2的先修，1不为4的先修。
    '''
    triple = np.array([[0, 0, 0]])
    pos_prelink = trn[trn[:, 2] == 1, ]
    neg_prelink = trn[trn[:, 2] == 0, ]
    for i in np.unique(trn[:, 1]):
        pos_pool = pos_prelink[pos_prelink[:, 0] == i, 1]
        if not len(pos_pool):
            continue
        neg_pool = neg_prelink[neg_prelink[:, 0] == i, 1]
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


if __name__ == '__main__':
    import numpy as np
    links = np.array([[1, 2], [1, 3], [2, 4]])
    trn = generate_trn(links, 5)
    tripple = generate_triple(trn)
    print(tripple)
    trn = generate_trn(links, 5, undirect=True)
    tripple = generate_triple(trn)
    print(tripple)
