# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from copy import deepcopy


def split_tripple(tripple, incre_idx):
    '''
    # 把新增了课程后新增加的课程三元对挑出来
    :param tripple: 二维numpy.ndarray，表示三元组序列。
    :param incre_idx: list，增量部分标号
    :return: T1,T2,T3: (i,j,k)次序表示第一、二、三个位置的元素在增量标号列表，其余两个元素不在。
    '''
    isin_mark = np.isin(tripple, incre_idx)
    # 用二进制表示第i列有没有在incre_idx中。
    bin_mark = isin_mark[:, 0] * 2 ** 0 + \
        isin_mark[:, 1] * 2 ** 1 + isin_mark[:, 2] * 2 ** 2
    T1 = tripple[bin_mark == 1]
    T2 = tripple[bin_mark == 2]
    T3 = tripple[bin_mark == 4]
    T = tripple[bin_mark == 0]
    return T, T1, T2, T3



def incre_cgl_rank(X, st_idx, T1, T2, T3, A0, lamb, eta, tolerrence=0.00001, silence=False):
    '''
    增量学习版本，可以支持一门课多个词的输入。
    '''
    # TODO: 这里需要改
    # X1~4分别表示分块矩阵的四块，按行编号。
    A = A0.copy()
    row_st_idx, col_st_idx = st_idx
    X1 = X[:row_st_idx, :col_st_idx]
    X2 = X[:row_st_idx, col_st_idx:]
    X3 = X[row_st_idx:, :col_st_idx]
    X4 = X[row_st_idx:, col_st_idx:]
    F = np.matmul(np.matmul(X, A), X.transpose())

    # loss function: max( (1-F_ij+F_ik),0 ) square.
    def loss_func(x): return max(
        (1 - F[x[0], x[1]] + F[x[0], x[2]]), 0)**2
    
    round_A2 = 0
    eta1 = eta
    old_loss = np.inf
    while True:
        F = np.matmul(np.matmul(X, A), X.transpose())
        p = A.shape[0] - col_st_idx
        # l1,l2的形状跟待更新的A2,A3一样
        l1, l2 = np.zeros((col_st_idx, p)), np.zeros((col_st_idx, p))
        for t in T2:
            i, j, k = t
            sigma = max(0, 1 - F[i, j] + F[i, k])
            l1 += sigma * np.matmul(X1.transpose()[:, i:i+1], X4)
        for t in T3:
            i, j, k = t
            sigma = max(0, 1 - F[i, j] + F[i, k])
            l2 += sigma * np.matmul(X1.transpose()[:, i:i+1], X4)
        
        # A1 after this round update.
        while True:
            F_old = F.copy()
            A_old = A.copy()
            A[:col_st_idx, col_st_idx:] -= eta1 * (lamb * A[:col_st_idx, col_st_idx:] - (l1-l2))
            F = np.matmul(np.matmul(X, A), X.transpose())
            
            total1 = 0
            correct1 = 0
            for i, j, k in T2:
                if F[i,j] - F[i,k] > 0:
                    correct1 += 1
                total1 += 1
            
            total2 = 0
            correct2 = 0
            for i,j,k in T3:
                if F[i,j] - F[i,k] > 0:
                    correct2 += 1
                total2 += 1
                    
            loss_part = sum(list(map(loss_func, T2))) + sum(list(map(loss_func, T3)))
            reg_part = lamb/2 * np.sqrt((A[:col_st_idx, col_st_idx:]**2).sum())
            cur_loss = loss_part + reg_part
            unit_loss_change = (old_loss - cur_loss) / A[:col_st_idx, col_st_idx:].size
            loss_change = (old_loss - cur_loss)
        
            if unit_loss_change < 0:
                eta1 *= 0.95
                F = F_old.copy()
                A = A_old.copy()
            else:
                break
            
        if not silence:
            if round_A2 % 10 == 0:
                print('num update A2: {}, eta: {}, loss decrease: {}'.format(round_A2, eta1, loss_change))
                print('loss_part={}, reg_part={}'.format(loss_part, reg_part))
                print('correct ratio T2:{}, T3:{}'.format(correct1/total1, correct2/total2))
                print('loss part T2:{}, T3:{}'.format(sum(list(map(loss_func, T2))),sum(list(map(loss_func, T3)))))
        round_A2 += 1
        if loss_change < tolerrence:
            break
        old_loss = cur_loss
    print('train A2 cost: {} step, final loss: {}, loss_change: {}'.format(round_A2, cur_loss, loss_change))

    round_A3 = 0
    old_loss = np.inf
    eta1 = eta
    while True:
        F = np.matmul(np.matmul(X, A), X.transpose())
        l3 = np.zeros((p, col_st_idx))
        for t in T1:
            i, j, k = t
            sigma = max(0, 1 - F[i, j] + F[i, k])
            l3 += sigma * \
                (np.matmul(X1.transpose()[:, j:j + 1] - X1.transpose()[:, k:k + 1], X4)).transpose()
        
        while True:
            A_old, F_old = A.copy(), F.copy()
            A[col_st_idx:, :col_st_idx] -= eta * (lamb * A[col_st_idx:, :col_st_idx] - l3)
            F = np.matmul(np.matmul(X, A), X.transpose())
            loss_part = sum(list(map(loss_func, T1)))
            reg_part = lamb/2 * np.sqrt((A[col_st_idx:, :col_st_idx]**2).sum())
            cur_loss = loss_part + reg_part
            unit_loss_change = (old_loss - cur_loss)/A[col_st_idx:, :col_st_idx].size
            if unit_loss_change < 0:
                eta1 *= 0.95
                A, F = A_old.copy(), F_old.copy()
            else:
                break
                    
        if not silence:
            print('num update A3: {}, eta: {}, loss decrease: {}'.format(round_A3, eta1, unit_loss_change))
            print('loss_part={}, reg_part={}'.format(loss_part, reg_part))
        round_A3 += 1
        if unit_loss_change < tolerrence:
            break
        old_loss = cur_loss
    return A


def gene_incre_matrix(M, n):
    '''
    increase M from m*m to (m+n)*(m+n)
    '''
    row_0 = np.zeros((n, M.shape[1]))
    #import pdb;pdb.set_trace()
    M = np.row_stack((M, row_0))
    col_0 = np.zeros((M.shape[0], n))
    M = np.column_stack((M, col_0))
    return M
    

def test_incre(data_file, link_file, concept_file, file_type, incre_course_num, incre_concept_num):
    from preprocessing import generate_triple, generate_trn, row_normlize, read_file
    X, links, concept = read_file(data_file, link_file, concept_file=concept_file, file_type=file_type)
    X = X.todense()
    n_course, n_concept = X.shape[0], X.shape[1]
    trn = generate_trn(links, n_course)
    tripple = generate_triple(trn)
    T, T1, T2, T3 = split_tripple(tripple, [n_course-1])
    # 找到一个只有最后一门课才有的词
    X = row_normlize(X)
    # 先训不带增量最后一行的
    # 用全量数据训练模型
    import model
    A, F = model.cgl_rank(X, tripple, lamb=0.01, eta=1,
                          tolerence=0.001, silence=False)
    print('finish training whole A\n\n')
    np.savetxt('result/ruc_A_whole.txt', A)
    # 用增量数据训练模型
    A, F = model.cgl_rank(X[:-incre_course_num, :-incre_concept_num], T, lamb=0.01,
                          eta=1, tolerence=0.001, silence=False)
    
    A = gene_incre_matrix(A, incre_concept_num)
    print('\n\n\n')
    A1 = incre_cgl_rank(X, (n_course-incre_course_num, n_concept-incre_concept_num), T1, T2, T3, A, eta=5, lamb=0.01,
                        tolerrence=0.001)
    np.savetxt('result/ruc_A_incre.txt', A1)


def test_split_tripple():
    test_links = np.array([[1, 2], [1, 3], [2, 4]])
    from preprocessing import generate_triple, generate_trn
    trn = generate_trn(test_links, 4)
    tripple = generate_triple(trn)
    test_links = np.array([[1, 2], [1, 3], [2, 4], [1, 5], [5, 2], [6, 1]])
    tripple_more = generate_triple(generate_trn(test_links, 7))
    #import pdb;pdb.set_trace()
    print(tripple)
    print('after add 5,tripple become: {}\nlength:{}'.format(
        tripple_more, len(tripple_more)))
    T, T1, T2, T3 = split_tripple(tripple_more, [5, 6])
    print('origin T:\n{}\nincre as pre:\n{}\nincre as post:\n{}\nincre post nor pre:\n{}\n'.format(
        T, T1, T2, T3))


if __name__ == '__main__':
    # test_split_tripple()
    data_file = 'data/all_ruc_word_info_new.csv'
    link_file = 'data/all_ruc_new_link.csv'
    concept_file = 'data/ruc_all_concepts_new.csv'
    file_type = 'sparse_row_col_val'
    incre_course_num = 1
    incre_concept_num = 4
    test_incre(data_file, link_file, concept_file, file_type, incre_course_num, incre_concept_num)
