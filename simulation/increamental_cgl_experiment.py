import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import normalize
from CGL_python.simulation.data_generation import generate_one_simulation_data
from CGL_python.model import cgl_rank
from CGL_python.preprocessing import row_normlize, generate_triple, generate_trn


def max_min(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))


def one_experiment(data_gene_para):
    word_num = data_gene_para['word_num']
    course_num = data_gene_para['course_num']
    course_link_num = data_gene_para['course_link_num']
    p = data_gene_para['p']
    data_gene_lab = data_gene_para['lab']
    seed = data_gene_para['seed']
    log_sigma = data_gene_para['log_sigma']
    incre_course = data_gene_para['incre_course']
    incre_word = data_gene_para['incre_word']
    data = generate_one_simulation_data(word_num=word_num, course_num=course_num, course_link_num=course_link_num,
                                        p=p, lab=data_gene_lab, seed=seed, log_sigma=log_sigma,
                                        incre_course=incre_course, incre_word=incre_word)

    X = data['X']
    links = data['F']
    X = row_normlize(X)
    trn = generate_trn(links, X.shape[0])
    tripple = generate_triple(trn)
    lamb = 10000
    A, F = cgl_rank(X, tripple, lamb=lamb, eta=10.0,
                    tolerence=0.00001, silence=False)

    from CGL_python.evaluation import generate_course_pairs
    weight_edge_list = generate_course_pairs(F, percentile=90)
    pre_edge_list = [[int(link[0]), int(link[1])] for link in weight_edge_list]
    cor_num = 0
    for i in data['F'].tolist():
        if any(list(map(lambda x: x == i, pre_edge_list))):
            cor_num += 1
    print('evaluation')
    print('course link: {}/{} are in top'.format(cor_num, len(data['F'])))

    A_min_max = max_min(A)
    A_min_max
    whole_f_norm = np.linalg.norm(data['A'] - A_min_max, ord='fro')

    # 增量学习
    incre_courses = range(course_num+1)[-incre_course:]
    from CGL_python.incremental_learning import split_tripple, incre_cgl_rank
    T, T1, T2, T3 = split_tripple(tripple, incre_courses)
    A, F = cgl_rank(X[:-incre_course, :-incre_word], T, lamb=1,
                    eta=0.01, tolerence=0.001, silence=False)
    row_0 = np.zeros((incre_word, A.shape[1]))
    A = np.row_stack((A, row_0))
    col_0 = np.zeros((A.shape[0], incre_word))
    A = np.column_stack((A, col_0))
    A1 = incre_cgl_rank(X=X, st_idx=(1, course_num-incre_word),
                        T1=T1, T2=T2, T3=T3, A=A, eta=0.01, lamb=1, silence=True)
    A1_min_max = max_min(A1)

    incre_f_norm = np.linalg.norm(data['A'] - A1_min_max, ord='fro')
    print('fro-loss:')
    print('whole: {}'.format(whole_f_norm))
    print('incre: {}'.format(incre_f_norm))
    result = {
        'whole_f_norm': whole_f_norm,
        'incre_f_norm': incre_f_norm,
        'whole_A': A,
        'incre_A': A1
    }
    return result


def one_group_experiment(data_gene_para, n_experiment, need_to_update_params):
    data_gene_para.update(need_to_update_params)
    st = datetime.now()
    incre_fnorms = []
    whole_fnorms = []
    for i in range(n_experiment):
        print('experiment {} begin!'.format(i))
        data_gene_para.update({'seed': i})
        result = one_experiment(data_gene_para)
        whole_fnorms.append(result['whole_f_norm'])
        incre_fnorms.append(result['incre_f_norm'])
    print('final result:')
    print('experiment num: {}'.format(n_experiment))
    print('avg whole F norm: {}'.format(sum(whole_fnorms)/len(whole_fnorms)))
    print('avg incre F norm: {}'.format(sum(incre_fnorms)/len(incre_fnorms)))
    print('top ten whole vs top ten incre\nwhole:{}\nincre:{}'.format(
        whole_fnorms[:10], incre_fnorms[:10]))
    with open('result/course_{}_word_{}.csv'.format(data_gene_para['course_num'], data_gene_para['word_num']), 'w+') as f:
        f.write('n,whole,trans\n')
        for n, (i, j) in enumerate(zip(whole_fnorms, incre_fnorms)):
            f.write('{},{:.4f},{:.4f}\n'.format(n, i, j))
        f.write('average,{:.4f},{:.4f}\n'.format(
            sum(whole_fnorms)/len(whole_fnorms), sum(incre_fnorms)/len(incre_fnorms)))
    ed = datetime.now()
    print('cost total: {} seconds'.format((ed-st).second))


if __name__ == '__main__':
    data_gene_para = {
        'word_num': 1000,
        'course_num': 30,
        'course_link_num': 100,
        'p': 0.01,
        'lab': 0.1,
        'seed': 0,
        'log_sigma': 2,
        'incre_course': 2,
        'incre_word': 5
    }
    n_experiment = 20
    course_nums = [30, 40, 50]
    word_nums = [500, 1000, 1500]
    need_to_update_params = []
    for i in course_nums:
        for j in word_nums:
            need_to_update_params.append({
                'word_num': j,
                'course_num': i,
                'course_link_num': i * 3,
            })
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(one_group_experiment,
                     [data_gene_para] * len(need_to_update_params),
                     [n_experiment] * len(need_to_update_params),
                     need_to_update_params,
                     )
