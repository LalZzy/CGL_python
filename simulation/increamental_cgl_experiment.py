import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import normalize
from CGL_python.simulation.data_generation import generate_one_simulation_data
from CGL_python.model import cgl_rank
from CGL_python.preprocessing import row_normlize, generate_triple, generate_trn
from CGL_python.simulation.evaluation import calc_map
from CGL_python.evaluation import evaluation as Evaluation

def max_min(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))


def one_experiment(data_gene_para):
    result = {}
    word_num = data_gene_para['word_num']
    course_num = data_gene_para['course_num']
    course_link_num = data_gene_para['course_link_num']
    p = data_gene_para['p']
    data_gene_lab = data_gene_para['lab']
    seed = data_gene_para['seed']
    log_sigma = data_gene_para['log_sigma']
    incre_course = data_gene_para['incre_course']
    incre_word = data_gene_para['incre_word']
    update_A1 = data_gene_para['update_A1']
    data = generate_one_simulation_data(word_num=word_num, course_num=course_num, course_link_num=course_link_num,
                                        p=p, lab=data_gene_lab, seed=seed, log_sigma=log_sigma,
                                        incre_course=incre_course, incre_word=incre_word)
    
    eta = data_gene_para['eta']
    tolerence = data_gene_para['tolerence']
    algo_lamb = data_gene_para['algo_lamb']

    X = data['X']
    result['X'] = X
    links = data['F']
    X = row_normlize(X)
    trn = generate_trn(links, X.shape[0])
    tripple = generate_triple(trn)
    lamb = 10000

    ## 全量学习
    A, F, round0 = cgl_rank(X, tripple, lamb=algo_lamb, eta=eta,
                    tolerence=tolerence, silence=False)
    
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
    map_whole = calc_map(data['A'][:, -incre_word:], A_min_max[:, -incre_word:])
    evaluation_m = Evaluation(F,tripple)
    whole_auc = evaluation_m.auc_liu()
    result['whole_map'] = map_whole
    result['whole_round'] = round0
    result['whole_A'] = A
    result['whole_f_norm'] = whole_f_norm
    result['whole_auc'] = whole_auc

    # 增量学习
    incre_courses = range(course_num)[-incre_course:]
    from CGL_python.incremental_learning import split_tripple, incre_cgl_rank
    T, T1, T2, T3 = split_tripple(tripple, incre_courses)
    result['tripples'] = {'T': T, 'T1': T1, 'T2': T2, 'T3': T3}
    print('reach here.')
    A, F, _ = cgl_rank(X[:-incre_course, :-incre_word], T, lamb=algo_lamb,
                    eta=eta, tolerence=tolerence, silence=False)
    row_0 = np.zeros((incre_word, A.shape[1]))
    A = np.row_stack((A, row_0))
    col_0 = np.zeros((A.shape[0], incre_word))
    A = np.column_stack((A, col_0))
    A1, F1, round1 = incre_cgl_rank(X=X, st_idx=(course_num-incre_course, word_num-incre_word),
                        tripple=tripple, T=T, T1=T1, T2=T2, T3=T3, A0=A, eta=eta, lamb=algo_lamb, 
                        tolerrence=tolerence, silence=False, update_A1=update_A1)
    A1_min_max = max_min(A1)

    incre_f_norm = np.linalg.norm(data['A'] - A1_min_max, ord='fro')
    map_incre = calc_map(data['A'][:, -incre_word:], A1_min_max[:, -incre_word:])
    result['whole_map'] = map_whole
    evaluation_m = Evaluation(F1,tripple)
    incre_auc = evaluation_m.auc_liu()
    print('incremental fro-loss:')
    print('whole: {}'.format(whole_f_norm))
    print('incre: {}'.format(incre_f_norm))
    result.update({
        'incre_f_norm': incre_f_norm,
        'incre_A': A1,
        'incre_map': map_incre,
        'incre_round': round1,
        'incre_auc': incre_auc
    })
    result['A_diff'] = np.linalg.norm(result['whole_A'] - result['incre_A'], ord='fro')
    print('result A_diff: {}'.format(result['A_diff']))
    return result


def one_group_experiment(data_gene_para, n_experiment, need_to_update_params, file_prefix):
    data_gene_para.update(need_to_update_params)
    st = datetime.now()
    incre_fnorms = []
    whole_fnorms = []
    whole_rounds = []
    incre_rounds = []
    whole_maps = []
    incre_maps = []
    A_diffs = []
    whole_aucs, incre_aucs = [], []
    for i in range(n_experiment):
        print('experiment {} begin!'.format(i))
        data_gene_para.update({'seed': i})
        result = one_experiment(data_gene_para)
        whole_fnorms.append(result['whole_f_norm'])
        incre_fnorms.append(result['incre_f_norm'])
        whole_rounds.append(result['whole_round'])
        incre_rounds.append(result['incre_round'])
        whole_maps.append(result['whole_map'])
        incre_maps.append(result['incre_map'])
        A_diffs.append(result['A_diff'])
        whole_aucs.append(result['whole_auc'])
        incre_aucs.append(result['incre_auc'])
    print('final result:')
    print('experiment num: {}'.format(n_experiment))
    print('avg whole F norm: {}'.format(sum(whole_fnorms)/len(whole_fnorms)))
    print('avg incre F norm: {}'.format(sum(incre_fnorms)/len(incre_fnorms)))
    print('top ten whole vs top ten incre\nwhole:{}\nincre:{}'.format(
        whole_fnorms[:10], incre_fnorms[:10]))
    with open('result/course_{}_word_{}_incre_word_{}_incre_course_{}_{}.csv'.
              format(data_gene_para['course_num'], data_gene_para['word_num'],
                     data_gene_para['incre_word'], data_gene_para['incre_course'],
                     str(data_gene_para['update_A1'])), 'w+') as f:
        f.write('n,whole,trans\n')
        for n, (i, j, k, i1, j1, i2, j2) in enumerate(zip(whole_fnorms, incre_fnorms, A_diffs, whole_rounds, incre_rounds, whole_maps, incre_maps)):
            f.write('{},{:.4f},{:.4f},{:.5f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(n, i, j, k, i1, j1, i2, j2))
        f.write('average,{:.4f},{:.4f},{:.5f}\n'.format(
            sum(whole_fnorms)/len(whole_fnorms), sum(incre_fnorms)/len(incre_fnorms),
            sum(A_diffs)/len(A_diffs)))
    
    whole_maps = [i for i in whole_maps if i >= 0]
    incre_maps = [i for i in incre_maps if i >= 0]
    with open('result/summary_{}.csv'.format(file_prefix), 'a+') as f:
        f.write('{},{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(
            data_gene_para['word_num'], data_gene_para['course_num'],
            data_gene_para['incre_word'], data_gene_para['incre_course'],
            str(data_gene_para['update_A1']),
            sum(whole_fnorms)/len(whole_fnorms), sum(incre_fnorms)/len(incre_fnorms),
            sum(whole_aucs)/len(whole_aucs), sum(incre_aucs)/len(incre_aucs),
            sum(whole_maps)/len(whole_maps), sum(incre_maps)/len(incre_maps),
            sum(whole_rounds)/len(whole_rounds), sum(incre_rounds)/len(incre_rounds)
        ))
    ed = datetime.now()
    print('cost total: {} seconds'.format((ed-st).total_seconds()))

def test_one():
    data_gene_para = {
        'word_num': 500,
        'course_num': 30,
        'course_link_num': 100,
        'p': 0.01,
        'lab': 0.1,
        'seed': 0,
        'log_sigma': 2,
        'incre_course': 1,
        'incre_word': 5,
        'tolerence': 0.0001,
        'algo_lamb': 1,
        'eta': 0.01,
        'update_A1': True
    }
    n_experiment = 1
    file_prefix = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    with open('result/summary_{}.csv'.format(file_prefix), 'a+') as f:
        f.write('word_num,course_num,incre_concept,incre_course,update_A1,whole_fnorms,incre_fnorms,whole_auc,incre_auc,whole_map,incre_map,whole_round,incre_round\n')
    one_group_experiment(data_gene_para, n_experiment, need_to_update_params={}, 
                         file_prefix=datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S'))

def run():
    data_gene_para = {
        'word_num': 1000,
        'course_num': 30,
        'course_link_num': 100,
        'p': 0.01,
        'lab': 0.1,
        'seed': 0,
        'log_sigma': 2,
        'incre_course': 3,
        'incre_word': 20,
        'tolerence': 0.00001,
        'algo_lamb': 1,
        'eta': 0.01,
        'update_A1': False
    }
    n_experiment = 40
    # course_nums = [30]
    # word_nums = [500]
    update_A1 = [True, False]
    course_nums = [30, 40, 50]
    word_nums = [500, 1000, 1500]
    need_to_update_params = []
    file_prefix = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    with open('result/summary_{}.csv'.format(file_prefix), 'a+') as f:
        f.write('word_num,course_num,incre_concept,incre_course,update_A1,whole_fnorms,incre_fnorms,whole_auc,incre_auc,whole_map,incre_map,whole_round,incre_round\n')
        
    for i in course_nums:
        for j in word_nums:
            for k in update_A1:
                need_to_update_params.append({
                    'word_num': j,
                    'course_num': i,
                    'course_link_num': i * 3,
                    'update_A1': k
                })
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(one_group_experiment,
                     [data_gene_para] * len(need_to_update_params),
                     [n_experiment] * len(need_to_update_params),
                     need_to_update_params,
                     [file_prefix] * len(need_to_update_params)
                     )


if __name__ == '__main__':
    # test_one()
    run()

