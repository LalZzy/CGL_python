#%%
import pyperclip
import sys,os
from preprocessing import *

from evaluation import *
import model
  
def split_train_test(triple,percentage = 0.2):
    train_index = np.random.choice(len(triple),round(len(triple)*percentage),replace = False)
    triple_train = triple[train_index]
    triple_test = np.delete(triple,train_index,0)
    return(triple_train,triple_test)

def main(school,save=True,cross_valid = False, lambs = None):
    input_path = './data'
    data_file = '{}/{}.csv'.format(input_path, school)
    link_file = '{}/{}.link'.format(input_path, school)
    X, links, concept = read_file(data_file, link_file, dense_input=True)
    print('Step 1: reading %s\'s file is done.===================' % school)
    links = links[:, [1, 0]]
    X = row_normlize(X)
    trn = generate_trn(links, X.shape[0])
    if cross_valid:
        trn_train,trn_test = split_train_test(trn,percentage = 0.5)
        triple_train = generate_triple(trn_train,sample_size = 0.8)
        print('Step 2: trn is generated.====================')

        if not lambs:
            raise ValueError('parameter lambs is a list, length >0.')
        eval_res = []
        for lamb in lambs:
            global A
            A,F = model.cgl_rank(X,triple_train,lamb=lamb,eta=0.01,tolerence=0.00001,silence=True)
            print('Step 3: model training is done.====================')

            train_model_eval = evaluation(F,triple_train)
            print('lambda = {}'.format(lamb))
            print('训练集auc(liu)为{0}'.format(train_model_eval.auc_liu()))
            print('训练集auc为{0}'.format(train_model_eval.auc(trn)))
            #print(train_model_eval.mapl(trn))
            triple_test = generate_triple(trn_test,sample_size = 0.8)
            test_model_eval = evaluation(F,triple_test)
            print('测试集auc(liu)为{0}'.format(test_model_eval.auc_liu()))
            print('测试集auc为{0}'.format(test_model_eval.auc(trn)))
            #weight_edge_list = visualization(F,percentile = 90)
            print('====================')
            eval_res.append({'parameter':lamb,'evaluation':test_model_eval.auc_liu()})
        bst_para = parameter_choose(eval_res)
        print('best parameter: lambda {} ,test auc: {}'.format(bst_para['parameter'],bst_para['evaluation']))
    # 用全量数据训练模型
    triple = generate_triple(trn, sample_size=0.8)
    A, F = model.cgl_rank(X, triple, lamb=0.01, eta=10, tolerence=0.00001, silence=False)
    weight_edge_list = generate_course_pairs(F, percentile=90)
    print(weight_edge_list)

    top_n_pairs = get_concept_pairs(A, concept, n=50)
    pyperclip.copy(str(top_n_pairs))
    print(top_n_pairs)

    if save:
        if not os.path.exists('./result'):
            os.system('mkdir -p ./result')
        np.savetxt('result/{}_A_cgl.txt'.format(school),A)
        np.savetxt('result/{}_F_cgl.txt'.format(school),F)

        with open('result/{}_top_n_concept_pairs_cgl.csv'.format(school), 'w', encoding='utf-8') as f:
            for pair in top_n_pairs:
                pair[2] = str(pair[2])
                f.write(','.join(pair) + '\n')

        with open('data/concept_name.csv', 'w', encoding='utf-8') as f:
            for concept_idx in range(len(concept)):
                f.write(concept[concept_idx]+'\n')

if __name__ == '__main__':
    schools = ['ruc']
    for school in schools:
        main(school,cross_valid=True,lambs=[1,0.1,0.01])