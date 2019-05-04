import sys
from preprocessing import *

from evaluation import *
import model


import networkx as nx
import matplotlib.pyplot as plt


  
def split_train_test(triple,percentage = 0.2):
    train_index = np.random.choice(len(triple),round(len(triple)*percentage),replace = False)
    triple_train = triple[train_index]
    triple_test = np.delete(triple,train_index,0)
    return(triple_train,triple_test)



def visualization(F,percentile = 99.5):
    G = nx.DiGraph()
    threshold = np.percentile(F,percentile)
    edge = np.argwhere(F>threshold)+1
    edge = edge[edge[:,0] != edge[:,1]]
    node = []
    for i in range(len(edge)):
        for j in range(2):
            node.append(edge[i,j])
    node = set(node)
    for i in node:
        G.add_node(i)
    weight = np.round(F[edge[:,0]-1,edge[:,1]-1],decimals = 7)
    weight.shape = (weight.shape[0],1)
    weight_edge = np.append(edge[:,[1,0]],weight,axis = 1)
    weight_edge_list = [i.tolist() for i in weight_edge]
    weight_edge = [tuple(i) for i in weight_edge]
    G.add_weighted_edges_from(weight_edge)
    pos=nx.spring_layout(G)
    #nx.draw_networkx_labels(G,pos,font_size=15,font_family='sans-serif')
    #nx.draw(G)
    #plt.savefig('simple.png')
    return(weight_edge_list)
    
def get_concept_pairs(A,concept,n = 20):
    top_n_Aij = list(-np.sort(-A,axis=None)[0:n])
    top_n_A_index = list(map(lambda x: np.argwhere(A == x),top_n_Aij))
    result = [[concept[i[0][0]],concept[i[0][1]],
               round(A[i[0][0],i[0][1]],4)]
              for i in top_n_A_index]
    return result


if __name__ == '__main__':
    schools = ['ruc_key']
    for school in schools:
        X,links,concept = read_file(school)
        print('Step 1: reading %s\'s file is done.==================='%school)
        links = links[:,[1,0]]
        X = row_normlize(X)
        trn = generate_trn(links,X.shape[0])
        trn_train,trn_test = split_train_test(trn,percentage = 0.5)
        triple_train = generate_triple(trn_train)
        print('Step 2: trn is generated.====================')

        A,F = model.cgl_rank(X,triple_train,lamb=0.001,eta=0.5,tolerence=0.001,silence=False)
        print('Step 3: model training is done.====================')

        train_model_eval = evaluation(F,triple_train)
        print('训练集auc(liu)为{0}'.format(train_model_eval.auc_liu()))
        print('训练集auc为{0}'.format(train_model_eval.auc(trn)))
        #print(train_model_eval.mapl(trn))
        triple_test = generate_triple(trn_test)
        test_model_eval = evaluation(F,triple_test)
        print('测试集auc(liu)为{0}'.format(test_model_eval.auc_liu()))
        print('测试集auc为{0}'.format(test_model_eval.auc(trn)))
        #weight_edge_list = visualization(F,percentile = 90)
        print('====================')

        # 用全量数据训练模型
        triple = generate_triple(trn)
        A,F = model.cgl_rank(X,triple,lamb=0.001,eta=0.5,tolerence=0.004)
        weight_edge_list = visualization(F,percentile = 90)
        print(weight_edge_list)
        top_n_pairs = get_concept_pairs(A, concept, n=50)
        print(top_n_pairs)
        '''
        np.savetxt('result/A_cgl.txt',A)
        np.savetxt('result/F_cgl.txt',F)

        top_n_pairs = get_concept_pairs(A,concept,n=3000)
        
        with open('result/top_n_concept_pairs_cgl.csv', 'w', encoding='utf-8') as f:
            for pair in top_n_pairs:
                pair[2] = str(pair[2])
                f.write(','.join(pair) + '\n')    
        
        with open('data/concept_name.csv', 'w', encoding='utf-8') as f:
            for concept_idx in range(len(concept)):
                f.write(concept[concept_idx]+'\n')
        '''