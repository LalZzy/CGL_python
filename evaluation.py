import numpy as np
from sklearn import metrics

class evaluation():
    def __init__(self,F,triple):
        self.F = F
        self.triple = triple
       
    def auc_liu(self):
        I = self.triple[:,0]
        J = self.triple[:,1]
        K = self.triple[:,2]
        auc = sum((self.F[I,J] - self.F[I, K]) > 0)/len(self.triple)
        return(auc)
    
    def auc(self,trn):
        y_true = trn[:,2]
        y_pred = np.empty_like(y_true, dtype=np.float)
        n = 0
        for i in range(len(self.F)):
            for j in range(len(self.F)):
                if i == j: continue
                y_pred[n] = self.F[i, j]
                n += 1
        auc = metrics.roc_auc_score(y_true,y_pred)
        return(auc)
    def mapl(self,trn):
        y_true = trn[:,2]
        y_pred = np.empty_like(y_true,dtype =  np.float)
        n = 0
        for i in range(len(self.F)):
            for j in range(len(self.F)):
                if i == j:continue
                y_pred[n] = self.F[i,j]
                n+=1
        mapl = metrics.average_precision_score(y_true, y_pred)
        return(mapl)


def parameter_choose(eval_results):
    '''
    :param eval_result: 存储着不同参数训练得到的结果。[{'parameters':0.01,'evaluation':0.85}]
    :return: 最大的那组参数。
    '''
    auc = np.array([eval_result['evaluation'] for eval_result in eval_results])
    return eval_results[auc.argmax()]


def get_concept_pairs(A,concept,n = 20):
    top_n_Aij = list(-np.sort(-A,axis=None)[0:n])
    top_n_A_index = list(map(lambda x: np.argwhere(A == x),top_n_Aij))
    result = [[concept[i[0][0]],concept[i[0][1]],
               round(A[i[0][0],i[0][1]],4)]
              for i in top_n_A_index]
    return result

def generate_course_pairs(F,percentile = 99.5):
    threshold = np.percentile(F,percentile)
    edge = np.argwhere(F>threshold)+1
    edge = edge[edge[:,0] != edge[:,1]]
    node = []
    for i in range(len(edge)):
        for j in range(2):
            node.append(edge[i,j])
    weight = np.round(F[edge[:,0]-1,edge[:,1]-1],decimals = 7)
    weight.shape = (weight.shape[0],1)
    weight_edge = np.append(edge[:,[1,0]],weight,axis = 1)
    weight_edge_list = [i.tolist() for i in weight_edge]
    return(weight_edge_list)