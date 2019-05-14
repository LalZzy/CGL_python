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
        auc = sum((self.F[I-1,J-1] - self.F[I-1, K-1]) > 0)/len(self.triple)
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
        mapl = metrics.average_precision_score(y_true,y_pred)
        return(mapl)


def parameter_choose(eval_results):
    '''
    :param eval_result: 存储着不同参数训练得到的结果。[{'parameters':0.01,'evaluation':0.85}]
    :return: 最大的那组参数。
    '''
    auc = np.array([eval_result['evaluation'] for eval_result in eval_results])
    return eval_results[auc.argmax()]



'''
y_true = trn[:,2]
y_pre = np.empty_like(y_true,dtype = np.float16)
y_pre
k = 0
for i in range(len(F)):
    for j in range(len(F)):
        if i == j:continue
        y_pre[k] = F[i,j]
        k+=1
        
from sklearn import metrics
eval_auc(F,triple)
'''