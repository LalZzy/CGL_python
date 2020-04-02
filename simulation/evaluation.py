import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score

def calc_map(y_true, y_pred):
    map_score = 0 
    for i in range(y_true.shape[1]):
        map_score += average_precision_score(y_true[:,i], y_pred[:,i])
    res = map_score/y_true.shape[1] 
    return res

def calc_ndcg(y_true, y_pred):
    ndcg_val = 0 
    for i in range(y_true.shape[1]):
        print(y_true[:,i], y_pred[:,i])
        ndcg_val += ndcg_score(y_true[:,i], y_pred[:,i],k=1)
    return ndcg_val/y_true.shape[1]

if __name__ == "__main__":
    a = np.array([[1,0,1],[0,1,1]])
    b = np.array([[0.7,0.4,0.2],[0.5,0.6,0.8]])
    print(calc_map(a,b))
    print(calc_ndcg(a,b))
    