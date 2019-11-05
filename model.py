import numpy as np
from scipy.sparse.linalg import spsolve 
from scipy import sparse

#cgl_rank算法
def cgl_rank(X,triple,lamb,eta,silence=True,tolerence = 1e-6,round = 10):
    #初始化参数B,Q,P
    K = X * X.transpose()
    n = X.shape[0]
    B = np.zeros((n,n))
    Q = np.zeros((n,n))
    r = 1
    F = K * B * K
    loss_func = lambda x: max((1 - F[x[0]-1,x[1]-1] + F[x[0]-1,x[2]-1]),0)**2
    # 按照定义计算loss，经常出现第一步损失变化为负的情况，所以先定义obj_old为Inf
    obj_old = np.inf
    while True:
        Delta = np.zeros((n,n))
        F = K * B * K
        for t in triple:
            # 调整索引从0开始
            i, j, k = t[0] - 1, t[1] - 1, t[2] - 1
            sigma = max(0, 1 - F[i, j] + F[i, k])
            Delta[i, j] += sigma
            Delta[i, k] += -sigma
        while True:
            B_old = B
            F_old = F
            P = B - eta*(lamb*F - 2*K*Delta*K)
            #终止条件：当前一步的目标函数值的变化
            B = P + (r-1)/(r+2) * (P - Q)
            F = K * B * K
            Q = P
            regularization_part = lamb/2 * (K * B * K*B).sum()
            loss_part = sum(list(map(loss_func,triple)))
            obj = loss_part + regularization_part  
            #print('old loss function: {}, new loss function: {}'.format(obj_old,obj))
            if obj_old < obj:
                eta = eta * 0.95
                # 此时并非发生更新，还原B,F矩阵
                B = B_old
                F = F_old
            else:
                break
        loss_change = obj_old - obj
        if (loss_change < tolerence) and (loss_change >= 0):
            print('stop at round {}'.format(r))
            break
        if (r%5 == 0) and (silence == False):
            print("Iteration: %d, eta: %f B's 'f-norm' decreases: %f, old_loss: %f, current_loss: %f" %(r,eta,loss_change,obj_old,obj))
        obj_old = obj
        r = r + 1
    A = X.transpose()*B*X
    return(A,F)

#transductive-cgl-rank算法    
def cgl_rank_trans(X,triple,lamb,eta,silence=False,tolerence = 0.01):
    #初始化参数B,Q,P
    K = X * X.transpose()
    n = X.shape[0]
    inv_K = spsolve(K,np.eye(n))
    F = np.random.normal((n,n))
    Q = np.zeros((n,n))
    r = 1
    last_loss = np.inf
    while True:
        Delta = np.zeros((n,n))
        
        for t in triple:
            #调整索引从0开始
            i,j,k = t[0]-1,t[1]-1,t[2]-1
            sigma = max(0,1-F[i,j]+F[i,k])
            Delta[i,j] += sigma
            Delta[i,k] += -sigma
        
        P = F - eta*(lamb*F - 2*K*Delta*K - 2*Delta)       
        #当前一步的F矩阵F范数的变化，以此作为终止准则
        loss_change = np.linalg.norm(F - (P + (r-1)/(r+2)*(P - Q)),ord = 'fro')
        if loss_change < tolerence:
            break
        
        if loss_change > last_loss:
            eta = eta *0.95
            continue
        last_loss = loss_change
        F = P + (r-1)/(r+2) * (P - Q)
        Q = P
        r = r + 1
        
        if (r%10 == 0) and (silence == False):
            print("Step: %d, Lambda: %f B's 'f-norm' decreases: %f" %(r,eta,loss_change))
        
        #当前一步的目标函数值
        #regularization_part = lamb/2 * 
        #obj = loss_part + regularization_part
    
    A = X.transpose()*inv_K*F*inv_K*X
    return(A,F)

#sparse_cgl_rank算法
def sparse_cgl_rank(X,triple,lamb,eta,silence=False,tolerence = 0.00001):
    #初始化A,P,Q
    n,p = X.shape
    A = sparse.csr_matrix((p,p))
    P = sparse.csr_matrix((p,p))
    Q = sparse.csr_matrix((p,p))
    r = 1
    F = X * A * X.transpose()
    loss_func = lambda x: max((1 - F[x[0]-1,x[1]-1] + F[x[0]-1,x[2]-1]),0)**2
    regularization_part = lamb * np.sum(abs(A))
    loss_part = sum(list(map(loss_func,triple)))
    obj_old = np.inf
    while True:
        Delta = np.zeros((n,n))
        F = X * A * X.transpose()
        for t in triple:
            i,j,k = t[0]-1,t[1]-1,t[2]-1
            sigma = max(0,1-F[i,j]+F[i,k])
            Delta[i,j] += sigma
            Delta[i,k] += -sigma
        for j in range(n):
            tmp = A[:,j] + 2*eta*X.transpose()*Delta*X[:,j]
            sgn = lambda x: 1 if x>0 else -1 if x<0 else 0
            soft_threshold = lambda x,y: sgn(x)*max((abs(x) - y),0)
            tmp = np.array(list(map(soft_threshold,tmp,np.zeros_like(tmp)+lamb*eta)))
            tmp.shape = (len(tmp),1)
            P[:,j] = sparse.csr_matrix(tmp)
        # TO DO: adaptive step size

        A = P + (r-1)/(r+2)*(P-Q)
        F = X * A * X.transpose()
        Q = P
        regularization_part = lamb * np.sum(abs(A))
        loss_part = sum(list(map(loss_func, triple)))
        obj = regularization_part + loss_part
        loss_change =  obj_old - obj
        if loss_change < tolerence:
            print('Iteration stopped after {0} steps.'.format(r))
            break
        obj_old = obj
        if (r%10 == 0) and (silence == False):
            print("Step: %d, Lambda: %f B's 'f-norm' decreases: %f" %(r,eta,loss_change))
        r += 1
    return(A.toarray(),F.toarray())