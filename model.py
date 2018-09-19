import numpy as np
from scipy.sparse.linalg import spsolve 
from scipy import sparse

#cgl_rank算法
def cgl_rank(X,triple,lamb,eta,silence=True,tolerence = 1e-6):
    #初始化参数B,Q,P
    K = X * X.transpose()
    n = X.shape[0]
    B = np.zeros((n,n))
    Q = np.zeros((n,n))
    r = 1
    F = K * B * K
    loss_func = lambda x: max((1 - F[x[0]-1,x[1]-1] + F[x[0]-1,x[2]-1]),0)**2
    regularization_part = lamb/2 * (K * B * K*B).sum()
        
    loss_part = sum(list(map(loss_func,triple)))
    obj_old = np.inf
        
    last_loss = np.inf
    while True:
        Delta = np.zeros((n,n))
        F = K * B * K
        for t in triple:
            #调整索引从0开始
            i,j,k = t[0]-1,t[1]-1,t[2]-1
            sigma = max(0,1-F[i,j]+F[i,k])
            Delta[i,j] += sigma
            Delta[i,k] += -sigma
        P = B - eta*(lamb*F - 2*K*Delta*K)       

        #第一种终止条件：当前一步的B矩阵F范数的变化，以此作为终止准则
        #c_loss = np.linalg.norm(B - (P + (r-1)/(r+2)*(P - Q)),ord = 'fro')
        #第二种终止条件：当前一步的目标函数值的变化
        regularization_part = lamb/2 * (K * B * K*B).sum()
        
        loss_part = sum(list(map(loss_func,triple)))
        obj = loss_part + regularization_part
        c_loss = obj_old - obj
        
        if c_loss < tolerence:
            break
        
        #if c_loss > last_loss:
        #    eta = eta * 0.95
        #    continue
        obj_old = obj
        last_loss = c_loss
        B = P + (r-1)/(r+2) * (P - Q)
        Q = P
        if (r%5 == 0) and (silence == False):
            print("Iteration: %d, Lambda: %f B's 'f-norm' decreases: %f" %(r,eta,c_loss))
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
        c_loss = np.linalg.norm(F - (P + (r-1)/(r+2)*(P - Q)),ord = 'fro')
        if c_loss < tolerence:
            break
        
        if c_loss > last_loss:
            eta = eta *0.95
            continue
        last_loss = c_loss
        F = P + (r-1)/(r+2) * (P - Q)
        Q = P
        r = r + 1
        
        if (r%10 == 0) and (silence == False):
            print("Step: %d, Lambda: %f B's 'f-norm' decreases: %f" %(r,eta,c_loss))
        
        #当前一步的目标函数值
        #regularization_part = lamb/2 * 
        #obj = loss_part + regularization_part
    
    A = X.transpose()*inv_K*F*inv_K*X
    return(A,F)

#sparse_cgl_rank算法
def sparse_cgl_rank(X,triple,lamb,eta,silence=False,tolerence = 0.01):
    #初始化A,P,Q
    n,p = X.shape
    A = sparse.csr_matrix((p,p))
    P = sparse.csr_matrix((p,p))
    Q = sparse.csr_matrix((p,p))
    r = 1
    last_loss = np.inf
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
        
        c_loss = sparse.linalg.norm(A - (P + (k-1)/(k+2)*(P - Q)),ord = 'fro')
        
        if c_loss < tolerence:
            break
        
        if c_loss > last_loss:
            eta = eta *0.95
            continue
        last_loss = c_loss
        A = P + (r-1)/(r+2)*(P-Q)
        Q = P

        
        if (r%10 == 0) and (silence == False):
            print("Step: %d, Lambda: %f B's 'f-norm' decreases: %f" %(r,eta,c_loss))
            r += 1
    return(A,F) 

     
            
     
    
    