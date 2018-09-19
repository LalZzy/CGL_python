from scipy import sparse
import pandas as pd
import pathlib

ruc = pd.read_csv('data/ruc.csv',index_col = 0)
sparse.csr_matrix(ruc)

path = pathlib.Path('data/ruc.lsvm')
print(path.exists())