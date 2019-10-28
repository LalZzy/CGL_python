 # -*- coding: utf-8 -*-

import pyperclip
import sys,os
from preprocessing import *
from concept_pairs import *
from evaluation import *
import model
import json
  
def topn_pairs(N):
    school = 'all'
    input_path = './data'
    data_file = '{}/{}.csv'.format(input_path, school)
    link_file = '{}/{}.link'.format(input_path, school)
    X, links, concept = read_file(data_file, link_file, dense_input=True)
    A_file = 'result/all_A_cgl.txt'
    A = pd.read_csv(A_file,sep = " ",header = None)
    top_n_pairs = get_concept_pairs(A.values, concept, n = N)
    pyperclip.copy(str(top_n_pairs))
    print(top_n_pairs)

if __name__ == '__main__':
    n = int(sys.argv[1])
    topn_pairs(n)
