#coding:utf-8

"""
@author: zhouzhenyu
@contact: zhouzy1293@foxmail.com
@time: 2018/9/18 10:45
"""
import pandas as pd
ruc = pd.read_csv('ruc1.csv',encoding='utf-8')
key_word_dict = {}
with open('cos-refD/concepts_names.txt',encoding='utf-8') as f:
    for line in f:
        word = line.strip().lower().split(' ')
        word = '_'.join(word)
        key_word_dict.setdefault(word,1)

key_word_index = [list(ruc.columns).index(word) for word in ruc.columns if word in key_word_dict]
key_ruc = ruc.iloc[:,[0]+key_word_index]
key_ruc.to_csv('ruc_key.csv',index=False)
print(key_word_index)

