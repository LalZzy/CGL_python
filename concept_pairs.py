# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:39:52 2019

@author: zhong
"""

#%%
import pandas as pd
import pyperclip

def concept_pair(word, top_n_pairs, direction, n = None):
# function:找出某个词的前n个概念对
#
# input：
#   word，需要查询的词
#   top_n_pairs，全部前3000个概念对
#   direction，方向，both表示该词为先修词或后修词都可以，out表示该词为先修词，in表示该词为后修词
#   n，获取需要查询的词的概念对的数量，默认值为None，表示查找前3000个概念对中包含该词的所有概念对
#
# output：需要输入html的结果已经复制在剪贴板中，输出结果为相关词
    result_as_source = []
    result_as_target = []
    pair_word_as_source = []
    pair_word_as_target = []
        
    for i in top_n_pairs.index:
        if word == top_n_pairs.loc[i, 'source']:
            result_as_source.append(list(top_n_pairs.loc[i]))
            pair_word_as_source.append(top_n_pairs.loc[i, 'target'])
        elif word == top_n_pairs.loc[i, 'target']:
            result_as_target.append(list(top_n_pairs.loc[i]))
            pair_word_as_target.append(top_n_pairs.loc[i, 'source'])
        if n != None and len(result_as_source)+len(result_as_target) >= n:
            break
        
    if direction == 'both':
        pyperclip.copy(str(result_as_source + result_as_target))
        return pair_word_as_source + pair_word_as_target
    elif direction == 'out':
        pyperclip.copy(str(result_as_source))
        return pair_word_as_source
    elif direction == 'in':
        pyperclip.copy(str(result_as_target))
        return pair_word_as_target

#%%
## 读入前3000个概念对
'''
top_n_pairs = pd.read_csv("result/top_n_concept_pairs_cgl.csv", encoding = 'utf-8', header = None, names = ['source', 'target','value'])

m = concept_pair('optimization', top_n_pairs, 'both', n = 30)
print(m)
'''
