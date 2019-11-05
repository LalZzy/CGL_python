# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 20:45:13 2019

@author: zhong
"""

import pandas as pd
import csv

cha = pd.read_csv("chapter_prerequisite.csv")
del cha[cha.columns[0]]

cha.index = range(1,151)
cha.columns = range(1,151)

res = []
for i in cha.index:
    col = [i for i,x in enumerate(cha.loc[i]) if x == 1]
    for j in col:
        res.append('%s %s'%(i,j+1))
        
print(res)

output = open('all.link', 'w', encoding = 'utf-8')
for item in res:
    output.writelines(item)
    output.writelines("\n")
