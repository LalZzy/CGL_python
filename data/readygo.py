# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:01:00 2019

@author: zhong
"""
import csv
import pandas as pd
import os
import json
import re

class readygo(object):
    def __init__(self):
        self.path = "result_all_word/"
        self.outpath = "finals/"
        self.book_chapter_page = {
                "CategoricalDataAnalysis":
                    [1,36,70,115,165,211,267,314,357,409,455,491,538,576,600,619,632],
                "cateregression":
                    [1,29,51,81,123,143,181,207,241,269,317,331,363,395,429,485],
                "ComputationalStatistics":
                    [1,21,59,97,129,151,201,237,287,325,363,393,421],
                "ConvexOptimizationbook":
                    [1,21,67,127,215,291,351,397,457,521,561,631],
                "DeepLearning":
                    [1,31,53,80,98,167,228,274,331,374,424,446,492,505,529,561,593,608,634,656,723],
                "RegressionModelingStrategies":
                    [1,13,45,63,103,127,143,161,181,219,275,291,311,327,359,389,399,423,453,475,521,535],
                "ReinforcementLearning":
                    [1,11,43,117,167,205,235],
                "RossProbability":
                    [1,22,58,117,186,232,297,388,417,438,457],
                "StatisticalModels":
                    [1,15,52,94,161,225,300,353,417,468,565,645,696],
                "StochasticProcesses":
                    [1,6,13,25,32,36,43,49,54,64,71,77,89,94,100,111,130,147,152,160,167,177,184,192,204,209,214,218,229,237,244,247,251,259,269,279,286,302,312,319,326,339,348]
                }
    
    def file_reader(self,file_name):
        data = []
        with open(self.path+file_name, "r", encoding = "utf-8") as input:
            read = csv.reader(input)
            for i in read:
                i[1] = int(i[1])
                i[2] = int(i[2])
                data.append(i)
        return data
    
    def match_page_and_chapter(self, data, chapter_page):
        result = []
        for word_page_num in data:
            for i in range(len(chapter_page)-1):
                if word_page_num[1] in range(chapter_page[i], chapter_page[i+1]):
                    word_page_num[1] = i+1
                    result.append(word_page_num)
                    break
        return result

    
    def deal_one(self, data_chapter, word, chapter_page):
        df = pd.DataFrame(data_chapter, columns=['word','chapter','num'])
        sum_result = df['num'].groupby([df['chapter'], df['word']]).sum()
        final = pd.DataFrame(columns = word)
        for i in range(len(chapter_page)-1):
            m = []
            for j in word:
                if j in dict(sum_result[i+1]):
                    m.append(sum_result[i+1][j])
                else:
                    m.append(0)
            final.loc[i] = m 
            
        col = list(final.columns)
        col.sort(key=lambda x:len(x),reverse=True) 
        for i in col:
            i = i.replace('+','\+')
            i = i.replace('[','\[')
            wordlist = set()
            for j in col:
                if bool(1-(j in wordlist)) and i != j and (re.search("^" + i + " ",j) or re.search(" "+ i + " ",j) or re.search(" " + i + "$" ,j)):
                    for k in range(len(final[i])):
                        final.loc[k,i] = max(final.loc[k,i]-final.loc[k,j],0)
                    wordlist.add(j)
        
        return final
    
    def run(self):
        filenames = [file for file in os.listdir(self.path) if '.csv' in file]
        with open("all_concepts.json", "r", encoding = "utf-8") as input:
            all_concepts = json.load(input)
        word = list(all_concepts.keys())
        for filename in filenames:
            data = self.file_reader(filename)
            bookname = filename.split('_')[4][:-4]
            print("=== working on", bookname, "===")
            data_chapter = self.match_page_and_chapter(data, self.book_chapter_page[bookname])
            final = self.deal_one(data_chapter, word, self.book_chapter_page[bookname])
            final.to_csv(self.outpath+bookname+"_final.csv", encoding = 'utf-8', index=False)
            print(bookname, "has done\n")
    
    def join(self):
        final_filenames = [file for file in os.listdir(self.outpath) if '.csv' in file]
        final_filenames.sort()
        print(final_filenames)
        finals = []
        for final_filename in final_filenames:
            final = pd.read_csv(self.outpath+final_filename, encoding = 'utf-8')
            finals.append(final)
        finals1 = pd.concat(finals)
        finals1.to_csv("all.csv", encoding = 'utf-8', index=False)

def main():
    dealer = readygo()
    dealer.run()
    dealer.join()


if __name__ == '__main__':
    main()
