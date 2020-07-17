
# coding: utf-8

# In[14]:


import numpy as np
import csv 


def read_file(path): 
    data =[]
    with open(path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            data.append(row)
    return data

def prob (col, value):
        return col.count(value)/len(col)
    
    
def gini(lst):
        col = []
        prob_squr = 0
        for i in lst:
            col.append(i[-1])
        unique_value = np.unique(col)
        for i in unique_value:
            prob_squr = prob_squr - (prob(col,i))**2
        gini = 1 + prob_squr
        return gini         
            
                
def majorityVote(data):
    y = list(np.array(data)[1:,-1])
    u = np.unique(y)
    a = 0
    b = 0
    for j in range (0,len(y)):
        if y[j] == u[0]:
            a +=1
        else:
            b +=1
    if a == b :
        lex = u
        lex.sort()
        majority_vote = lex[1]
    else:    
        if a > b:
            majority_vote = u[0]
        else:
            majority_vote = u[1]
            
    return majority_vote
            
def calculateError(data, majority):
    error = 0
    for i in range (0, len(data[1:])):
        if data[i][-1] != majority:
            error += 1
    error = error/len(data[1:])
    return error    

small_data_train = read_file('C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW02/handout/small_train.tsv')
small_data_test = read_file('C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW02/handout/small_test.tsv')

g = gini(small_data_train[1:])
m = majorityVote(small_data_train)
e = calculateError(small_data_train, m)
print(g,m,e)

