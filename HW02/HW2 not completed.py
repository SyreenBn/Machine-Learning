
# coding: utf-8

# In[16]:


import numpy as np
import csv 

def read_file(path): 
    data =[]
    with open(path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            data.append(row)
    return data


        

class Node:
    def __init__(self, data):
        self.right = None
        self.left = None
        self.data = data
        self.depth = 0
        self.majority_vote = ""

class Tree:
    def __init__(self, trainData, testData, Max_depth):
        self.root = Node(trainData[1:])
        self.test_data = testData[1:]
        self.MAX_DEPTH = Max_depth
        self.leaves_nodes = []
        
        self.train_error = 0
        self.test_error = 0
        
        self.train_predict = []
        self.test_predict = []
        
        self.cal_attr_gini(self.root)
        self.evaluate('tr')
        self.evaluate('ts')
        
        
    def prob (self, col, value):
        return col.count(value)/len(col)
    
    def splitData(self, data,index):
        A = data[0][index]
        lst1 =[]
        lst2 = []
        lst1.append(data[0])
        for i in range (1,len(data)):
            if data[i][index] == A:
                lst1.append(data[i])
            else:
                lst2.append(data[i])
        return(lst1,lst2)
                
    def majorityVote(self, node):
        datalist = node.data
        checkA = datalist[0][-1]
        checkB = ""
        a = 0
        b = 0 
        for j in range (0,len(datalist)):
            if datalist[j][-1] == checkA:
                a +=1
            else:
                checkB = datalist[j][-1]
                b +=1
        if a == b :
            node.majority_vote = np.random.choice((checkA,checkB))
        else:    
            if a > b:
                node.majority_vote = checkA
            else:
                node.majority_vote = checkB
            
    def cal_attr_gini(self,node):
        if (node.depth < self.MAX_DEPTH):
            data = node.data
            G_data = self.gini(data)
            G_attr = []
            for i in range (0,len(data[0])-1):
                splited_data = self.splitData(data,i)
                G_A = self.gini(splited_data[0])
                G_B = self.gini(splited_data[1])
                gini_gain = G_data - ((len(splited_data[0])/len(data)) * G_A) - ((len(splited_data[1])/len(data)) * G_B)
                G_attr.append(gini_gain)
            largest_gini = np.max(G_attr)
            if (largest_gini > 0):
                best_attr_index = G_attr.index(largest_gini)
                left_right = self.best_attr_split(node,best_attr_index)
                self.cal_attr_gini(left_right[0])
                self.cal_attr_gini(left_right[1])
            else:           
                self.majorityVote(node)
                self.predict(node)
        else:
            self.majorityVote(node)
            self.predict(node)
            
    def error(self, a, b):
        if a == b:
            return 0
        else:
            return 1
        
    def predict(self,node):
        data_lst = node.data
        data_pred =[]
        data_error = 0
        for i in range(0, len(data_lst)):
            data_pred.append(node.majority_vote)
            if data_lst[i][-1] != node.majority_vote:
                data_error += self.error(data_lst[i][-1],node.majority_vote)
            data_lst[i].append(node.majority_vote)
        self.train_error = self.train_error + (data_error/len(self.root.data))
        node.data = data_lst[:] 
        self.leaves_nodes.append(node)
    
    def best_attr_split(self,node,indx):
        data = node.data
        best_splited_data = self.splitData(data,indx)
        node.left = Node(best_splited_data[0])
        node.right = Node(best_splited_data[1])
        node.left.depth = node.depth + 1
        node.right.depth = node.depth + 1 
        return (node.left,node.right)
        
        
    def gini(self,lst):
        col = []
        prob_squr = 0
        for i in lst:
            col.append(i[-1])
        unique_value = np.unique(col)
        for i in unique_value:
            prob_squr = prob_squr - (self.prob(col,i))**2
        gini = 1 + prob_squr
        return gini
    
    def evaluate(self,char):
        if char == 'tr':
            data = self.root.data
        else: 
             data = self.test_data
                
        predict = []
        for i in data:
            for j in self.leaves_nodes:
                if i[:-2] == j.data[0][:-2]:
                    predict.append(j.data[0][-1])
        if char == 'tr':
            self.train_predict = predict 
        else:
            print(predict)
            self.test_predict = predict
    
class_ex = [['Y','A','B'],[1,0,0],[1,0,0],[1,0,1],[1,0,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]
small_data_train = read_file('C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW02/handout/small_train.tsv')
small_data_test = read_file('C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW02/handout/small_test.tsv')

education_data_train = read_file('C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW02/handout/education_train.tsv')
education_data_test = read_file('C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW02/handout/education_train.tsv')

# education_tree_train = Tree(education_data_train[1:],3)
# education_tree_test = Tree(education_data_test[1:],2)
# for i in education_data_train:
#     print (i)

# print(len(education_tree_train.root.data))
# print()
# print(len(education_tree_train.root.left.data))
# print()
# print(len(education_tree_train.root.right.data))
# print()
# print(len(education_tree_train.root.left.left.data))
# print()
# print(len(education_tree_train.root.left.right.data))
# print()
# print(len(education_tree_train.root.right.left.data))
# print()
# print(len(education_tree_train.root.right.right.data))


politicians_train = read_file('C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW02/handout/politicians_train.tsv')
politicians_test = read_file('C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW02/handout/politicians_test.tsv')

small_data_train_tree = Tree(small_data_train, small_data_test, 3)
small_data_train_tree.evaluate('tr')
# small_data_test_tree = Tree(small_data_test[1:],3)
# for i in small_data_train_tree.leaves_nodes:
#     print (i.data)
print(small_data_train_tree.test_predict)
# t = Tree(class_ex[1:],2)
# print(small_data_train_tree.root.data)
# print()
# print(small_data_train_tree.root.left.data)
# print()
# print(small_data_train_tree.root.right.data)
# print()
# print(small_data_train_tree.root.left.left.data)
# print()
# print(small_data_train_tree.root.left.right.data)
# print()
# print(small_data_train_tree.root.right.left.data)
# print()
# print(small_data_train_tree.root.right.right.data)




# t.cal_attr_gini(t.root)
# print(train.root.data)
# print(train.root.right.data)
# print(train.root.left.data)


# In[ ]:


import csv
import numpy as np

def read_file(path):
    data =[]
    with open(path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            data.append(row)
    return data


def prob (col, value):
        return col.count(value)/len(col)
    
    
data = read_file('C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW02/handout/small_train.tsv')
read = []
for i in range (1, len(data)):
    read.append(data[i][0])
class_ex = [['Y','A','B'],[1,0,0],[1,0,0],[1,0,1],[1,0,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]

def gini(col):
    unique_value = np.unique(col)
    if (len(unique_value) == 2):
        x = prob(col,unique_value[0])
        y = prob(col,unique_value[1])
    else:
        x = prob(col,unique_value[0])
        y = 0
    gini = 1 - ((x)**2) - ((y)**2)
    return gini
# print(np.array(class_ex)[:,0])
x = list(np.array(class_ex)[1:,1])
print(x)
gini(x)


# In[ ]:


Y

