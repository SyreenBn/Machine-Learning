
# coding: utf-8

# In[4]:


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
        self.split_index = -1
        self.list_split_index = []

class Tree:
    def __init__(self, trainData, testData, Max_depth):
        self.header = trainData[0]
        self.root = Node(trainData[1:])
        self.test_data = testData[1:]
        
        self.label = []
        self.splitIndx = []
        
        self.StoreLabel(self.root.data)
        
        self.train_error = 0
        self.test_error = 0
        
        
        self.MAX_DEPTH = Max_depth
        self.leaves_nodes = []
        
        self.train_predict = []
        self.test_predict = []
        
        self.cal_attr_gini(self.root)
        
        self.evaluate('tr')
        self.evaluate('ts')
        
        self.calculateError(self.root.data, self.train_predict,'tr')
        self.calculateError(self.test_data, self.test_predict,'ts')
        
        
    def StoreLabel(self, data):
        self.label.append(data[0][-1])
        for i in range (1, len(data)):
            if data[i][-1] != data[0][-1]:
                self.label.append(data[i][-1])
                return
                       
            
        
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
        a = 0
        b = 0
        for j in range (0,len(datalist)):
            if datalist[j][-1] == self.label[0]:
                a +=1
            else:
                b +=1
        if a == b :
            if self.label[0] > self.label:
                node.majority_vote = self.label[0]
            else:
                node.majority_vote = self.label[1]
        else:    
            if a > b:
                node.majority_vote = self.label[0]
            else:
                node.majority_vote = self.label[1]
            
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
            list_argmax_ginis = [i for i, x in enumerate(G_attr) if x == largest_gini]
            if (largest_gini > 0):
                best_attr_index = np.max(list_argmax_ginis)
                left_right = self.best_attr_split(node,best_attr_index)
                node.left.split_index = best_attr_index
                node.right.split_index = best_attr_index
                for i in node.list_split_index:
                    node.left.list_split_index.append(i)
                    node.right.list_split_index.append(i)
                self.cal_attr_gini(left_right[0])
                self.cal_attr_gini(left_right[1])
            else:           
                self.majorityVote(node)
                self.leaves_nodes.append(node)
        else:
            self.majorityVote(node)
            self.leaves_nodes.append(node)

    def printTree(self):
        print(self.printedTree(self.root))
        
    def printedTree(self, node ,level=0):
        data = node.data
        collection = []
        for i in data:
            collection.append(i[-1])
        unique = np.unique(collection)
#         print (unique)
#         print(node.left.data)
        if node.split_index >=0:
            header = self.header[node.split_index] + " = " + node.data[0][node.split_index] + " : "
        else :
            header = ""

        strg = header + "[" + str(collection.count(self.label[0])) + " " + self.label[0] + " /"+ str(collection.count(self.label[1])) + " " + self.label[1] + "]"
        ret = "| "*level +strg+"\n"
        if node.left != None and node.right != None:
            ret += self.printedTree(node.left, level+1)
            ret += self.printedTree(node.right, level+1)
#         out = print(ret)
        return ret
        
    
    def calculateError(self, data_label, data_predict,char):    
        error = 0
        for i in range (0, len(data_label)):
            if data_label[i][-1] != data_predict[i]:
                error += 1
        error = error/len(data_label)
        if char == 'tr':
            self.train_error = error
        else:
            self.test_error = error
    
    def best_attr_split(self,node,indx):
        self.splitIndx.append(indx)
        node.list_split_index.append(indx)
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
        for i in range (0, len(data)):
            for j in self.leaves_nodes:
                check = True
                indx_counter = 0
                while (check == True and indx_counter < len(j.list_split_index)):
                    if (data[i][j.list_split_index[indx_counter]] != j.data[0][j.list_split_index[indx_counter]]):
                        check = False
                        indx_counter = len(j.list_split_index)
                    else:
                        indx_counter += 1
                        
                if (check is True):
                    predict.append(j.majority_vote)
                    
        if char == 'tr':
            self.train_predict = predict 
        else:
            self.test_predict = predict[:]
    
class_ex = [['Y','A','B'],[1,0,0],[1,0,0],[1,0,1],[1,0,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]
small_data_train = read_file('C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW02/handout/small_train.tsv')
small_data_test = read_file('C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW02/handout/small_test.tsv')

education_data_train = read_file('C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW02/handout/education_train.tsv')
education_data_test = read_file('C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW02/handout/education_test.tsv')

# for i in range (3,4):
#     print (education_data_train[i])
#     print()

education_tree_train = Tree(education_data_train,education_data_test ,3)
education_tree_train.printTree()
# print (education_tree_train.train_error)
# print (education_tree_train.test_error)
# print (education_tree_train.splitIndx)

# for j in education_tree_train.leaves_nodes:
# #     print (j.data[0]) 
# #     print(j.majority_vote)
#     print(j.list_split_index)
# print(education_tree_train.printedTree(education_tree_train.root, 0))
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
print(small_data_train_tree.gini(small_data_train_tree.root.data))
# print(small_data_train_tree.train_predict)
# print(small_data_train_tree.train_error)
# print(small_data_train_tree.test_error)
# print(small_data_train_tree.printedTree(small_data_train_tree.root, 0))

# small_data_train_tree.evaluate('tr')
# small_data_test_tree = Tree(small_data_test[1:],3)
# for i in small_data_train_tree.leaves_nodes:
#     print (i.data)
# print(small_data_train_tree.test_predict)
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
# print(small_data_train_)

