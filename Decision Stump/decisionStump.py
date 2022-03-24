
# coding: utf-8

# In[14]:


import sys
import csv
import numpy as np

if __name__ == '__main__':
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    split_index = sys.argv[3]
    
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics = sys.argv[6]



class Node:
    def __init__(self, data):
        self.data = data
        self.majority_vote = ""
        self.attr = ""

class Tree:
    def __init__(self, user_data, testData, index):
        self.root = Node(user_data)
        self.right = Node([])
        self.left = Node([])
        self.testData = testData
        self.index = index
        
        self.splitData()


    
    def splitData(self):
        A = self.root.data[0][int(self.index)]
        self.right.data.append(self.root.data[0])
        for i in range (1,len(self.root.data)):
            if self.root.data[i][self.index] == A:
                self.right.data.append(self.root.data[i])
                self.right.attr = A
            else:
                self.left.data.append(self.root.data[i])
                self.left.attr = self.root.data[i][int(self.index)]
        
        self.majorityVote(self.right)
        self.majorityVote(self.left)
        
        
                
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
            
        
    def test_predict(self):
        test_pred =[]
        test_error = 0
        for i in range(0, len(self.testData)):
            if self.testData[i][self.index] == self.right.attr:
                test_pred.append(self.right.majority_vote)
                test_error += self.error(self.testData[i][-1],self.right.majority_vote)
            else:
                test_pred.append(self.left.majority_vote)
                test_error += self.error(self.testData[i][-1],self.left.majority_vote)
        test_error = test_error/len(self.testData)
        return (test_pred,test_error)
    
    def train_predict(self):
        train_pred =[]
        train_error = 0
        for i in range(0, len(self.root.data)):
            if self.root.data[i][self.index] == self.right.attr:
                train_pred.append(self.right.majority_vote)
                train_error += self.error(self.root.data[i][-1],self.right.majority_vote)
            else:
                train_pred.append(self.left.majority_vote)
                train_error += self.error(self.root.data[i][-1],self.left.majority_vote)
        train_error = train_error/len(self.root.data)
        return (train_pred,train_error)    

    def error(self, a, b):
        if a == b:
            return 0
        else:
            return 1
        

def read_file(path):
    data =[]
    with open('./'+path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            data.append(row)
    return data [1:]

def write_file(path, predictions, mets=False):
    text = open('./'+path, 'w')
    if mets == True:
        text.write('error(train): '+str(predictions[0])+'\n'+'error(test): '+str(predictions[1]))
    else:
        text.write("\n".join(predictions))
    
    text.close()
    return text

train_data = read_file(train_in)
test_data = read_file(test_in)
indx = int(split_index)
t = Tree(train_data,test_data,indx)

write_file(train_out, t.train_predict()[0], mets = False)
write_file(test_out, t.test_predict()[0], mets = False)
write_file(metrics, [t.train_predict()[1], t.test_predict()[1]], mets = True)

