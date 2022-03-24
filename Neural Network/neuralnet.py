
# coding: utf-8

# In[21]:


import sys
import numpy as np
import math
import csv 



if __name__ == '__main__':
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics = sys.argv[5]
    epoch = sys.argv[6]
    nb_hidden = sys.argv[7]
    flag_in = sys.argv[8]
    learning_rate = sys.argv[9]


class NeuralNetwork:
    def __init__(self, train, test, epoch, hidden, flag, lr):
        self.flag = int(flag)
        self.learning_rate = float(lr)
        self.nb_hidden = int(hidden)
        self.epoch = int(epoch)
        
        self.X_train_matrix, self.X_test_matrix, self.Y_train_vector, self.Y_test_vector = self.format_data(train, test)
        
        self.Y_train_encode = self.oneHotEncoding(self.Y_train_vector)
        self.Y_test_encode = self.oneHotEncoding(self.Y_test_vector) 
        
        self.X_train_matrix = np.transpose(np.append(np.ones((1,np.shape(self.X_train_matrix)[0])),self.X_train_matrix.T, axis=0))
        self.X_test_matrix = np.transpose(np.append(np.ones((1,np.shape(self.X_test_matrix)[0])), self.X_test_matrix.T, axis=0))
        
        self.ALPHAS = [] 
        self.BETAS = [] 
        
        self.SGDTrain()
        
        self.cross_entropy_train, self.train_error, self.train_pred = self.evaluate(self.X_train_matrix, self.Y_train_encode, 1)        
        self.cross_entropy_test, self.test_error, self.test_pred = self.evaluate( self.X_test_matrix, self.Y_test_encode, 2)
        
    def sigmoid(self,a):
        return 1 / (1 + np.exp(-a))
    
    def softmax(self, b):
        return np.exp(b) / np.sum(np.exp(b), axis=0)
    
    def CrossEntropyLoss(self, Y, Y_hat):
        return -1*Y.dot(np.log(Y_hat))
    
    def predict(self, X,Y,alpha, beta):
        prediction = []
        for i in range(0, len(X)):
            a, z, b, Y_hat, loss = self.forward(X[i,:],Y[i,:], alpha, beta)
            prediction.append(np.argmax(Y_hat))
        return prediction
    
    def format_data(self, train_data, test_data):
        train_data = np.array(train_data)
        X_train_matrix = train_data[:,1:int(train_data.shape[1])]
        
        Y_train_vector = train_data[:,0]
        Y_train_vector = np.vstack(Y_train_vector*1)
        
        test_data = np.array(test_data)
        X_test_matrix = test_data[:,1:int(test_data.shape[1])]
        
        Y_test_vector = test_data[:,0]
        Y_test_vector = np.vstack(Y_test_vector*1)
        
        return X_train_matrix, X_test_matrix, Y_train_vector, Y_test_vector
        
    def oneHotEncoding(self, Y):
        y= []
        for i in Y:
            y.append(int(i))

        if len(np.unique(Y)) >=10:
            return np.eye(len(np.unique(y))).astype(int)[np.array(y).reshape(-1)] 
        else:
            return np.eye(10).astype(int)[np.array(y).reshape(-1)] 
    
    def initialization(self, Xtr, nb_classes):
        if self.flag == 1:
            alpha = np.random.uniform(-0.1,0.1,size = (self.nb_hidden, np.shape(Xtr)[1]))
            alpha[:,0] = 0
            beta = np.random.uniform(-0.1,0.1,size = (nb_classes, self.nb_hidden+1))
            beta[:,0] = 0
        else:
            alpha = np.zeros((self.nb_hidden, np.shape(Xtr)[1]))
            beta = np.zeros((nb_classes, self.nb_hidden+1))
        return alpha, beta
    
    def forward(self, X_Data, Y_Data, alpha, beta):
            
        X_Data = X_Data.reshape(1,np.shape(X_Data)[0])
        Y_Data = Y_Data.reshape(1,np.shape(Y_Data)[0])

        a = np.matmul(alpha,X_Data.T)
        
        z = self.sigmoid(a)
        z_bais = np.ones((1,np.shape(z)[1]))
        z = np.append(z_bais,z, axis=0)
        
        b = np.matmul(beta,z)

        Y_hat = self.softmax(b)

        loss = self.CrossEntropyLoss(Y_Data,Y_hat)
        
        return a, z, b, Y_hat, loss
    
    def updateParameters(self, alpha, beta, d_alpha, d_beta):
        alpha = alpha - (self.learning_rate * d_alpha)
        beta = beta - (self.learning_rate * d_beta)

        return alpha,beta
    
    def backward(self, X_Data, Y_Data, alpha, beta):
        a, z, b, Y_hat, loss = self.forward(X_Data, Y_Data, alpha, beta)
        
        X_Data = X_Data.reshape(1,np.shape(X_Data)[0])
        Y_Data = Y_Data.reshape(1,np.shape(Y_Data)[0])
        
        d_b = -Y_Data.T+Y_hat

        d_beta = np.matmul(d_b, z.T)
        
        d_z = np.matmul(beta[:,1:].T, d_b)
        
        d_a = np.multiply(d_z,np.multiply(z[1:,:],(1-z[1:,:])))

        d_alpha = np.matmul(d_a, X_Data)

        alpha, beta = self.updateParameters(alpha, beta, d_alpha, d_beta)
    
        return alpha, beta
    
    def SGDTrain(self):
        alpha, beta = self.initialization(self.X_train_matrix,self.Y_train_encode.shape[1])
   
        for e in range (self.epoch):
            for i in range (self.X_train_matrix.shape[0]):
                alpha,beta = self.backward(self.X_train_matrix[i,:], self.Y_train_encode[i,:],alpha,beta)
            self.ALPHAS.append(alpha.copy())
            self.BETAS.append(beta.copy())
            
    def evaluate(self, X, Y, f):
       
        if f == 1:
            Y_ = self.Y_train_vector
        else:
            Y_ = self.Y_test_vector
            
        cross_entropy = []
        for e in range(self.epoch):
            l = []
            for i in range(X.shape[0]):
                a, z, b, Y_hat, loss = self.forward(X[i,:], Y[i,:], self.ALPHAS[e], self.BETAS[e])
                l.append(loss)
            cross_entropy.append(np.mean(l))
            

        pred = self.predict(X,Y, self.ALPHAS[-1], self.BETAS[-1])
        er = [i for i in range(0, len(Y_)) if list(Y_)[i] != list(pred)[i]]
        error = float(len(er)/len(X))
        return cross_entropy, error, pred
    

    


def read_file(path): 
    data =[]
    with open(path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            data.append(row)
    return data

# train_in = 'C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW05/hw5/handout/smallTrain.csv'
# test_in = 'C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW05/hw5/handout/smallTest.csv'


train_in = np.loadtxt(fname=train_in, delimiter=",")
test_in = np.loadtxt(fname=test_in, delimiter=",")


nn= NeuralNetwork(train_in, test_in, epoch, nb_hidden, flag_in, learning_rate)

# nn= NeuralNetwork(train_in, test_in, 2, 4, 2, 0.1)
# print(nn.cross_entropy_train)
# print(nn.cross_entropy_test)
# print(nn.train_error)
# print(nn.test_error)


out_train_pred = open ('./'+train_out,'w')
for x in range(len(nn.train_pred)):
    t = ''
    t = str(nn.train_pred[x])
    out_train_pred.writelines(t+'\n')
out_train_pred.close()

out_test_pred = open ('./'+test_out,'w')
for x in range(len(nn.test_pred)):
    t = ''
    t = str(nn.test_pred[x])
    out_test_pred.writelines(t+'\n')
out_test_pred.close()

out_metrics = open ('./'+metrics,'w')
for x in range(len(nn.cross_entropy_train)):
    l1 = 'epoch='+str(x+1)+' crossentropy(train): '+str(nn.cross_entropy_train[x])
    l2 = 'epoch='+str(x+1)+' crossentropy(test): '+str(nn.cross_entropy_test[x])
    out_metrics.writelines(l1+'\n')
    out_metrics.writelines(l2+'\n')
e1 = 'error(train): '+str(nn.train_error)
e2 = 'error(test): '+str(nn.test_error)
out_metrics.writelines(e1+'\n')
out_metrics.writelines(e2+'\n')
out_metrics.close()


