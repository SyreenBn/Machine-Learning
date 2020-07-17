import sys
import numpy as np
import csv
import math

if __name__ == '__main__':
        
#     formatted_train_input = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/smalldata/formatted_train_data.tsv"
#     formatted_validation_input = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/smalldata/formatted_valid_data.tsv"
#     formatted_test_input = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/smalldata/formatted_test_data.tsv"
    
#     dict_input = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/dict.txt"
    
#     train_out = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/smalldata/train_out.labels"
#     test_out = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/smalldata/test_out.labels"
#     metrics_out = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/smalldata/metrics_out.txt"
    
#     num_epoch = 30

    formatted_train_input = sys.argv[1]
    formatted_validation_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    
    dict_input = sys.argv[4]
    
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    
    num_epoch = sys.argv[8]
    

class LR:
    def __init__(self, formatted_train, formatted_valid, formatted_test, dict_data, num_epoch):
        self.theta = [0]*(len(dict_data)+1)
        self.dict_data = dict_data
        
        (self.train_label,self.train_features) = self.stochastic_gradient_descent(formatted_train,num_epoch)
        (self.test_label,self.test_features) = self.clean_data(formatted_test)
        
        self.train_predict = self.predict(self.train_features)
        self.test_predict = self.predict(self.test_features)
        
        self.train_error = format(self.error(self.train_predict,self.train_label),".6f")
        self.test_error = format(self.error(self.test_predict,self.test_label), ".6f")
        
        
    def dot_product(self, X_vector):
        product = 0.0
        for k,v in X_vector.items():
            product += float(v) * self.theta[int(k)]
        return product
    
    def sigmoid(self,z):
        return math.exp(z)/(1+math.exp(z))
    
    def str_to_dic(self, strg):
        splt = strg.split(":")
        return (splt[0],splt[1])
    
    def clean_data(self,data_row):
        labels = []
        features=[]
        for i in range(len(data_row)):
            feature = {}
            labels.append(data_row[i][0])
            for j in range (1, len(data_row[i])):
                (key,value) = self.str_to_dic(data_row[i][j])
                feature[key] = value
            feature[len(self.dict_data)] = 1
            features.append(feature)
        return (labels,features)
    
    def stochastic_gradient_descent(self, data ,epoch_num):
        (labels,features) = self.clean_data(data)
        for i in range(epoch_num):
            for j in range (len(features)):
                z = self.sigmoid(self.dot_product(features[j]))
                for m,n in features[j].items():
                    self.theta[int(m)] += 0.1 * float(n) * (float(labels[int(j)])-z)
        return(labels,features)

    def predict(self, x_vector):
        prediction = []
        for i in range(len(x_vector)):
            if self.sigmoid(self.dot_product(x_vector[i])) >= 0.5:
                prediction.append('1')
            else:
                prediction.append('0')
        return prediction
    
    def error(self, prediction, label):
        error = [i for i in range(0,len(prediction)) if prediction[i] != label[i]]
        return len(error)/len(prediction)
        
# This function is to read the information from a file and return a list of data
def read_file(path): 
    data =[]
    with open(path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            data.append(row)
    return data
                
            
#def write_file(path, predictions, mets=False):
#    text = open("./" + path, 'w')
#    if mets == True:
#        text.write('error(train): '+str(predictions[0])+'\n'+'error(test): '+str(predictions[1]))
#    else:
#        text.write("\n".join(predictions))
#    
#    text.close()
#    return text

def write_file(path, predictions, mets=False):
    text = open('./'+path, 'w')
    if mets == True:
        text.write('error(train): '+str(predictions[0])+'\n'+'error(test): '+str(predictions[1]))
    else:
        text.write("\n".join(predictions))
    
    text.close()
    return text

train_data = read_file(formatted_train_input)
valid_data = read_file(formatted_validation_input)
test_data = read_file(formatted_test_input)
dict_data = read_file(dict_input)

lr = LR(train_data, valid_data, test_data, dict_data, int(num_epoch))


write_file(train_out, lr.train_predict, mets = False)
write_file(test_out, lr.test_predict, mets = False)
write_file(metrics_out, [lr.train_error, lr.test_error], mets = True)



