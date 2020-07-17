import sys
import numpy as np
import csv
from collections import Counter


if __name__ == '__main__':
    
#     train_input = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/smalldata/train_data.tsv"
#     validation_input = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/smalldata/valid_data.tsv"
#     test_input = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/smalldata/test_data.tsv"
#     dict_input = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/dict.txt"
        
#     formatted_train_out = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/smalldata/formatted_train_data.tsv"
#     formatted_validation_out = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/smalldata/formatted_valid_data.tsv"
#     formatted_test_out = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW04/handout/smalldata/formatted_test_data.tsv"
#     feature_flag = 2

    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
        
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = sys.argv[8]

class Feature:
    def __init__(self,train_data, valid_data, test_data, dic_data, model):
        self.word_in_dict = self.dict_file_to_list(dic_data)
        (self.formatted_train_data,self.formatted_valid_data,self.formatted_test_data) = self.choose_model(train_data, valid_data, test_data, model)
        

    # This function take a list data and return a dictionary where each word hold its index 
    def dict_file_to_list(self,data):
        dic = {}
        for i in data:
            splt = i[0].split(" ")
            dic[splt[0]] = splt[1]
        return dic

    # This function is take a list of data and return a list of rate 1 where the movie is good while 0 is when the movie is not good
    def rate_reviews(self,data):
        rate = []
        reviews = []
        for i in data:
            rate.append(i[0])
            reviews.append(i[1])
        return (rate, reviews)

    # This function is take a list of reviews data and return a list of dictionary where each dictionary of word and its index  
    def occure_formatted_reviews(self,reviews):
        formatted_review=[]
        for i in range(0,len(reviews)):
            f = reviews[i].split()
            formatted = {}
            for j in range (0, len(f)):
                if f[j] in self.word_in_dict:
                    formatted.update({self.word_in_dict[f[j]]:1})
                else:
                    continue
            formatted_review.append(formatted)
        return formatted_review

    def trim_formatted_reviews(self,reviews):
        formatted_review=[]
        for i in range(0,len(reviews)):
            f = reviews[i].split()
            formatted = {}
            counts = Counter(list(f))
            f =[j for j,k in counts.items() if k <4 and j in self.word_in_dict]
            
            for l in range(0,len(f)):
                formatted.update({self.word_in_dict[f[l]]:1})
            formatted_review.append(formatted)
        return formatted_review


    # formatted the data label \t index:1 index2:1 index3:1 ....
    def dic_to_str(self,strg, dic):
        formatted_str = strg + '\t'
        for i in dic:
            formatted_str = formatted_str + i + ":"+"1\t"
        formatted_str = formatted_str.rstrip('\t\n ')
        return formatted_str

    # this function take two lists, the first rate(label1, label2, ....) and 
    # the second one formatted_review({index:1 index2:1 index3:1 ...}, {index:1 index2:1 index3:1 ...}, ....)
    def formatted_data(self,formatted_review, rate):
        formatted_all_data = []
        count = 0
        for count in range (0, len(rate)):
            formatted_all_data.append(self.dic_to_str(rate[count],formatted_review[count]))
        return formatted_all_data

    # This fuction is for occure model
    def occure (self,data):
        (rate,reviews) = self.rate_reviews(data)
        formatted_review = self.occure_formatted_reviews(reviews)
        formatted_all_data = self.formatted_data(formatted_review, rate)
        return formatted_all_data

    def trim (self,data):
        (rate,reviews) = self.rate_reviews(data)
        formatted_review = self.trim_formatted_reviews(reviews)
        formatted_all_data = self.formatted_data(formatted_review, rate)
        return formatted_all_data
    
    def choose_model(self, train, valid, test, model):
        if model == 1:
            train_data = self.occure (train)
            valid_data = self.occure (valid)
            test_data = self.occure (test)
            
        elif model == 2:
            train_data = self.trim (train)
            valid_data = self.trim (valid)
            test_data = self.trim (test)
        else:
            train_data = []
            valid_data = []
            test_data = []
        return (train_data,valid_data,test_data)
            
# This function is to read the information from a file and return a list of data
def read_file(path): 
    data =[]
    with open(path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            data.append(row)
    return data

# This function is to write the information lst to a file
def write_file(path,lst):
    f_output =  open("./"+path, 'w')
    strg=""
    for i in lst:
        strg = strg + i +"\n"
    f_output.write(strg)
    f_output.close()
        
    


train_data = read_file(train_input)
valid_data = read_file(validation_input)
test_data = read_file(test_input)
dict_data = read_file(dict_input)
model_num = int(feature_flag)

f = Feature(train_data, valid_data, test_data, dict_data, model_num)

write_file(formatted_train_out, f.formatted_train_data)
write_file(formatted_validation_out, f.formatted_valid_data)
write_file(formatted_test_out, f.formatted_test_data)



