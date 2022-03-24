
# coding: utf-8

# In[69]:


import sys
import numpy as np

# if __name__ == '__main__':
#     test_input = sys.argv[0]
#     index_to_word = sys.argv[1]
#     index_to_tag = sys.argv[2]
#     hmmprior = sys.argv[3]
#     hmmemit = sys.argv[4]
#     hmmtrans = sys.argv[5]
#     predicted_file = sys.argv[6]
#     metric_file = sys.argv[7]
    

def read_file(path):
    data =[]
    with open(path) as line:
        train = []
        for l in line:
            item = l.rstrip()
            item = l.rstrip("\n").split(" ")
            train.append(np.array(item))
        return train

class LearnHMM:
    def __init__(self, test_in, index_to_tag, index_to_word, hmmprior, hmmemit, hmmtrans ):
        self.test_Data = self.split_data(test_in)

        self.index_to_tag = [y for x in index_to_tag for y in x]
        self.index_to_word = [y for x in index_to_word for y in x]
        self.all_possible_cases = self.find_all_possible_cases(index_to_tag, index_to_word)
        self.all_possible_tag_cases = self.find_all_possible_tag_cases(index_to_tag)
#         self.all_tag, self.all_word, self.num_tag, self.num_word, self.word_order = self.all_tag_word()
       
        self.Pi = hmmprior
        self.transition = hmmtrans
        self.emission = hmmemit
       
        self.alpha = self.find_alpha(self.test_Data)
        self.beta = self.find_beta(self.test_Data)
       
        self.avg_likelihood = self.avg_likelihood(self.test_Data)
        self.error = self.predict(self.test_Data)[1]
        self.predictions = self.predict(self.test_Data)[0]
       
       
    def split_data(self, lst):
        tag_word_lst = []
        for i in lst:
            sub_tag_word_lst = []
            for j in i:
                tag_word = j.split("_")
                sub_tag_word_lst.append((tag_word[1],tag_word[0]))
            tag_word_lst.append(sub_tag_word_lst)
        return tag_word_lst
   
    def find_all_possible_cases(self, index_to_tag, index_to_word):
        all_possible_cases = []
        for i in (index_to_tag):
            for j in (index_to_word):
                all_possible_cases.append((i[0], j[0]))
        return all_possible_cases
   
    def find_all_possible_tag_cases(self, index_to_tag):
        all_possible_tag_cases = []
        for i in (index_to_tag):
            for j in (index_to_tag):
                all_possible_tag_cases.append((i[0], j[0]))
        return all_possible_tag_cases
   
    def find_alpha(self,data):
        ALPHA = []
        for n in range(len(data)):
            alpha = np.zeros((len(data[n]),len(self.index_to_tag)))
            inds = int(self.index_to_word.index(data[n][0][1]))
            for z in range(len(self.index_to_tag)):
#                 print(n,z)
#                 print(self.Pi[z][0])
#                 print(self.emission[int(z)])
                alpha[0,z] = float(self.Pi[int(z)][0]) * float(self.emission[int(z),inds])
            for t in range(1,len(data[n])):
                xt = int(self.index_to_word.index(data[n][t][1]))
                for j in range(len(self.index_to_tag)):
                    part1 = 0
                    for l in range(len(self.index_to_tag)):
                        part1 += alpha[(t-1),l]*float(self.transition[l,j])
                    alpha[t,j] = float(self.emission[j,xt]) * part1
            ALPHA.append(alpha)
        return ALPHA

   
    def find_beta(self,data):
        BETA = []
        for n in range(len(data)):
            beta = np.zeros((len(data[n]),len(self.index_to_tag)))
            for z in range(len(self.index_to_tag)):
                beta[len(data[n])-1,z] = 1
            for t in range(len(data[n])-2,-1,-1):
                xt = int(self.index_to_word.index(data[n][t+1][1]))
                for j in range(len(self.index_to_tag)):
                    part1 = 0
                    for l in range(len(self.index_to_tag)):
                        part1 += float(self.emission[l,xt]) * beta[(t+1),l] * float(self.transition[j,l])
                    beta[t,j] =  part1
            BETA.append(beta)
        return BETA
        
    def log_sum_exp(self, x):
        m = np.max(x)
        y = x-m
        return m + np.log(np.sum(np.exp(y)))

    def avg_likelihood(self,data):
        L = np.zeros((len(data),1))
        for n in range(len(data)):
            L[n,0] = np.log(np.sum(self.alpha[n][-1,:]))
        return np.mean(L)


    def predict(self,data):
        predictions = []
        for n in range(len(data)):
            Prop = self.alpha[n]*self.beta[n]
            preds = np.argmax(Prop,axis=1)
            predictions.append(preds)
        error = 0
        total = 0
        for i in range(len(data)):
            for j in range(len(data[i])):
                total +=1
                if index_to_tag[int(predictions[i][j])][0] == data[i][j][0]:
                    error += 1
        
        error = (error/total)
        return predictions, error
           

index_to_word = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/hw7/hw7/handout/index_to_word.txt"
index_to_tag = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/hw7/hw7/handout/index_to_tag.txt"
test_input = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/hw7/hw7/handout/testwords.txt"

hmmprior = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/GradeScoop/hmmprior.txt"
hmmemit = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/GradeScoop/hmmemit.txt"
hmmtrans = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/GradeScoop/hmmtrans.txt"


# hmmprior = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/hw7/hw7/handout/hmmprior.txt"
# hmmemit = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/hw7/hw7/handout/hmmemit.txt"
# hmmtrans = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/hw7/hw7/handout/hmmtrans.txt"

# test_input = read_file("./"+test_input)

# index_to_word = read_file("./"+index_to_word)
# index_to_tag = read_file("./"+index_to_tag)

# hmmprior = np.array(read_file("./"+hmmprior))
# hmmemit = np.array(read_file("./"+hmmemit))
# hmmtrans = np.array(read_file("./"+hmmtrans))


test_input = read_file(test_input)

index_to_word = read_file(index_to_word)
index_to_tag = read_file(index_to_tag)

hmmprior = np.array(read_file(hmmprior))
hmmemit = np.array(read_file(hmmemit))
hmmtrans = np.array(read_file(hmmtrans))


hmm = LearnHMM(test_input, index_to_tag, index_to_word, hmmprior, hmmemit, hmmtrans )
predicted_file = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/GradeScoop/predicted_file.txt"
metric_file = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/GradeScoop/metric_file.txt"

out_test_pred = open (predicted_file,'w')
for x in range(len(hmm.predictions)):
    t = ''
    t = str(hmm.predictions[x])
    out_test_pred.writelines(t+'\n')
out_test_pred.close()

out_metrics = open (metric_file,'w')
out_metrics.writelines("Average Log-Likelihood: " + str(hmm.avg_likelihood)+'\n')
out_metrics.writelines("Accuracy: "+str(hmm.error)+'\n')
out_metrics.close()


