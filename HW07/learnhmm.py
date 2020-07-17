
# coding: utf-8

# In[16]:



# coding: utf-8

# In[5]:


import sys
import numpy as np


# if __name__ == '__main__':
#     train_input = sys.argv[0]
#     index_to_word = sys.argv[1]
#     index_to_tag = sys.argv[2]
#     hmmprior = sys.argv[3]
#     hmmemit = sys.argv[4]
#     hmmtrans = sys.argv[5]
    


def read_file(path):
    data =[]
    with open(path) as line:
        train = []
        for l in line:
            item = l.rstrip()
            item = l.rstrip("\n").split(" ")
            train.append(item)
        return train

class LearnHMM:
    def __init__(self, train_in, index_to_tag, index_to_word):
        self.train_Data = self.split_data(train_in)

        self.index_to_tag = [y for x in index_to_tag for y in x]
        self.index_to_word = [y for x in index_to_word for y in x]
        self.all_possible_cases = self.find_all_possible_cases(index_to_tag, index_to_word)
        self.all_possible_tag_cases = self.find_all_possible_tag_cases(index_to_tag)
        self.all_tag, self.all_word, self.num_tag, self.num_word, self.word_order = self.all_tag_word()
       
        self.Pi = self.compute_Pi()
        self.transition = self.compute_Transition()
        self.emission = self.compute_Emission()
       
       
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
       
    def all_tag_word(self):
        all_tag = []
        all_word = []
        for i in self.train_Data:
            for j in i:
                all_tag.append(j[0])
                all_word.append(j[1])
        words=[]
        for word in all_word:
            if word not in words:
                words.append(word)
               
        num_tag = len(np.unique(np.array(all_tag)))
        num_word = len(np.unique(np.array(all_word)))
        return all_tag, all_word, num_tag,num_word, word

    
    def compute_Pi(self):
        start = []
        for i in self.train_Data:
            start.append(i[0][0])
        Pi = np.ones((self.num_tag, 1))
        for s in start:
            for t in range (len(self.index_to_tag)):
                if s == self.index_to_tag[t]:
                    Pi[t] += 1
                   
        Pi = Pi/(np.sum(Pi,0))
        Pi = Pi.flatten()

        return Pi
           
    def compute_Transition(self):
        transition = np.ones((self.num_tag,self.num_tag))
       
        for a in range (self.num_tag):
            for b in range (self.num_tag):
                for i in range (len(self.train_Data)):
                    for j in range (len(self.train_Data[i])-1):
                        if (self.train_Data[i][j][0] == self.index_to_tag[a] and self.train_Data[i][j+1][0] == self.index_to_tag[b]):
                            transition[a,b] +=1
                                       
        transition = transition.T/(np.sum(transition,1))
        return transition.T
   
    def compute_Emission(self):
        emission = np.ones((self.num_tag,self.num_word))
        for a in range (len(self.train_Data)):
            for b in range(len(self.train_Data[a])):
                i = self.index_to_tag.index(self.train_Data[a][b][0])
                j = self.index_to_word.index(self.train_Data[a][b][1])
                emission[i,j] += 1

        sum_vec= np.sum(emission,axis=1)
        for inx in range (len(emission)):
            emission[inx,:] = emission[inx,:]/sum_vec[inx]  
        return emission
   
    def find_alpha(self,data):
        ALPHA = []
        for n in range(len(data)):
            alpha = np.zeros((len(data[n]),self.num_tag))
            for z in range(self.num_tag):
                alpha[0,z] = self.Pi[z] * self.emission[z,int(self.index_to_word.index(data[n][0][1]))]
            for t in range(1,len(data[n])):
                xt = int(self.index_to_word.index(data[n][t][1]))
                for j in range(self.num_tag):
                    part1 = 0
                    for l in range(self.num_tag):
                        part1 += alpha[(t-1),l]*self.transition[l,j]
                    alpha[t,j] = self.emission[j,xt] * part1
            ALPHA.append(alpha)
        return ALPHA

   
    def find_beta(self,data):
        BETA = []
        for n in range(len(data)):
            beta = np.zeros((len(data[n]),self.num_tag))
            for z in range(self.num_tag):
                beta[len(data[n])-1,z] = 1
            for t in range(len(data[n])-2,-1,-1):
                xt = int(self.index_to_word.index(data[n][t+1][1]))
                for j in range(self.num_tag):
                    part1 = 0
                    for l in range(self.num_tag):
                        part1 += self.emission[l,xt] * beta[(t+1),l] * self.transition[j,l]
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
           


train_input = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/hw7/hw7/handout/trainwords.txt"
index_to_word = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/hw7/hw7/handout/index_to_word.txt"
index_to_tag = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/hw7/hw7/handout/index_to_tag.txt"


# train_input = read_file("./"+train_input)
# index_to_word = read_file("./"+index_to_word)
# index_to_tag = read_file("./"+index_to_tag)

train_input = read_file(train_input)
index_to_word = read_file(index_to_word)
index_to_tag = read_file(index_to_tag)

hmm = LearnHMM(train_input, index_to_tag, index_to_word)

hmmprior = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/GradeScoop/hmmprior.txt"
hmmemit = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/GradeScoop/hmmemit.txt"
hmmtrans = "C:/Users/bnabi/Desktop/Master/Spring 2020/Machine Learning/HW07/GradeScoop/hmmtrans.txt"

hmmprior_out = open (hmmprior,'w')
for x in range(len(hmm.Pi)):
    t = ''
    t = str(hmm.Pi[x])
    hmmprior_out.writelines(t+'\n')
hmmprior_out.close()

hmmemit_out = open (hmmemit,'w')
for x in range(len(hmm.emission)):
    t = ''
    for y in range(len(hmm.emission[x])):
        t = str(hmm.emission[x][y])
        hmmemit_out.writelines(t+' ')
    hmmemit_out.writelines('\n')
hmmemit_out.close()


hmmtrans_out = open (hmmtrans,'w')
for x in range(len(hmm.transition)):
    t = ''
    for y in range(len(hmm.transition[x])):
        t = str(hmm.transition[x][y])
        hmmtrans_out.writelines(t+' ')
    hmmtrans_out.writelines('\n')
hmmtrans_out.close()




