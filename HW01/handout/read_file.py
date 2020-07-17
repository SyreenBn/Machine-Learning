import csv
data = []
dataA = []
dataB = []
majorityVoteA = ""
majorityVoteB = ""
with open('education_train.tsv') as tsvfile:
          reader = csv.reader(tsvfile, delimiter='\t')
          for row in reader:
              data.append(row)
          data = data[1:]  
def splitData(data, index):
    A = data[0][int(index)]
    dataA.append(data[0])
    for i in range (1,len(data)):
        if data[i][4] == A:
            dataA.append(data[i])
        else:
            dataB.append(data[i])

def magorityVote(datalist):
    checkA = datalist[0][int(len(datalist)-1)]
    checkB = ""
    a = 0
    b = 0 
    for j in range (0,len(data)):
        print(datalist[j][1])
        if datalist[j][int(len(datalist[j])-1)] == checkA:
            a +=1
        else:
            checkB = datalist[j][int(len(datalist[j])-1)]
            b +=1
    if a > b :
        return checkA
    else:
        return checkB
        

            
splitData(data,4)           
#print (dataA[len(dataA)-1])

#print (dataB[1])
 
magorityVoteA = magorityVote(dataA)
print (magorityVote)
