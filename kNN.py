from utils import preprocess
from math import log10

def kNN(testStr, tfIdfVector, bowAll, nSentences, wordList, allSentencesListWithLabel, k):
    nNvalues = []

    testStr = preprocess(testStr)
    
    nUniqueWords = len(wordList)

    vector = [0] * nUniqueWords

    for i in range(nUniqueWords):
        if(wordList[i] in testStr):
            vector[i] += 1

    similarity = [0] * nSentences
    # print(len(wordList), nUniqueWords, nSentences, len(tfIdfVector), len(tfIdfVector[0]), len(vector))
    for i in range(nSentences):
        res = 0
        for j in range(nUniqueWords):
            if((tfIdfVector[i][j] > 0) and (vector[j] > 0)):
               res += (log10(1 / (bowAll[wordList[j]] / nUniqueWords))) 
        similarity[i] = res

    # print(similarity)

    return allSentencesListWithLabel[similarity.index(max(similarity))][0]