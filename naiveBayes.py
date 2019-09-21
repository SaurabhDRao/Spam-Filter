from getStopWords import getStopWords

stopWords = getStopWords()

def getWordCount(word, wcDict, bow):
    if(word in wcDict):
        wcDict[word] *= wcDict[word]
    else:
        if(word in bow):
            wcDict[word] = bow[word]
        else:
            wcDict[word] = 0
    return wcDict
        
def getDictProbability(wcDict, n, nuq, alpha):
    pwDict = dict()
    flag = 0

    for w in wcDict:
        if(wcDict[w] == 0):
            flag = 1
            break

    for w in wcDict:
        if(flag):
            pwDict[w] = (wcDict[w] + alpha) / (n + (alpha * nuq))
        else:
            pwDict[w] = wcDict[w] / n

    return pwDict

def naiveBayes(testStr, bowAll, bowSpam, bowHam):
    nWords = sum(bowAll.values())
    nUniqueWords = len(bowAll)
    nSpam = sum(bowSpam.values())
    nHam = sum(bowHam.values())
    pSpam = nSpam / nWords
    pHam = nHam / nWords
    # print(nWords, nSpam, nHam, nUniqueWords)

    wcSpam = dict()
    wcHam = dict()
    wcAll = dict()

    words = [w for w in testStr.split() if (w.lower() not in stopWords)]
    # print(words)

    for word in words:
        wcSpam = getWordCount(word, wcSpam, bowSpam)
        wcHam = getWordCount(word, wcHam, bowHam)
        wcAll = getWordCount(word, wcAll, bowAll)

    pwSpam = getDictProbability(wcSpam, nSpam, nUniqueWords, 1)
    pwHam = getDictProbability(wcHam, nHam, nUniqueWords, 1)
    pwAll = getDictProbability(wcAll, nWords, nUniqueWords, 1)
    
    # print(wcSpam, wcHam, sep = "\n")

    pXS = 1
    pXNS = 1
    pX = 1
    for p in pwSpam:
        pXS *= pwSpam[p]
    for p in pwHam:
        pXNS *= pwHam[p]
    for p in pwAll:
        pX *= pwAll[p]

    pSX = (pXS * pSpam) / pX
    pNSX = (pXNS * pHam) / pX

    if(pSX > pNSX):
        return "spam"
    else:
        return "ham"