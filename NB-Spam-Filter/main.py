from utils import preprocess, bagOfWords
from naiveBayes import naiveBayes
from random import shuffle
from prettytable import PrettyTable
from time import time

import colorama
colorama.init()

table = PrettyTable(["Dataset information", "Values"], )
table.align["Dataset information"] = "l"

f = open("./newSpamDataSet.csv")

allSentencesListWithLabel = []

for line in f:
	allSentencesListWithLabel.append([line.split(",")[0], line.split(",", 1)[1]])

allSentencesListWithLabel.pop(0)

shuffle(allSentencesListWithLabel)

allSentencesList = [preprocess(sentence[1]) for sentence in allSentencesListWithLabel[:1600]]
spamSentencesList = [preprocess(sentence[1]) for sentence in allSentencesListWithLabel[:1600] if (sentence[0] == "spam")]
hamSentencesList = [preprocess(sentence[1]) for sentence in allSentencesListWithLabel[:1600] if (sentence[0] == "ham")]
testSentencesList = allSentencesListWithLabel[1600:]

# print(allSentencesList, spamSentencesList, hamSentencesList)
programStart = time()

bowAll = bagOfWords(allSentencesList)
bowSpam = bagOfWords(spamSentencesList)
bowHam = bagOfWords(hamSentencesList)

# print(bowAll, bowSpam, bowHam, sep = "\n")

count = 0
spamCount = 0
hamCount = 0

for testList in testSentencesList:
    res = naiveBayes(preprocess(testList[1]), bowAll, bowSpam, bowHam)
    # print(testList[0], res)
    if(res == testList[0]):
        if(((res == "ham") and (hamCount < 2)) or ((res == "spam") and (spamCount < 2))):
            print("Input:", testList[1], end = "")
            print("Expected output:", testList[0])
            print("Predicted output:", res)
            print()
            if(res == "ham"): 
                hamCount += 1
            else: 
                spamCount += 1
        count += 1

programEnd = time()

print(colorama.Fore.LIGHTBLUE_EX)

table.add_row(["Total data", len(allSentencesListWithLabel)])
table.add_row(["Total train data", (len(spamSentencesList) + len(hamSentencesList))])
# table.add_row(["Total spams in test data", len(spamSentencesList)])
# table.add_row(["Total hams in test data", len(hamSentencesList)])
table.add_row(["Total test data", len(testSentencesList)])
# table.add_row(["Total correctly classified test data", count])
table.add_row(["Accuracy", str(round(((count * 100) / len(testSentencesList)), 2)) + " %"])
table.add_row(["Total time", str(round((programEnd - programStart), 4)) + " sec"])
print(table)

print(colorama.Style.RESET_ALL)