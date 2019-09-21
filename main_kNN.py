from kNN import kNN
from utils import preprocess, bagOfWords, tfIdfVectorizer
from random import shuffle
from prettytable import PrettyTable

def useKNN():
	table = PrettyTable(["Dataset information", "Values"], )
	table.align["Dataset information"] = "l"

	import colorama
	colorama.init()

	f = open("./dataset.csv", "r")

	allSentencesListWithLabel = []

	for line in f:
		sentenceList = preprocess(line.split(",", 1)[1])
		allSentencesListWithLabel.append([line.split(",")[0], sentenceList])

	allSentencesListWithLabel.pop(0)

	shuffle(allSentencesListWithLabel)

	allSentencesList = [sentence[1] for sentence in allSentencesListWithLabel[:1600]]
	testSentencesList = allSentencesListWithLabel[1600:]
	nSentences = len(allSentencesList)

	bowAll = bagOfWords(allSentencesList)
	nUniqueWords = len(bowAll)
	wordList = list(bowAll.keys())

	tfIdfVector = tfIdfVectorizer(bowAll, nSentences, wordList, nUniqueWords, allSentencesList)

	print(colorama.Fore.BLUE)

	# testStr = "winner you will receive $1000 as reward"
	count = 0
	for testList in testSentencesList:
		res = kNN(testList[1], tfIdfVector, bowAll, nSentences, wordList, allSentencesListWithLabel, 1)
		# print(testList[0], res)
		if(res == testList[0]):
				count += 1
	# print(allSentencesList, bowAll, wordList, tfIdfVector, sep = "\n")
	# print(kNN(testStr, tfIdfVector, bowAll, nSentences, wordList, allSentencesListWithLabel, 1))

	table.add_row(["Total data", len(allSentencesListWithLabel)])
	table.add_row(["Total test data", len(testSentencesList)])
	table.add_row(["Total data in test correctly classified", count])
	table.add_row(["Accuracy", str(round(((count * 100) / len(testSentencesList)), 2)) + " %"])
	print(table)

	print(colorama.Style.RESET_ALL)

# useKNN()