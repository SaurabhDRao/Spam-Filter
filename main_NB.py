from naiveBayes import naiveBayes
from utils import preprocess, bagOfWords
from random import shuffle
from prettytable import PrettyTable

def useNaiveBayes():
	table = PrettyTable(["Dataset information", "Values"], )
	table.align["Dataset information"] = "l"

	import colorama
	colorama.init()

	f = open("./dataset.csv")

	allSentencesListWithLabel = []

	for line in f:
		sentenceList = preprocess(line.split(",", 1)[1])
		allSentencesListWithLabel.append([line.split(",")[0], sentenceList])

	allSentencesListWithLabel.pop(0)

	shuffle(allSentencesListWithLabel)

	allSentencesList = [sentence[1] for sentence in allSentencesListWithLabel[:1600]]
	spamSentencesList = [sentence[1] for sentence in allSentencesListWithLabel[:1600] if (sentence[0] == "spam")]
	hamSentencesList = [sentence[1] for sentence in allSentencesListWithLabel[:1600] if (sentence[0] == "ham")]
	testSentencesList = allSentencesListWithLabel[1600:]

	# print(allSentencesList, spamSentencesList, hamSentencesList)

	bowAll = bagOfWords(allSentencesList)
	bowSpam = bagOfWords(spamSentencesList)
	bowHam = bagOfWords(hamSentencesList)

	print(colorama.Fore.LIGHTBLUE_EX)

	# print(bowAll, bowSpam, bowHam, sep = "\n")

	count = 0
	for testList in testSentencesList:
		res = naiveBayes(testList[1], bowAll, bowSpam, bowHam)
		# print(testList[0], res)
		if(res == testList[0]):
			count += 1


	table.add_row(["Total data", len(allSentencesListWithLabel)])
	table.add_row(["Total train data", (len(spamSentencesList) + len(hamSentencesList))])
	table.add_row(["Total spams in test data", len(spamSentencesList)])
	table.add_row(["Total hams in test data", len(hamSentencesList)])
	table.add_row(["Total test data", len(testSentencesList)])
	table.add_row(["Total data in test correctly classified", count])
	table.add_row(["Accuracy", str(round(((count * 100) / len(testSentencesList)), 2)) + " %"])
	print(table)

	print(colorama.Style.RESET_ALL)

# useNaiveBayes()