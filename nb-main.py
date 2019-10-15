import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from prettytable import PrettyTable
from time import time

import colorama
colorama.init()

table = PrettyTable(["Dataset information", "Values"], )
table.align["Dataset information"] = "l"

programStart = time()

df = pd.read_csv("./newSpamDataSet.csv", sep = ",")

x = df["Message"]
y = df["Category"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 42)
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(xTrain.values)

print(colorama.Fore.LIGHTBLUE_EX)

classifier = MultinomialNB()
targets = yTrain.values
classifier.fit(counts, targets)

predictions = classifier.predict(vectorizer.transform(xTest))

programEnd = time()

# print(metrics.classification_report(yTest, predictions))
# print("Accuracy: ", round(metrics.accuracy_score(yTest, predictions) * 100), "%")
# print("Total time: ", round((programEnd - programStart), 4), "sec")

table.add_row(["Total data", 2047])
table.add_row(["Total train data", xTrain.shape[0]])
table.add_row(["Total test data", xTest.shape[0]])
table.add_row(["Accuracy", str(round(metrics.accuracy_score(yTest, predictions) * 100)) + " %"])
# table.add_row(["Total time", str(round((programEnd - programStart), 4)) + " sec"])
print(table)