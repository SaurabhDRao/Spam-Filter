from main_NB import useNaiveBayes
from main_kNN import useKNN

import colorama
colorama.init()

print(colorama.Fore.CYAN)

print("nb - Naive Bayes\tknn - k nearest neighbour")
choice = input("Enter which algorithm to use: ")
if(choice == "nb"):
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Using Naive Bayes, please wait...")
    print(colorama.Style.RESET_ALL)
    useNaiveBayes()
elif(choice == "knn"):
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Using K Nearest Neighbour, please wait...")
    print(colorama.Style.RESET_ALL)
    useKNN()
else:
    print(colorama.Fore.RED)
    print("Invalid choice!")
    print(colorama.Style.RESET_ALL)