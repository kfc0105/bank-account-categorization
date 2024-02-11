import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

#Stream line the import of dataset and giving out the result

def predict(data, cv, clf, target):
    #docs_new has to be one dimentional list
    docs_new = data
    X_new_counts = cv.transform(docs_new)
    predicted = clf.predict(X_new_counts)
    result = []
    for i in predicted:
        result.append(target[1][i])
    return result

def main():
    #Importing csv files
    df_train_data = pd.read_csv('datasets/2024-02-11_transaction_download.csv')
    df_test_data = pd.read_csv('/Users/kmiyahara/Projects/repo/bank-account-categorization/datasets/transactions_test.csv')


    df_train_data=df_train_data[['Description', 'Category']]
    data_np = df_train_data.to_numpy()

    df_train_data.describe()
    Xtrain, Xtest = train_test_split(data_np, random_state=42)

    cv = CountVectorizer(token_pattern=r'[^\s,][^,]+')
    X_train_counts = cv.fit_transform(Xtrain[:, 0])
    categories = Xtrain[:, 1]
    target = pd.factorize(categories)
    clf = MultinomialNB().fit(X_train_counts, target[0])

    df = pd.DataFrame(zip(predict(Xtest[:, 0], cv, clf, target), Xtest[:, 0]), columns=['predicted_category', 'given_data'])
    

    # Create directory called "result" if it doesn't exist
    if not os.path.exists("result"):
        os.makedirs("result")

    df.to_csv("result/prediction_result.csv", index=False)


if __name__ == "__main__":
    main()