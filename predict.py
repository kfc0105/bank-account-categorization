import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def main():
    #Importing csv
    df_train_data = pd.read_csv('../datasets/2024-02-11_transaction_download.csv')
    df_test_data = pd.read_csv('../datasets/transactions_test.csv')
    df_train_data=df_train_data[['Description', 'Category']]
    data_np = df_train_data.to_numpy()

    df_train_data.describe()

    Xtrain, Xtest = train_test_split(data_np, random_state=42)

    cv = CountVectorizer(token_pattern=r'[^\s,][^,]+')
    X_train_counts = cv.fit_transform(Xtrain[:, 0])
    categories = Xtrain[:, 1]
    target = pd.factorize(categories)
    clf = MultinomialNB().fit(X_train_counts, target[0])

    df = pd.DataFrame(zip(predict(Xtest[:, 0]), Xtest[:, 0]), columns=['predicted_category', 'given_data'])
if __name__ == "__main__":
    main()

def predict(data):
    #docs_new has to be one dimentional list
    docs_new = data
    X_new_counts = cv.transform(docs_new)
    predicted = clf.predict(X_new_counts)
    result = []
    for i in predicted:
        result.append(target[1][i])
    return result