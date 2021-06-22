# This is a sample Python script.
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_20newsgroups
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    train_20news = fetch_20newsgroups(subset='train', shuffle=True)
    test_20news = fetch_20newsgroups(subset='test', shuffle=True)
    '''
    vectorizer = CountVectorizer(stop_words='english')#, binary=True)
    TR_vectorized_20News = vectorizer.fit_transform(train_20news.data)
    TE_vectorized_20News = vectorizer.fit_transform(test_20news.data)
    #print(TE_vectorized_20News.data)
    '''
    vectorizer = TfidfVectorizer()
    TR_vectorized_20News = vectorizer.fit_transform(train_20news.data)
    TE_vectorized_20News = vectorizer.transform(test_20news.data)
    bnb = BernoulliNB()
    prc = Perceptron()
    print(train_20news.target_names)
    print(test_20news.target_names)
    bnb.fit(TR_vectorized_20News, train_20news.target)
    prc.fit(TR_vectorized_20News, train_20news.target)
    predict = bnb.predict(TE_vectorized_20News)
    predictPRC = prc.predict(TE_vectorized_20News)
    print(test_20news.target)
    print(predict)
    print(predictPRC)
    accuracy = metrics.accuracy_score(test_20news.target, predict)
    print("Accuracy:", accuracy)
    print("Accuracy:", metrics.accuracy_score(test_20news.target, predictPRC))

    print("f1:", metrics.f1_score(test_20news.target, predict, average='weighted'))
    print("f1:", metrics.f1_score(test_20news.target, predictPRC, average='weighted'))

    print("LOL")

    '''
    #tfidf_transformer = TfidfTransformer()
    train_sizes = ([0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.])
    train_sizes, train_scores, test_scores = learning_curve(bnb, TR_vectorized_20News, train_20news.target, cv=10, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure()
    plt.title("Learning Curves on 20newsgroups")
    plt.xlabel("Samples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="w")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="w")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="BernoulliNB")
    plt.legend(loc="best")
    plt.show()

    #X_train_tfidf_20news = tfidf_transformer.fit_transform(TR_vectorized_20News)
    ##X_train_tfidf_reuters = tfidf_transformer.fit_transform(Xmul_reuters)

    #print(X_train_tfidf_20news.data)
    #bnb.fit(train_20news, test_20news)
    #print(bnb.score(train_20news, test_20news))

    #pcp = Perceptron()

    print("Fine")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# Bernoullian naive bayes ->
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB

# Perceptron
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron
    '''