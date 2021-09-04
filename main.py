from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from pathlib import Path
import re
import time
import pickle
import random
import string
import Reuters
import numpy as np
import matplotlib.pyplot as plt


class LemmaTokenizer(object):
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))

    def __call__(self, doc):
        stemmer = PorterStemmer()
        clean = [self.regex.sub('', w) for w in word_tokenize(doc)]
        return [stemmer.stem(t) for t in clean if len(t) > 2]


def learning_curves_for_means(classifier, x, y):
    train_sizes = np.logspace(np.log10(.1), np.log10(1.0), 8)
    cv = ShuffleSplit(n_splits=100, test_size=0.2)
    train_sizes, train_scores, test_scores = learning_curve(classifier, x, y, train_sizes=train_sizes, cv=cv, n_jobs=-1)
    return test_scores, train_sizes


def shuffler(group1, group2):
    c = list(zip(group1, group2))
    random.shuffle(c)
    return zip(*c)


def binarize(y, category_number):
    bintime = time.time()
    new_y = []
    count = 0
    for i in range(0, len(y)):
        if y[i] == category_number:
            new_y.append(0)
            count += 1
        else:
            new_y.append(1)
    print("category number ", category_number + 1, ", count ", count, ", tempo ", time.time() - bintime)
    return new_y


def plot_part(title, train_sizes, mean_bnb, mean_prc):
    new_title = "Learning Curves on " + title
    plt.figure()
    plt.title(new_title)
    plt.xlabel("Numero di Esempi")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(train_sizes, mean_bnb, 'o-', color="g", label="BernoulliNB")
    plt.plot(train_sizes, mean_prc, 'o-', color="r", label="Perceptron")
    plt.legend(loc="best")
    plt.savefig("bnb_prc_" + title + ".png", dpi=100)
    plt.show()


def news_gatherer():
    if Path(r"cache_20news/cache_nonbinary_X.p").exists() and Path(r"cache_20news/cache_nonbinary_y.p").exists():
        X = pickle.load(open("cache_20news/cache_nonbinary_X.p", "rb"))
        y = pickle.load(open("cache_20news/cache_nonbinary_y.p", "rb"))
        return X, y
    else:
        remove = ('headers', 'footers', 'quotes')
        X_20news, y_20news = fetch_20newsgroups(return_X_y=True, remove=remove)
        pickle.dump(X_20news, open("cache_20news/cache_nonbinary_X.p", "wb"))
        pickle.dump(y_20news, open("cache_20news/cache_nonbinary_y.p", "wb"))
        return X_20news, y_20news


def all_20news_group():
    X_20news, y_20news = news_gatherer()
    first_run = True
    scores_bnb = 0
    scores_prc = 0
    train_sizes = 0
    count = 0
    for cat in range(0, 20):
        count += 1
        print("In esecuzione: (", count, ")/ 20", end=" ")
        y_binarized_20news = binarize(y_20news, cat)

        X_shuffled_20news, y_shuffled_20news = shuffler(X_20news, y_binarized_20news)
        X_vectorized_20news = vectorizer.fit_transform(X_shuffled_20news)

        bnb = BernoulliNB()
        prc = Perceptron()
        score_bnb, train_sizes = learning_curves_for_means(bnb, X_vectorized_20news, y_shuffled_20news)
        score_prc, train_sizes1 = learning_curves_for_means(prc, X_vectorized_20news, y_shuffled_20news)
        if first_run:
            scores_bnb = score_bnb
            scores_prc = score_prc
            first_run = False
        else:
            scores_bnb = (np.array(scores_bnb) + np.array(score_bnb)) / 2
            scores_prc = (np.array(scores_prc) + np.array(score_prc)) / 2
    mean_bnb = np.mean(scores_bnb, axis=1)
    mean_prc = np.mean(scores_prc, axis=1)

    plot_part("20newsgroups", train_sizes, mean_bnb, mean_prc)


def all_reuters():
    top10cat = Reuters.top10categories()
    scores_bnb = 0
    scores_prc = 0
    train_sizes = 0
    count = 0
    first_run = True
    for cat in top10cat:
        count += 1
        print("In esecuzione: ", cat, " (", count, ")/ 10")

        X_reut, y_reut_noTok = Reuters.return_X_y_reut(cat)
        X_reut_shuffled, y_reut_noTok_shuffled = shuffler(X_reut, y_reut_noTok)

        X_vectorized_reut = vectorizer.fit_transform(X_reut_shuffled)
        y_reut = y_reut_noTok_shuffled

        bnb = BernoulliNB()
        prc = Perceptron()
        score_bnb, train_sizes = learning_curves_for_means(bnb, X_vectorized_reut, y_reut)
        score_prc, train_sizes1 = learning_curves_for_means(prc, X_vectorized_reut, y_reut)
        if first_run:
            scores_bnb = score_bnb
            scores_prc = score_prc
            first_run = False
        else:
            scores_bnb = (np.array(scores_bnb) + np.array(score_bnb)) / 2
            scores_prc = (np.array(scores_prc) + np.array(score_prc)) / 2

    mean_bnb = np.mean(scores_bnb, axis=1)
    mean_prc = np.mean(scores_prc, axis=1)
    plot_part("Reuters-21578", train_sizes, mean_bnb, mean_prc)


if __name__ == '__main__':
    start = time.time()
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words="english", min_df=3)

    print("Reperimento Datasets: 20News groups")
    all_20news_group()

    print("Reperimento Datasets: Reuters")
    all_reuters()

    print("Elapsed time: ", (time.time() - start)/60, " minutes")
