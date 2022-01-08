from sklearn import metrics
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
from matplotlib.ticker import ScalarFormatter


class LemmaTokenizer(object):
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))

    def __call__(self, doc):
        stemmer = PorterStemmer()
        clean = [self.regex.sub('', w) for w in word_tokenize(doc)]
        return [stemmer.stem(t) for t in clean if len(t) > 2]


def learning_curves_for_means(classifier, x, y):
    # train_sizes = np.logspace(np.log10(.1), np.log10(1.0), 20)
    # cv = ShuffleSplit(n_splits=100, test_size=0.2)
    # cv = ShuffleSplit(test_size=0.2)

    train_sizes, train_scores, test_scores = learning_curve(classifier, x, y, n_jobs=-1)
    return test_scores, train_sizes, train_scores


def news_gatherer(percent, low_data, categ):
    if Path(r"cache_20news/cache_X_train.p").exists() and Path(r"cache_20news/cache_y_train.p").exists() and Path(
            r"cache_20news/cache_X_test.p").exists() and Path(r"cache_20news/cache_y_test.p").exists():
        print("-Cache-")
        print("NOo")
        X_train = pickle.load(open("cache_20news/cache_X_train.p", "rb"))
        y_train = pickle.load(open("cache_20news/cache_y_train.p", "rb"))
        X_test = pickle.load(open("cache_20news/cache_X_test.p", "rb"))
        y_test = pickle.load(open("cache_20news/cache_y_test.p", "rb"))
    elif low_data:
        print("low_data = True")
        categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space', ]
        remove = ('headers', 'footers', 'quotes')
        X_train, y_train = fetch_20newsgroups(subset='train', return_X_y=True, remove=remove,
                                              shuffle=True, categories=categories, random_state=25)  # SOLO TRAIN
        X_test, y_test = fetch_20newsgroups(subset='test', return_X_y=True, remove=remove,
                                            shuffle=True, categories=categories, random_state=25)  # SOLO TEST
    elif categ:
        full_cat_list = list(fetch_20newsgroups().target_names)
        categories = full_cat_list[:categ]
        remove = ('headers', 'footers', 'quotes')
        X_train, y_train = fetch_20newsgroups(subset='train', return_X_y=True, remove=remove,
                                              shuffle=True, categories=categories, random_state=25)  # SOLO TRAIN
        X_test, y_test = fetch_20newsgroups(subset='test', return_X_y=True, remove=remove,
                                            shuffle=True, categories=categories, random_state=25)
    else:
        remove = ('headers', 'footers', 'quotes')
        X_train, y_train = fetch_20newsgroups(subset='train', return_X_y=True, remove=remove,
                                              shuffle=True, random_state=25)  # SOLO TRAIN
        X_test, y_test = fetch_20newsgroups(subset='test', return_X_y=True, remove=remove,
                                            shuffle=True, random_state=25)  # SOLO TEST
        print("-No Cache-")
        # pickle.dump(X_train, open("cache_20news/cache_X_train.p", "wb"))
        # pickle.dump(y_train, open("cache_20news/cache_y_train.p", "wb"))
        # pickle.dump(X_test, open("cache_20news/cache_X_test.p", "wb"))
        # pickle.dump(y_test, open("cache_20news/cache_y_test.p", "wb"))

    # FIXME
    if percent:
        print("Percent = True")
        percented_Train = int(req_percentage(len(X_train), percent))
        percented_Test = int(req_percentage(len(X_test), percent))
        print(len(X_train), end="->")
        X_train_percented, y_train_percented, X_test_percented, y_test_percented = X_train[:percented_Train], \
                                                                                   y_train[:percented_Train], \
                                                                                   X_test[:percented_Test], \
                                                                                   y_test[:percented_Test]
        X_train, y_train, X_test, y_test = X_train_percented, y_train_percented, X_test_percented, y_test_percented
        print(len(X_train), end="\n--------------------------\n")
    X_vectorized_train = vectorizer.fit_transform(X_train)
    X_vectorized_test = vectorizer.transform(X_test)
    return X_vectorized_train, y_train, X_vectorized_test, y_test


def shuffler(group1, group2):
    c = list(zip(group1, group2))
    random.shuffle(c)
    return zip(*c)


'''
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
'''


def plot_part(title, train_sizes_bnb, train_sizes_prc, test_bnb, test_prc, train_bnb, train_prc):
    new_title = "Learning Curves on " + title
    plt.figure()
    plt.title(new_title)
    plt.xlabel("Numero di Esempi")
    plt.ylabel("Score")
    plt.grid()
    # plt.plot(train_sizes_bnb, test_bnb, 'o-', color="g", label="Test BernoulliNB")
    # plt.plot(train_sizes_prc, test_prc, 'o-', color="r", label="Test Perceptron")
    plt.plot(train_sizes_bnb, train_bnb, 'x-', color="g", label="Train BernoulliNB")
    plt.plot(train_sizes_prc, train_prc, 'x-', color="r", label="Train Perceptron")
    plt.legend(loc="best")
    plt.savefig("bnb_prc_" + title + ".png", dpi=100)
    plt.show()


def plot_score(clf, score, samples):
    new_title = "Learning Curves on " + clf
    plt.figure()
    plt.title(new_title)
    plt.xlabel("Esempi")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(samples, score, 'x-', color="g", label=clf)
    # plt.savefig("bnb_prc_" + title + ".png", dpi=100)
    plt.show()


def classify(clf, clf_name, X_train, y_train, X_test, y_test):
    scores = []
    steps = []
    step = int(X_test.shape[0] / 10)
    clf.fit(X_train, y_train)
    for i in range(step, X_test.shape[0], step):
        pred = clf.predict(X_test[:i])
        steps.append(i)
        scores.append(metrics.accuracy_score(y_test[:i], pred))
    # plot_score(clf_name, scores, steps)
    return scores[-1]
    # print(clf_name, parameter, score)


def all_20news_group(low_data):
    # bnbplt = plt.figure(1)
    # prcplt = plt.figure(2)
    total_cat = list(fetch_20newsgroups().target_names)
    for categories in range(2, len(total_cat) + 1):
        print("Cat " + str(categories) + "/20")
        scores_prc = []
        scores_bnb = []
        samples = []
        for percent in range(10, 101, 10):
            print(str(percent) + "%")
            X_train, y_train, X_test, y_test = news_gatherer(percent, low_data, categories)
            samples.append(X_train.shape[0])

            scores_bnb.append(
                classify(BernoulliNB(alpha=0.0001), "Bernoullian Naive Bayes", X_train, y_train, X_test, y_test))

            scores_prc.append(classify(Perceptron(max_iter=100), "Perceptron", X_train, y_train, X_test, y_test))

            # plt.savefig("bnb_prc_" + title + ".png", dpi=100)

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        subtitle = "20NewsGroup with " + str(categories) + " categories"
        fig.suptitle(subtitle)
        ax1.plot(samples, scores_bnb, "x-", color='r', label="Bernoullian Naive Bayes")
        ax1.grid()
        ax1.legend()
        ax2.plot(samples, scores_prc, "o-", color='g', label="Perceptron")
        ax2.grid()
        ax2.legend()
        y_formatter = ScalarFormatter(useOffset=False)
        ax1.yaxis.set_major_formatter(y_formatter)
        ax2.yaxis.set_major_formatter(y_formatter)
        plt.show()


def not_all_reut():
    total_cat = list(Reuters.top10categories())
    for categories in range(2, len(total_cat) + 1):
        print("Cat " + str(categories) + "/20")
        scores_prc = []
        scores_bnb = []
        samples = []
        for percent in range(10, 101, 10):
            if percent == 100:
                print("cane")
            print(str(percent) + "%")
            X, y = Reuters.return_X_y_reut_notallcat(percent)

            percented_Data = len(X) - int(req_percentage(len(X), percent))

            # percented_Test = int(req_percentage(len(X_test), percent))
            print(len(X), end="->")
            X_train_percented, y_train_percented, X_test_percented, y_test_percented = X[:percented_Data], \
                                                                                       y[:percented_Data], \
                                                                                       X[percented_Data:], \
                                                                                       y[percented_Data:]
            y_train, y_test = y_train_percented, y_test_percented

            X_train = vectorizer.fit_transform(X_train_percented)
            X_test = vectorizer.transform(X_test_percented)

            print(X_test.shape[0], end="\n--------------------------\n")

            samples.append(X_test.shape[0])

            scores_bnb.append(
                classify(BernoulliNB(alpha=0.0001), "Bernoullian Naive Bayes", X_train, y_train, X_test, y_test))

            scores_prc.append(classify(Perceptron(max_iter=100), "Perceptron", X_train, y_train, X_test, y_test))

            # plt.savefig("bnb_prc_" + title + ".png", dpi=100)

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        subtitle = "Reuter " + str(categories) + " categories"
        fig.suptitle(subtitle)
        ax1.plot(samples, scores_bnb, "x-", color='r', label="Bernoullian Naive Bayes")
        ax1.grid()
        ax1.legend()
        ax2.plot(samples, scores_prc, "o-", color='g', label="Perceptron")
        ax2.grid()
        ax2.legend()

        y_formatter = ScalarFormatter(useOffset=False)
        ax1.yaxis.set_major_formatter(y_formatter)
        ax2.yaxis.set_major_formatter(y_formatter)

        plt.show()


def all_reuters():
    top10cat = Reuters.top10categories()
    scores_bnb = 0
    scores_prc = 0
    train_sizes = 0
    count = 0
    first_run = True

    X_reut, y_reut_noTok = Reuters.return_X_y_reut()
    X_reut_shuffled, y_reut_noTok_shuffled = shuffler(X_reut, y_reut_noTok)
    X_vectorized_reut = vectorizer.fit_transform(X_reut_shuffled)
    y_reut = y_reut_noTok_shuffled
    for cat in top10cat:
        count += 1
        print("In esecuzione: ", " (", count, ")/ 10")

        bnb = BernoulliNB()
        prc = Perceptron(max_iter=200)
        test_score_bnb, train_sizes, train_score_bnb = learning_curves_for_means(bnb, X_vectorized_reut, y_reut)
        test_score_prc, train_sizes1, train_score_prc = learning_curves_for_means(prc, X_vectorized_reut, y_reut)
        if first_run:
            scores_bnb = train_score_bnb
            scores_prc = train_score_prc
            first_run = False
        else:
            scores_bnb = (np.array(scores_bnb) + np.array(train_score_bnb)) / 2
            scores_prc = (np.array(scores_prc) + np.array(train_score_prc)) / 2

    mean_bnb = np.mean(scores_bnb, axis=1)
    mean_prc = np.mean(scores_prc, axis=1)
    plot_part("Reuters-21578", train_sizes, mean_bnb, mean_prc)


def proportion(a):
    # a/b = c/d | 80/100 = x/1234 | x = 80*1234/100
    return 80 * a / 100


def req_percentage(full_number, percentage_you_want):
    return percentage_you_want * full_number / 100


if __name__ == '__main__':
    start = time.time()
    # news_alt()

    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words="english", min_df=3)
    # vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')
    # vectorizer = CountVectorizer(stop_words="english")

    print("Reperimento Datasets: 20News groups")
    #all_20news_group(False)
    print("Reperimento Datasets: Reuters")
    # all_reuters()
    not_all_reut()
    print("Tempo impiegato: ", time.time() - start, " secondi")

# y = target
