import random
import time

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
import Reuters
from sklearn.model_selection import train_test_split


def plot_learning_curves(bernoullian_nb, perceptron, title, x, y):
    n_jobs = -1
    train_sizes = np.logspace(np.log10(.1), np.log10(1.0), 8)
    cv = ShuffleSplit(n_splits=100, test_size=0.2)

    plt.figure()
    plt.title(title)
    plt.xlabel("Numero di Esempi")

    plt.ylabel("Score")
    train_sizes, train_scores, test_scores_bnb = learning_curve(bernoullian_nb, x, y, cv=cv, n_jobs=n_jobs,
                                                                train_sizes=train_sizes)
    train_sizes1, train_scores1, test_scores_prc = learning_curve(perceptron, x, y, cv=cv, n_jobs=n_jobs,
                                                                  train_sizes=train_sizes)

    test_scores_mean_bnb = np.mean(test_scores_bnb, axis=1)

    test_scores_mean_prc = np.mean(test_scores_prc, axis=1)

    plt.grid()

    plt.plot(train_sizes, test_scores_mean_bnb, 'o-', color="g", label="BernoulliNB")
    plt.plot(train_sizes1, test_scores_mean_prc, 'o-', color="r", label="Perceptron")
    plt.legend(loc="best")
    return plt


def plot_learning_curves_bnb(bernoullian_nb, title, x, y):
    n_jobs = -1
    train_sizes = np.logspace(np.log10(.1), np.log10(1.0), 8)
    cv = ShuffleSplit(n_splits=100, test_size=0.2)

    plt.figure()
    plt.title(title)
    plt.xlabel("Numero di Esempi")

    plt.ylabel("Score")
    train_sizes, train_scores, test_scores_bnb = learning_curve(bernoullian_nb, x, y, cv=cv, n_jobs=n_jobs,
                                                                train_sizes=train_sizes)

    test_scores_mean_bnb = np.mean(test_scores_bnb, axis=1)

    plt.grid()

    plt.plot(train_sizes, test_scores_mean_bnb, 'o-', color="g", label="BernoulliNB")
    plt.legend(loc="best")
    return plt


def learning_curves_for_means(classifier, x, y):
    n_jobs = -1
    train_sizes = np.logspace(np.log10(.1), np.log10(1.0), 8)
    cv = ShuffleSplit(n_splits=100, test_size=0.2)

    # plt.figure()
    # plt.title(title)
    # plt.xlabel("Numero di Esempi")

    # plt.ylabel("Score")
    # train_sizes, train_scores, test_scores_bnb = learning_curve(bernoullian_nb, x, y, cv=cv, n_jobs=n_jobs,
    # train_sizes=train_sizes)
    train_sizes1, train_scores1, test_scores_prc = learning_curve(classifier, x, y, cv=cv, n_jobs=n_jobs,
                                                                  train_sizes=train_sizes)

    # return test_scores_prc, train_sizes, train_sizes1
    return test_scores_prc, train_sizes1


class LemmaTokenizer(object):
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))

    def __call__(self, doc):
        stemmer = PorterStemmer()
        clean = [self.regex.sub('', w) for w in word_tokenize(doc)]
        return [stemmer.stem(t) for t in clean if len(t) > 2]


if __name__ == '__main__':
    start = time.time()
    remove = ('headers', 'footers', 'quotes')

    #  FIXME
    vectorizerMini = CountVectorizer(tokenizer=LemmaTokenizer())
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words="english", min_df=3)
    '''
    print("Reperimento Datasets: 20News groups", end=" ")
    X_20news, y_20news = fetch_20newsgroups(return_X_y=True, remove=remove)
    print("Completato")

    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words="english", min_df=3)

    print("Vettorizzazione dei documenti")
    X_vectorized_20news = vectorizer.fit_transform(X_20news)
    print("20News Completato")

    print("Generazione grafici")
    bnb20 = BernoulliNB()
    prc20 = Perceptron()

    Title1 = "Learning Curves on 20newsgroups"
    plot_learning_curves(bnb20, prc20, Title1, X_vectorized_20news, y_20news)
    plt.savefig("bnb_prc_20group.png", dpi=100)
    plt.show()
    '''
    # ------------ FINE 20NEWS GROUP ------------
    # -------------- INIZIO REUTERS --------------

    Title2 = "Learning Curves on Reuters-21578"

    print("Reperimento Datasets: Reuters", end=" ")

    top10cat = Reuters.top10categories(Reuters.file_list())
    scores_bnb = 0
    scores_prc = 0
    train_sizes = 0
    train_sizes1 = 0
    count = 0
    first_run = True
    historic_score_bnb = []
    historic_score_prc = []
    for cat in top10cat:
        count += 1
        print("In esecuzione: ", cat, " (", count, ")/ 10")
        X_reut, y_reut_noTok = Reuters.return_X_y_reut(cat)

        # X_reut_shuffled, y_reut_noTok_shuffled = shuffle_in_unison(X_reut, y_reut_noTok)

        c = list(zip(X_reut, y_reut_noTok))

        random.shuffle(c)

        X_reut_shuffled, y_reut_noTok_shuffled = zip(*c)

        X_vectorized_reut = vectorizer.fit_transform(X_reut_shuffled)
        y_reut = vectorizerMini.fit_transform(y_reut_noTok_shuffled).indices

        # -----------------------------------------------------------------------------------------------------
        # shuffled_X, shuffled_y = shuffle_in_unison(X_vectorized_reut, y_reut)
        # CUT by scikit
        # X_train, X_test, y_train, y_test = train_test_split(shuffled_X, shuffled_y, test_size=0.2)
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized_reut, y_reut, test_size=0.2)
        clf1 = BernoulliNB()
        clf2 = Perceptron()

        clf1.fit(X_train, y_train)
        clf2.fit(X_train, y_train)
        y_predict1 = clf1.predict(X_test)
        y_predict2 = clf2.predict(X_test)
        score1 = clf1.score(X_test, y_test)
        score2 = clf2.score(X_test, y_test)
        print("Score BernoulliNB: ", score1)
        print("Score2 Perceptron: ", score2)
        historic_score_bnb.append(score1)
        historic_score_prc.append(score2)
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        # print(classification_report(y_test, y_predict1))
        # print("-----------------------------------------------------------------------------------------------------")
        # print(classification_report(y_test, y_predict2))

        # -----------------------------------------------------------------------------------------------------

        score_bnb, train_sizes = learning_curves_for_means(BernoulliNB(), X_vectorized_reut, y_reut)  # alpha=0.01
        score_prc, train_sizes1 = learning_curves_for_means(Perceptron(), X_vectorized_reut, y_reut)  # max_iter=50
        if first_run:
            scores_bnb = score_bnb
            scores_prc = score_prc
            first_run = False
        else:
            scores_bnb = (np.array(scores_bnb) + np.array(score_bnb)) / 2
            scores_prc = (np.array(scores_prc) + np.array(score_prc)) / 2

            #  ------------------------------------------------------------------------
        '''
        plt.figure()
        plt.title(Title2 + " - " + cat)
        plt.xlabel("Numero di Esempi")

        plt.ylabel("Score")
        mean_bnb = np.mean(scores_bnb, axis=1)
        mean_prc = np.mean(scores_prc, axis=1)

        mean_bnbs = np.mean(score_bnb, axis=1)
        mean_prcs = np.mean(score_prc, axis=1)

        plt.grid()

        plt.plot(train_sizes, mean_bnb, 'o-', color="g", label="BernoulliNB")
        plt.plot(train_sizes1, mean_prc, 'o-', color="r", label="Perceptron")

        plt.plot(train_sizes, mean_bnbs, 'o-', color="b", label="BernoulliNB - Single")
        plt.plot(train_sizes1, mean_prcs, 'o-', color="y", label="Perceptron - Single")
        plt.legend(loc="best")
        # return plt

        plt.savefig("bnb_prc_Reuters.png", dpi=100)
        plt.show()
    '''
        # scores_bnb.append(score_bnb)
        # scores_prc.append(scores_prc)
    '''
    X_reut, y_reut_noTok = Reuters.return_X_y_reut_bnb()
    vectorizerMini = CountVectorizer(tokenizer=LemmaTokenizer())
    y_reut = vectorizerMini.fit_transform(y_reut_noTok).indices
    X_vectorized_reut = vectorizer.fit_transform(X_reut)
    scores_bnb, train_sizes = learning_curves_for_means(BernoulliNB(), X_vectorized_reut, y_reut)
    '''
    mean_bnb = np.mean(scores_bnb, axis=1)
    mean_prc = np.mean(scores_prc, axis=1)

    plt.figure()
    plt.title(Title2)
    plt.xlabel("Numero di Esempi")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(train_sizes, mean_bnb, 'o-', color="g", label="BernoulliNB")
    plt.plot(train_sizes1, mean_prc, 'o-', color="r", label="Perceptron")
    plt.legend(loc="best")
    # return plt

    plt.savefig("bnb_prc_Reuters.png", dpi=100)
    plt.show()
    # ----------------------------------------

    plt.figure()
    plt.ylabel("Score")
    plt.xlabel("Categorie")
    plt.plot(top10cat, historic_score_bnb, 'o-', color="g", label="BernoulliNB")
    plt.plot(top10cat, historic_score_prc, 'o-', color="r", label="Perceptron")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("bnb_prc_Reuters_score.png", dpi=100)
    plt.show()

    print("Elapsed time: ", time.time() - start)
