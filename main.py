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


def plot_learning_curves(bnb, prc, title, x, y):
    n_jobs = -1
    train_sizes = np.logspace(np.log10(.1), np.log10(1.0), 8)
    cv = ShuffleSplit(n_splits=100, test_size=0.2)

    plt.figure()
    plt.title(title)
    plt.xlabel("Numero di Esempi")

    plt.ylabel("Score")
    train_sizes, train_scores, test_scores_bnb = learning_curve(bnb, x, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_sizes1, train_scores1, test_scores_prc = learning_curve(prc, x, y, cv=cv, n_jobs=n_jobs,
                                                               train_sizes=train_sizes)

    test_scores_mean_bnb = np.mean(test_scores_bnb, axis=1)

    test_scores_mean_prc = np.mean(test_scores_prc, axis=1)

    plt.grid()

    plt.plot(train_sizes, test_scores_mean_bnb, 'o-', color="g", label="BernoulliNB")
    plt.plot(train_sizes1, test_scores_mean_prc, 'o-', color="r", label="Perceptron")
    plt.legend(loc="best")
    return plt


class LemmaTokenizer(object):
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))

    def __call__(self, doc):
        stemmer = PorterStemmer()
        clean = [self.regex.sub('', w) for w in word_tokenize(doc)]
        return [stemmer.stem(t) for t in clean if len(t) > 2]


if __name__ == '__main__':
    remove = ('headers', 'footers', 'quotes')
    print("Reperimento Datasets: 20News groups", end=" ")
    X_20news, y_20news = fetch_20newsgroups(return_X_y=True, shuffle=True, remove=remove)
    print("Completato")

    print("Reperimento Datasets: Reuters", end=" ")
    X_reut, y_reut_noTok = Reuters.return_X_y_reut()
    print("Completato")

    print("Vettorizzazione dei documenti")
    vectorizerMini = CountVectorizer(tokenizer=LemmaTokenizer())
    y_reut = vectorizerMini.fit_transform(y_reut_noTok).indices

    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words="english", min_df=3)
    X_vectorized_20news = vectorizer.fit_transform(X_20news)
    print("20News Completato")
    X_vectorized_reut = vectorizer.fit_transform(X_reut)
    print("Reuters Completato")

    print("Generazione grafici")
    bnb20 = BernoulliNB()
    prc20 = Perceptron()

    Title1 = "Learning Curves on 20newsgroups"
    plot_learning_curves(bnb20, prc20, Title1, X_vectorized_20news, y_20news)
    plt.savefig("bnb_prc_20group.png", dpi=100)
    plt.show()

    bnbre = BernoulliNB()
    prcre = Perceptron()

    Title2 = "Learning Curves on Reuters-21578"
    plot_learning_curves(bnbre, prcre, Title2, X_vectorized_reut, y_reut)
    plt.savefig("bnb_prc_Reuters.png", dpi=100)
    plt.show()
