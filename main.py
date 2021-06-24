# This is a sample Python script.
from os import getcwd

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_20newsgroups, load_files
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import DatasetRetriver
import Reuters


def show_top(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-20:]
        print("\x1b[31m %s \x1b[0m : %s"
              % (category, ", ".join(feature_names[top10])))


def plot_learning_curves(estimator1, estimator2, title, x1, y, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Samples")

    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator1, x1, y, cv=10, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_sizes1, train_scores1, test_scores1 = learning_curve(estimator2, x1, y, cv=10, n_jobs=n_jobs,
                                                               train_sizes=train_sizes)

    # Computes average and standard deviation for the test score and the train score
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    train_scores_mean1 = np.mean(train_scores1, axis=1)
    train_scores_std1 = np.std(train_scores1, axis=1)
    test_scores_mean1 = np.mean(test_scores1, axis=1)
    test_scores_std1 = np.std(test_scores1, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="w")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="w")

    plt.fill_between(train_sizes1, train_scores_mean1 - train_scores_std1,
                     train_scores_mean1 + train_scores_std1, alpha=0.1,
                     color="w")
    plt.fill_between(train_sizes1, test_scores_mean1 - test_scores_std1,
                     test_scores_mean1 + test_scores_std1, alpha=0.1, color="w")
    plt.plot(train_sizes1, test_scores_mean1, 'o-', color="r",
             label="Percy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="BernoulliNB")
    plt.legend(loc="best")
    return plt


class LemmaTokenizer(object):
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))

    def __call__(self, doc):
        stemmer = PorterStemmer()
        clean = [self.regex.sub('', w) for w in word_tokenize(doc)]
        return [stemmer.stem(t) for t in clean if len(t) > 2]


def plot_curve(train_sizes, train_scores, test_scores, title='', ylim=None):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("F1 weighted score")
    plt.grid(True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training set")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation set")
    plt.legend(loc="best")


def report(target, pred, target_names):
    cm = metrics.confusion_matrix(target, pred)
    print('Contingency matrix: (rows: examples of a given true class)')
    print(cm)
    print('Classification report')
    print(metrics.classification_report(target, pred, target_names=target_names))


if __name__ == '__main__':
    remove = ('headers', 'footers', 'quotes')
    print("Reperimento Datasets: 20News groups", end=" ")
    train_20news = fetch_20newsgroups(subset='train', shuffle=True, remove=remove)  # , categories=categories)
    test_20news = fetch_20newsgroups(subset='test', shuffle=True, remove=remove)  # , categories=categories)
    print("Completato")
    print("Reperimento Datasets: Reuters", end=" ")
    print("E chi ne ha voglia? Io? Ahahaha ti sbagli!!", end="...")
    Reuters.warmup()
    # X_train_reuters, X_test_reuters, y_train_reuters, y_test_reuters = Reuters.giveXyreuters()
    import os

    # cwd = os.getcwd()

    X_train_reuters = load_files(container_path='./data-set/training-set', shuffle=True, random_state=42)
    X_test_reuters = load_files(container_path='./data-set/test-set', shuffle=True, random_state=42)
    # Reuters
    print("Completato")
    '''
    X, y = fetch_20newsgroups(return_X_y=True, shuffle=True, remove=remove)
    bnb = BernoulliNB()
    prc = Perceptron()
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words="english", min_df=3)#, binary=True)
    Vect_X = vectorizer.fit_transform(X)
    TR_X, TE_X = train_test_split(Vect_X, shuffle=False)
    TR_y, TE_y = train_test_split(y, shuffle=False)
    bnb.fit(TR_X, TR_y)
    prc.fit(TR_X, TR_y)
    predict = bnb.predict(TE_X)
    predictPRC = prc.predict(TE_X)
    report(TE_y, predict, train_20news.target_names)
    report(TE_y, predictPRC, train_20news.target_names)

    print(train_20news.target_names)
    print(test_20news.target_names)
    n = train_20news.target.shape[0]
    print(n, "training documents")
    '''
    print("Vettorizzazione dei documenti")
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words="english", min_df=3)  # , binary=True)
    TR_vectorized_20News = vectorizer.fit_transform(train_20news.data)
    TE_vectorized_20News = vectorizer.transform(test_20news.data)
    print("20News Completato")
    TR_vectorized_reuters = vectorizer.fit_transform(X_train_reuters.data)
    TE_vectorized_reuters = vectorizer.transform(X_test_reuters.data)
    print("Reuters Completato")
    bnb20 = BernoulliNB()
    prc20 = Perceptron()

    print("Addestramento (fitting) delle reti per 20News groups:")
    print("Bernoullian Naive Bayes", end=" ")
    bnb20.fit(TR_vectorized_20News, train_20news.target)
    print("Completato")
    print("Perceptron", end=" ")
    prc20.fit(TR_vectorized_20News, train_20news.target)
    print("Completato")

    bnbre = BernoulliNB()
    prcre = Perceptron()
    print("Addestramento (fitting) delle reti per Reuters:")
    print("Bernoullian Naive Bayes", end=" ")
    bnbre.fit(TR_vectorized_reuters, X_train_reuters.target)
    print("Completato")
    print("Perceptron", end=" ")
    prcre.fit(TR_vectorized_reuters, X_train_reuters.target)
    print("Completato")

    predict_bnb20 = bnb20.predict(TE_vectorized_20News)
    predict_prc20 = prc20.predict(TE_vectorized_20News)
    print("Predizioni fatte per 20News group")
    predict_bnbreut = bnbre.predict(TE_vectorized_reuters)
    predict_prcreut = prcre.predict(TE_vectorized_reuters)

    print("Report Bernoullian Naive Bayes")
    print("20News groups: ")
    report(test_20news.target, predict_bnb20, train_20news.target_names)
    print("Reuters")
    report(X_test_reuters.target, predict_bnbreut, X_train_reuters.target_names)

    # report

    print("Report Perceptron")
    print("20News groups: ")
    report(test_20news.target, predict_prc20, train_20news.target_names)
    print("Reuters")
    report(X_test_reuters.target, predict_prcreut, X_test_reuters.target_names)

    # report

    print("Generazione grafici")
    cv = ShuffleSplit(n_splits=100, test_size=0.2)  # , random_state=0)
    train_sizes = np.logspace(np.log10(.1), np.log10(1.0), 8)
    '''
    Title1 = "Learning Curves on 20newsgroups"
    Title2 = "Learning Curves on Reuters-21578"

    plot_learning_curves(bnb, prc, Title1, TR_vectorized_20News, train_20news.target,
                                    n_jobs=1,
                                    train_sizes=([0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.]))
    plt.show()
    plt.savefig('grafico1.png', bbox_inches='tight')

    plot_learning_curves(bnb, prc, Title2, TR_vectorized_reut,
                                    X_reuters.target, n_jobs=1,
                                    train_sizes=([0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.]))
    plt.show()
    plt.savefig('grafico2.png', bbox_inches='tight')
    '''
    train_sizes, train_scores, test_scores = learning_curve(bnb20, TR_vectorized_20News, train_20news.target, cv=cv,
                                                            scoring="f1_weighted", train_sizes=train_sizes, n_jobs=-1)
    plot_curve(train_sizes, train_scores, test_scores, title="Bernoulli Naive Bayes 20Group")

    train_sizes, train_scores, test_scores = learning_curve(prc20, TR_vectorized_20News, train_20news.target, cv=cv,
                                                            scoring="f1_weighted", train_sizes=train_sizes, n_jobs=-1)
    plot_curve(train_sizes, train_scores, test_scores, title="Perceptron 20Group")
##############################################################################################
    train_sizes, train_scores, test_scores = learning_curve(bnbre, TR_vectorized_reuters, X_train_reuters.target, cv=cv,
                                                            scoring="f1_weighted", train_sizes=train_sizes, n_jobs=-1)
    plot_curve(train_sizes, train_scores, test_scores, title="Bernoulli Naive Bayes reuters")

    train_sizes, train_scores, test_scores = learning_curve(prcre, TR_vectorized_reuters, X_train_reuters.target, cv=cv,
                                                            scoring="f1_weighted", train_sizes=train_sizes, n_jobs=-1)
    plot_curve(train_sizes, train_scores, test_scores, title="Perceptron reuters")
    plt.show()
    plt.savefig('grafici.png')  # , bbox_inches='tight')

    print("Fine")

    '''
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    title = "Learning Curves (Naive Bayes)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = bnb
    X = TR_vectorized_20News.append(TE_vectorized_20News)
    y = train_20news.target.append(test_20news.target)
    plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
                        cv=cv, n_jobs=-1)

    title = r"Learning Curves (Perceptron)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = prc
    plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
                        cv=cv, n_jobs=-1)

    plt.show()
    '''
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# Bernoullian naive bayes ->
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB

# Perceptron
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron
