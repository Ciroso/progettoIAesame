# This is a sample Python script.
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_20newsgroups
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model -" + title)

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model -" + title)
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
    #Reuters
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
    print("Vettorizzazione dei documenti", end=" ")
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words="english", min_df=3)#, binary=True)
    TR_vectorized_20News = vectorizer.fit_transform(train_20news.data)
    TE_vectorized_20News = vectorizer.transform(test_20news.data)
    print("Completata")
    bnb = BernoulliNB()
    prc = Perceptron()

    print("Addestramento (fitting) delle reti per 20News groups:")
    print("Bernoullian Naive Bayes", end=" ")
    bnb.fit(TR_vectorized_20News, train_20news.target)
    print("Completato")
    print("Perceptron", end=" ")
    prc.fit(TR_vectorized_20News, train_20news.target)
    print("Completato")

    '''
    print("Addestramento (fitting) delle reti per Reuters:")
    print("Bernoullian Naive Bayes", end=" ")
    bnb.fit(TR_vectorized_20News, train_20news.target)
    print("Completato")
    print("Perceptron", end=" ")
    prc.fit(TR_vectorized_20News, train_20news.target)
    print("Completato")
    '''
    predict = bnb.predict(TE_vectorized_20News)
    predictPRC = prc.predict(TE_vectorized_20News)

    print("Predizioni fatte")
    print("Report Bernoullian Naive Bayes")
    print("20News groups: ")
    report(test_20news.target, predict, train_20news.target_names)
    print("Reuters")
    #report

    print("Report Perceptron")
    print("20News groups: ")
    report(test_20news.target, predictPRC, train_20news.target_names)
    print("Reuters")
    #report

    print("Generazione grafici")
    cv = ShuffleSplit(n_splits=100, test_size=0.2)#, random_state=0)
    train_sizes = np.logspace(np.log10(.1), np.log10(1.0), 8)
    train_sizes, train_scores, test_scores = learning_curve(bnb, TR_vectorized_20News, train_20news.target, cv=cv,
                                                            scoring="f1_weighted", train_sizes=train_sizes, n_jobs=-1)
    plot_curve(train_sizes, train_scores, test_scores, title="Bernoulli Naive Bayes")

    train_sizes, train_scores, test_scores = learning_curve(prc, TR_vectorized_20News, train_20news.target, cv=cv,
                                                            scoring="f1_weighted", train_sizes=train_sizes, n_jobs=-1)
    plot_curve(train_sizes, train_scores, test_scores, title="Perceptron")
    plt.show()

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
