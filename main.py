from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_20newsgroups, load_files
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import Reuters


def plot_learning_curves(estimator1, estimator2, title, x, y):
    n_jobs = -1
    train_sizes = np.linspace(.1, 1.0, 10)
    # train_sizes = np.logspace(np.log10(.1), np.log10(1.0), 8)
    cv = ShuffleSplit(n_splits=100, test_size=0.2)

    plt.figure()
    plt.title(title)
    plt.xlabel("Numero di Esempi")

    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator1, x, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_sizes1, train_scores1, test_scores1 = learning_curve(estimator2, x, y, cv=cv, n_jobs=n_jobs,
                                                               train_sizes=train_sizes)

    # Calcolo la media degli attributi dello score
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    train_scores_mean1 = np.mean(train_scores1, axis=1)
    train_scores_std1 = np.std(train_scores1, axis=1)
    test_scores_mean1 = np.mean(test_scores1, axis=1)
    test_scores_std1 = np.std(test_scores1, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="w")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color="w")

    plt.fill_between(train_sizes1, train_scores_mean1 - train_scores_std1, train_scores_mean1 + train_scores_std1,
                     color="w")
    plt.fill_between(train_sizes1, test_scores_mean1 - test_scores_std1, test_scores_mean1 + test_scores_std1,
                     color="w")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="BernoulliNB")
    plt.plot(train_sizes1, test_scores_mean1, 'o-', color="r", label="Perceptron")
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
    # train_20news = fetch_20newsgroups(subset='train', shuffle=True, remove=remove)  # , categories=categories)
    # test_20news = fetch_20newsgroups(subset='test', shuffle=True, remove=remove)  # , categories=categories)
    X_20news, y_20news = fetch_20newsgroups(return_X_y=True, shuffle=True, remove=remove)
    print("Completato")

    print("Reperimento Datasets: Reuters", end=" ")
    #Reuters.warmup()
    #X_train_reuters = load_files(container_path='data-set/training-set', shuffle=True, random_state=42)
    #X_test_reuters = load_files(container_path='data-set/test-set', shuffle=True, random_state=42)
    X_reut, y_reut = Reuters.return_X_y_reut()
    print("Completato")

    print("Vettorizzazione dei documenti")
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words="english", min_df=3)  # , binary=True)
    #X_vectorized_20news = vectorizer.fit_transform(X_20news)
    # TR_vectorized_20News = vectorizer.fit_transform(train_20news.data)
    # TE_vectorized_20News = vectorizer.transform(test_20news.data)
    print("20News Completato")
    X_vectorized_reut = vectorizer.fit_transform(X_reut)
    # TR_vectorized_reuters = vectorizer.fit_transform(X_train_reuters.data)
    # TE_vectorized_reuters = vectorizer.transform(X_test_reuters.data)
    print("Reuters Completato")

    bnb20 = BernoulliNB()
    prc20 = Perceptron()
    print("Addestramento (fitting) delle reti per 20News groups:")
    print("Bernoullian Naive Bayes", end=" ")
    # bnb20.fit(TR_vectorized_20News, train_20news.target)
    print("Completato")
    print("Perceptron", end=" ")
    # prc20.fit(TR_vectorized_20News, train_20news.target)
    print("Completato")

    bnbre = BernoulliNB()
    prcre = Perceptron()
    print("Addestramento (fitting) delle reti per Reuters:")
    print("Bernoullian Naive Bayes", end=" ")
    # bnbre.fit(TR_vectorized_reuters, X_train_reuters.target)
    print("Completato")
    print("Perceptron", end=" ")
    # prcre.fit(TR_vectorized_reuters, X_train_reuters.target)
    print("Completato")

    print("Calcolo delle predizioni per 20News group")
    # predict_bnb20 = bnb20.predict(TE_vectorized_20News)
    # predict_prc20 = prc20.predict(TE_vectorized_20News)
    print("Predizioni fatte per 20News group")
    print("Calcolo delle predizioni per Reuters")
    # predict_bnbreut = bnbre.predict(TE_vectorized_reuters)
    # predict_prcreut = prcre.predict(TE_vectorized_reuters)
    print("Predizioni fatte per Reuters")

    print("Report Bernoullian Naive Bayes")
    print("20News groups: ")
    # report(test_20news.target, predict_bnb20, train_20news.target_names)
    print("Reuters: ")
    # report(X_test_reuters.target, predict_bnbreut, X_train_reuters.target_names)

    # report

    print("Report Perceptron")
    print("20News groups: ")
    # report(test_20news.target, predict_prc20, train_20news.target_names)
    print("Reuters: ")
    # report(X_test_reuters.target, predict_prcreut, X_test_reuters.target_names)

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
    '''
    train_sizes, train_scores, test_scores = learning_curve(bnb20, TR_vectorized_20News, train_20news.target, cv=cv,
                                                            scoring="f1_weighted", train_sizes=train_sizes, n_jobs=-1)
    plot_curve(train_sizes, train_scores, test_scores, title="Bernoulli Naive Bayes 20Group")
    plt.show()

    train_sizes = np.logspace(np.log10(.1), np.log10(1.0), 8)
    train_sizes, train_scores, test_scores = learning_curve(prc20, TR_vectorized_20News, train_20news.target, cv=cv,
                                                            scoring="f1_weighted", train_sizes=train_sizes, n_jobs=-1)
    plot_curve(train_sizes, train_scores, test_scores, title="Perceptron 20Group")
    plt.show()

##############################################################################################
    train_sizes = np.logspace(np.log10(.1), np.log10(1.0), 8)
    train_sizes, train_scores, test_scores = learning_curve(bnbre, TR_vectorized_reuters, X_train_reuters.target, cv=cv,
                                                            scoring="f1_weighted", train_sizes=train_sizes, n_jobs=-1)
    plot_curve(train_sizes, train_scores, test_scores, title="Bernoulli Naive Bayes reuters")
    plt.show()

    train_sizes = np.logspace(np.log10(.1), np.log10(1.0), 8)
    train_sizes, train_scores, test_scores = learning_curve(prcre, TR_vectorized_reuters, X_train_reuters.target, cv=cv,
                                                            scoring="f1_weighted", train_sizes=train_sizes, n_jobs=-1)
    plot_curve(train_sizes, train_scores, test_scores, title="Perceptron reuters")
    plt.show()
    plt.savefig('grafici.png')  # , bbox_inches='tight')

    print("Fine")
    '''
    Title1 = "Learning Curves on 20newsgroups"
    Title2 = "Learning Curves on Reuters-21578"

    # plot_learning_curves(bnb20, prc20, Title1, TR_vectorized_20News, train_20news.target)
    #plot_learning_curves(bnb20, prc20, Title1, X_vectorized_20news, y_20news)
    #plt.savefig("bnb_prc_20group.png", dpi=100)
    #plt.show()
    #plt.savefig("primo.jpg")
    # plot_learning_curves(bnbre, prcre, Title2, TR_vectorized_reuters, X_train_reuters.target)
    #X_vectorized_reut = TR_vectorized_reuters
    #y_reut = X_train_reuters.target
    plot_learning_curves(bnbre, prcre, Title2, X_vectorized_reut, y_reut)
    plt.savefig("bnb_prc_Reuters.png",dpi = 100)
    plt.show()

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
