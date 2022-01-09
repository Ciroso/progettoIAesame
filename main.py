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
import matplotlib.pyplot as plt


class LemmaTokenizer(object):
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))

    def __call__(self, doc):
        stemmer = PorterStemmer()
        clean = [self.regex.sub('', w) for w in word_tokenize(doc)]
        return [stemmer.stem(t) for t in clean if len(t) > 2]


def news_gatherer_cat_counter(categ):
    remove = ('headers', 'footers', 'quotes')
    full_cat_list = list(fetch_20newsgroups().target_names)
    categories = full_cat_list[:categ]
    X_train, y_train = fetch_20newsgroups(subset='train', return_X_y=True, remove=remove,
                                          shuffle=True, categories=categories)
    X_test, y_test = fetch_20newsgroups(subset='test', return_X_y=True, remove=remove,
                                        shuffle=True, categories=categories)
    X_vectorized_train = vectorizer.fit_transform(X_train)
    X_vectorized_test = vectorizer.transform(X_test)
    return X_vectorized_train, y_train, X_vectorized_test, y_test


def news_gatherer(category):
    if Path(r"cache_20news/cache_X_train_" + category + ".p").exists() and Path(
            r"cache_20news/cache_y_train_" + category + ".p").exists() and Path(
        r"cache_20news/cache_X_test_" + category + ".p").exists() and Path(
        r"cache_20news/cache_y_test_" + category + ".p").exists():
        print("CACHE")
        X_vectorized_train = pickle.load(open("cache_20news/cache_X_train_" + category + ".p", "rb"))
        y_train = pickle.load(open("cache_20news/cache_y_train_" + category + ".p", "rb"))
        X_vectorized_test = pickle.load(open("cache_20news/cache_X_test_" + category + ".p", "rb"))
        y_test = pickle.load(open("cache_20news/cache_y_test_" + category + ".p", "rb"))

    else:
        print("NO CACHE")
        remove = ('headers', 'footers', 'quotes')
        rest_cat_list = list(fetch_20newsgroups().target_names)
        rest_cat_list.remove(category)
        selected_category = [category]
        X_one_cat, y_one_cat = fetch_20newsgroups(return_X_y=True, remove=remove, shuffle=True,
                                                  categories=selected_category,
                                                  random_state=42)
        X_other_cat, y_other_cat = fetch_20newsgroups(return_X_y=True, remove=remove, shuffle=True,
                                                      categories=rest_cat_list,
                                                      random_state=42)
        new_y = []
        for i in range(len(y_other_cat)):
            new_y.append(1)
        split_one_cat = int(req_percentage(len(y_one_cat), 80))
        split_other_cat = int(req_percentage(len(y_other_cat), 80))
        X_train = X_one_cat[:split_one_cat] + X_other_cat[:split_other_cat]
        X_test = X_one_cat[split_one_cat:] + X_other_cat[split_other_cat:]  # fixme +1+1

        y_train = []
        y_test = []
        for i in range(split_one_cat):
            y_train.append(0)
        for i in range(split_other_cat):
            y_train.append(1)

        for i in range(len(y_one_cat) - split_one_cat):
            y_test.append(0)
        for i in range(len(y_other_cat) - split_other_cat):
            y_test.append(1)

        X_train, y_train = shuffler(X_train, y_train)
        X_test, y_test = shuffler(X_test, y_test)

        X_vectorized_train = vectorizer.fit_transform(X_train)
        X_vectorized_test = vectorizer.transform(X_test)

        pickle.dump(X_vectorized_train, open("cache_20news/cache_X_train_" + category + ".p", "wb"))
        pickle.dump(y_train, open("cache_20news/cache_y_train_" + category + ".p", "wb"))
        pickle.dump(X_vectorized_test, open("cache_20news/cache_X_test_" + category + ".p", "wb"))
        pickle.dump(y_test, open("cache_20news/cache_y_test_" + category + ".p", "wb"))

    return X_vectorized_train, y_train, X_vectorized_test, y_test


def single_20_news():
    score_bnb = []
    score_prc = []
    for category in fetch_20newsgroups().target_names:
        X_train, y_train, X_test, y_test = news_gatherer(category)
        mean_score_bnb = []
        mean_score_prc = []
        samples = []
        for cap in range(7, 0, -1):
            print("cap: " + str(cap))
            new_cap_train = int(len(y_train) / cap) + 1
            new_cap_test = int(len(y_test) / cap) + 1
            mean_score_bnb.append(
                classify(BernoulliNB(), X_train[:new_cap_train], y_train[:new_cap_train], X_test[:new_cap_test],
                         y_test[:new_cap_test]))
            mean_score_prc.append(
                classify(Perceptron(), X_train[:new_cap_train], y_train[:new_cap_train], X_test[:new_cap_test],
                         y_test[:new_cap_test]))
            samples.append(new_cap_train+new_cap_test)
        score_bnb.append(mean_score_bnb[-1])
        score_prc.append(mean_score_prc[-1])
        #plot_scores(mean_score_bnb, mean_score_prc, samples, category,"20NewsGroup")

    multiplotter(list(fetch_20newsgroups().target_names), score_bnb, score_prc, "20NewsGroup")
    plt.show()


def reut(vectorizer):
    score_bnb = []
    score_prc = []
    for category in Reuters.top10categories():
        X_train, y_train, X_test, y_test = Reuters.return_X_y_reut(category, vectorizer)
        mean_score_bnb = []
        mean_score_prc = []
        samples = []
        for cap in range(10, 0, -1):
            print("cap: " + str(cap))
            new_cap_train = int(len(y_train) / cap) + 1
            new_cap_test = int(len(y_test) / cap) + 1
            mean_score_bnb.append(
                classify(BernoulliNB(), X_train[:new_cap_train], y_train[:new_cap_train], X_test[:new_cap_test],
                         y_test[:new_cap_test]))
            mean_score_prc.append(
                classify(Perceptron(), X_train[:new_cap_train], y_train[:new_cap_train], X_test[:new_cap_test],
                         y_test[:new_cap_test]))
            samples.append(new_cap_train+new_cap_test)
        score_bnb.append(mean_score_bnb[-1])
        score_prc.append(mean_score_prc[-1])
        plot_scores(mean_score_bnb, mean_score_prc, samples, category, "Reuters")
    multiplotter(Reuters.top10categories(), score_bnb, score_prc, "Reuters")
    plt.show()


def shuffler(group1, group2):
    c = list(zip(group1, group2))
    random.shuffle(c)
    return zip(*c)


def plot_scores(score_bnb, score_prc, samples, cat, dataset):
    new_title = "Learning Curves on " + dataset + " (" + cat + ")"
    plt.figure()
    plt.title(new_title)
    plt.xlabel("Esempi")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(samples, score_bnb, label='BenoullianNB')
    plt.plot(samples, score_prc, color="orange", label='Perceptron')
    plt.legend()
    # plt.savefig("bnb_prc_" + title + ".png", dpi=100)
    plt.show()


def classify(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    return score


def all_20news_group_cat():
    score_bnb = []
    score_prc = []
    cat_numbers = [2, 4, 8, 12, 16, 20]
    repetition = 2
    for categories in cat_numbers:
        print("Cat " + str(categories) + "/20")
        mean_score_bnb = 0
        mean_score_prc = 0
        for i in range(repetition):
            X_train, y_train, X_test, y_test = news_gatherer_cat_counter(categories)
            mean_score_bnb += (classify(BernoulliNB(), X_train, y_train, X_test, y_test))  # alpha=0.0001
            mean_score_prc += (classify(Perceptron(), X_train, y_train, X_test, y_test))  # max_iter=100
        score_bnb.append(mean_score_bnb / repetition)
        score_prc.append(mean_score_prc / repetition)

    multiplotter(cat_numbers, score_bnb, score_prc, "20 News Group ")


def not_all_reut():
    score_bnb = []
    score_prc = []
    num_cat = range(2, 11)
    cat_numbers = [2, 4, 6, 8, 10]
    repetition = 2
    for categories in range(2, 11):
        print("Cat " + str(categories) + "/10")
        mean_score_bnb = 0
        mean_score_prc = 0
        for i in range(repetition):
            X, y = Reuters.return_X_y_reut_notallcat(categories)
            X_shuffled, y_shuffled = shuffler(X, y)
            percented_quantity = int(req_percentage(len(X), 80))
            X_train_p = X_shuffled[:percented_quantity]
            X_test_p = X_shuffled[percented_quantity:]
            y_train = y_shuffled[:percented_quantity]
            y_test = y_shuffled[percented_quantity:]
            X_train = vectorizer.fit_transform(X_train_p)
            X_test = vectorizer.transform(X_test_p)

            mean_score_bnb += (classify(BernoulliNB(), X_train, y_train, X_test, y_test))
            mean_score_prc += (classify(Perceptron(), X_train, y_train, X_test, y_test))
        score_bnb.append(mean_score_bnb / repetition)
        score_prc.append(mean_score_prc / repetition)
    multiplotter(num_cat, score_bnb, score_prc, "Reuters ")
    # plt.savefig("bnb_prc_" + title + ".png", dpi=100)


def multiplotter(num_cat, scores_bnb, scores_prc, dataset):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    subtitle = dataset + " categories"
    fig.suptitle(subtitle)
    ax1.plot(range(1,len(scores_bnb)+1), scores_bnb, label="Bernoullian Naive Bayes")
    ax1.grid()
    ax1.legend()
    ax2.plot(range(1,len(scores_prc)+1,1), scores_prc, color='orange', label="Perceptron")
    ax2.grid()
    ax2.legend()
    plt.xticks(rotation=45)
    plt.show()


def req_percentage(full_number, percentage_you_want):
    return percentage_you_want * full_number / 100


if __name__ == '__main__':
    start = time.time()

    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words="english", min_df=3)

    print("Reperimento Datasets: 20News groups")
    #all_20news_group_cat()
    single_20_news()
    print("Reperimento Datasets: Reuters")
    # all_reuters()
    #not_all_reut()
    reut(vectorizer)
    print("Tempo impiegato: ", time.time() - start, " secondi")


