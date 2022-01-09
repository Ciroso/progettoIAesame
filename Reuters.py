from pathlib import Path
from bs4 import BeautifulSoup
import os
import pickle

from main import shuffler, req_percentage


def top10categories():
    reuters_files = file_list()
    if Path(r"cacheTop10.txt").exists():
        with open("cacheTop10.txt", 'r') as f:
            top10 = [line.rstrip('\n') for line in f]
        return top10

    file = open("reuters21578/all-topics-strings.lc.txt", "r")
    categories = [line.rstrip('\n') for line in file]
    file.close()
    all_categories = {}
    for cat in categories:
        all_categories.update({cat: 0})
    for cat in categories:
        for x in reuters_files:
            io = open(x, "r")
            strr = io.read()
            docs = strr.split("</REUTERS>")
            for segment in docs:
                if ('LEWISSPLIT="TRAIN"' or 'LEWISSPLIT="TEST"') in segment and 'TOPICS="YES"' in segment:
                    if "<D>" + str(cat).rstrip("']").lstrip("['") + "</D>" in segment:
                        all_categories[cat] += 1
    print("")
    best_cat_number = 0
    top_10_cat = []
    t = []
    while best_cat_number < 10:
        temp_best_cat = ""
        temp_best_val = 0
        for cat in categories:
            if cat in top_10_cat:
                pass
            else:
                if all_categories[cat] > temp_best_val:
                    temp_best_val = all_categories[cat]
                    temp_best_cat = cat
        top_10_cat.append(temp_best_cat)
        t.append(temp_best_val)
        all_categories.pop(temp_best_cat)
        best_cat_number += 1

    with open("cacheTop10.txt", 'w') as f:
        for s in top_10_cat:
            f.write(str(s) + '\n')
    return top_10_cat


def file_list():
    if Path(r"cache_list.txt").exists():
        with open("cache_list.txt", 'r') as f:
            cache_list = [line.rstrip('\n') for line in f]
        return cache_list
    else:
        cwf = os.getcwd()
        reut_files = []
        for i in range(0, 22):
            if i < 10:
                reut_files.append(cwf + "/reuters21578/reut2-00" + str(i) + ".sgm")
            else:
                reut_files.append(cwf + "/reuters21578/reut2-0" + str(i) + ".sgm")
        with open("cache_list.txt", 'w') as f:
            for s in reut_files:
                f.write(str(s).replace("['", "").replace("']", "") + '\n')
        return reut_files


def return_X_y_reut(category, vectorizer):
    if Path(r"cache_reuters/cache_X_train_" + category + ".p").exists() and Path(
            r"cache_reuters/cache_y_train_" + category + ".p").exists() and Path(
        r"cache_reuters/cache_X_test_" + category + ".p").exists() and Path(
        r"cache_reuters/cache_y_test_" + category + ".p").exists():
        print("CACHE")
        X_train = pickle.load(open("cache_reuters/cache_X_train_" + category + ".p", "rb"))
        y_train = pickle.load(open("cache_reuters/cache_y_train_" + category + ".p", "rb"))
        X_test = pickle.load(open("cache_reuters/cache_X_test_" + category + ".p", "rb"))
        y_test = pickle.load(open("cache_reuters/cache_y_test_" + category + ".p", "rb"))
        return X_train, y_train, X_test, y_test
    else:
        print("NO CACHE")
        reut_files = file_list()
        top10cat = top10categories()
        X_one_cat = []
        y_one_cat = []
        X_other_cat = []
        y_other_cat = []
        for x in reut_files:
            io = open(x, "r")
            strr = io.read()
            docs = strr.split("</REUTERS>")
            for segment in docs:
                if ('LEWISSPLIT="TRAIN"' or 'LEWISSPLIT="TEST"') in segment and 'TOPICS="YES"' in segment:
                    inlist = False
                    for cat in top10cat:
                        if ("<D>" + str(cat).rstrip("']").lstrip("['") + "</D>" in segment) and (not inlist):
                            soup = BeautifulSoup(segment, features="html.parser")
                            reut_body = soup.findAll("body")
                            if len(reut_body) != 0:
                                body = str(reut_body[0].string)
                                if cat == category:
                                    X_one_cat.append(body)
                                    y_one_cat.append(0)
                                else:
                                    X_other_cat.append(body)
                                    y_other_cat.append(1)

        new_y = []
        for i in range(len(y_other_cat)):
            new_y.append(1)
        split_one_cat = int(req_percentage(len(y_one_cat), 80))
        split_other_cat = int(req_percentage(len(y_other_cat), 80))
        X_train = X_one_cat[:split_one_cat] + X_other_cat[:split_other_cat]
        X_test = X_one_cat[split_one_cat:] + X_other_cat[split_other_cat:]

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

        pickle.dump(X_vectorized_train, open("cache_reuters/cache_X_train_" + category + ".p", "wb"))
        pickle.dump(y_train, open("cache_reuters/cache_y_train_" + category + ".p", "wb"))
        pickle.dump(X_vectorized_test, open("cache_reuters/cache_X_test_" + category + ".p", "wb"))
        pickle.dump(y_test, open("cache_reuters/cache_y_test_" + category + ".p", "wb"))
        return X_vectorized_train, y_train, X_vectorized_test, y_test

def return_X_y_reut_notallcat(numcat):
    '''
    if Path(r"cache_reuters/cache_nonbinary_X.p").exists() and Path(
            r"cache_reuters/cache_nonbinary_y.p").exists():
        X = pickle.load(open("cache_reuters/cache_nonbinary_X.p", "rb"))
        y = pickle.load(open("cache_reuters/cache_nonbinary_y.p", "rb"))
        return X, y
    else:
    '''
    reut_files = file_list()
    top10cat = top10categories()
    X = []
    y = []
    for x in reut_files:
        io = open(x, "r")
        strr = io.read()
        docs = strr.split("</REUTERS>")
        for segment in docs:
            if ('LEWISSPLIT="TRAIN"' or 'LEWISSPLIT="TEST"') in segment and 'TOPICS="YES"' in segment:
                inlist = False
                for cat in top10cat[:numcat]:
                    if ("<D>" + str(cat).rstrip("']").lstrip("['") + "</D>" in segment) and (not inlist):
                        soup = BeautifulSoup(segment, features="html.parser")
                        reut_body = soup.findAll("body")
                        if len(reut_body) != 0:
                            body = str(reut_body[0].string)
                            X.append(body)
                            y.append(cat)

        # pickle.dump(X, open("cache_reuters/cache_nonbinary_X.p", "wb"))
        # pickle.dump(y, open("cache_reuters/cache_nonbinary_Y.p", "wb"))
        return X, y
