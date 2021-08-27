from pathlib import Path
from bs4 import BeautifulSoup
import os
import pickle


def top10categories(reuters_files, use_cache=True):
    if use_cache:
        # print("Utilizzo la cache di top10categories Reuters", end=" ")
        if Path(r"cacheTop10.txt").exists():
            with open("cacheTop10.txt", 'r') as f:
                top10 = [line.rstrip('\n') for line in f]
            return top10
        else:
            print("ma non esiste, quindi procedo a trovarla...", end=" ")
    file = open("reuters21578/all-topics-strings.lc.txt", "r")
    categories = []
    parts = []
    for line in file:
        parts = line.rstrip()
        parts = line.split(' ')
        parts = [p.rstrip() for p in parts]
        categories.append(parts)
    file.close()
    print("Inizio conteggio delle ricorrenze")
    # Conto quante occorrenze ci sono
    nomi = {}
    for i in range(len(categories)):
        nomi[i] = categories[i]
    conteggi = {}
    for j in range(len(categories)):
        conteggi[j] = 0
    for x in reuters_files:
        y = open(x, "r")
        io = y.readlines()
        for line in io:
            for j in nomi.keys():
                if "<D>" + str(nomi.get(j)).rstrip("']").lstrip("['") + "</D>" in line:
                    conteggi[j] = conteggi[j] + 1
        y.close()
    sorted_top10_categories = {}
    count = 0
    while count != 10:
        maxKey = 0
        maxValue = 0
        for key, value in conteggi.items():
            if value >= maxValue:
                maxKey = key
                maxValue = value
        count += 1
        del conteggi[maxKey]
        sorted_top10_categories.__setitem__(maxKey, maxValue)
    Top10 = []
    for key, value in nomi.items():
        if key in sorted_top10_categories.keys():
            Top10.append(value)
    with open("cacheTop10.txt", 'w') as f:
        for s in Top10:
            f.write(str(s).replace("['", "").replace("']", "") + '\n')
    return Top10


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


def return_X_y_reut(selected_category):
    if Path(r"cache_reuters/cache_binary_" + selected_category + "_X").exists() and Path(
            r"cache_reuters/cache_binary_" + selected_category + "_y").exists():
        with open("cache_reuters/cache_binary_" + selected_category + "_X", "rb") as f:
            X = pickle.load(f)
        f.close()
        with open("cache_reuters/cache_binary_" + selected_category + "_y", "rb") as f:
            y = pickle.load(f)
        return X, y
    else:
        reut_files = file_list()
        top10cat = top10categories(reut_files)
        X = []
        y = []
        for x in reut_files:
            io = open(x, "r")
            strr = io.read()
            docs = strr.split("</REUTERS>")
            for segment in docs:
                if ('LEWISSPLIT="TRAIN"' or 'LEWISSPLIT="TEST"') in segment and 'TOPICS="YES"' in segment:
                    inlist = False
                    if "<D>" + str(selected_category).rstrip("']").lstrip("['") + "</D>" in segment:
                        soup = BeautifulSoup(segment, features="html.parser")
                        reut_body = soup.findAll("body")
                        if len(reut_body) != 0:
                            body = str(reut_body[0].string)
                            X.append(body)
                            y.append("cate")
                    else:
                        for cate in top10cat:
                            if "<D>" + str(cate).rstrip("']").lstrip("['") + "</D>" in segment and not inlist:
                                soup = BeautifulSoup(segment, features="html.parser")
                                reut_body = soup.findAll("body")
                                if len(reut_body) != 0:
                                    inlist = True
                                    body = str(reut_body[0].string)
                                    X.append(body)
                                    y.append("not_cate")
        with open("cache_reuters/cache_binary_" + selected_category + "_X", 'wb') as f:
            pickle.dump(X, f)
        f.close()
        with open("cache_reuters/cache_binary_" + selected_category + "_y", "wb") as f:
            pickle.dump(y, f)
        return X, y


def return_X_y_reut_bnb():
    # all_cat = dict()
    reut_files = file_list()
    top10cat = top10categories(reut_files)
    X = []
    y = []
    for x in reut_files:
        io = open(x, "r")
        strr = io.read()
        docs = strr.split("</REUTERS>")
        for segment in docs:
            # if any("<D>" + str(cat).rstrip("']").lstrip("['") + "</D>" in segment for cat in top10cat):
            if ('LEWISSPLIT="TRAIN"' or 'LEWISSPLIT="TEST"') in segment and 'TOPICS="YES"' in segment:
                inlist = False
                # for cate in top10cat:

                for cate in top10cat:
                    if "<D>" + str(cate).rstrip("']").lstrip("['") + "</D>" in segment and not inlist:
                        soup = BeautifulSoup(segment, features="html.parser")
                        reut_body = soup.findAll("body")
                        if len(reut_body) != 0:
                            inlist = True
                            body = str(reut_body[0].string)
                            X.append(body)
                            y.append(cate)

                            # if cate not in all_cat:
                            #    all_cat[cate] = list()
                            # all_cat[cate].append(body)
    return X, y
