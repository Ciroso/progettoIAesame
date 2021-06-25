from pathlib import Path
from bs4 import BeautifulSoup
import codecs
import os


def top10categories(reuters_files, use_cache=True):
    if use_cache:
        print("Utilizzo la cache di top10categorie reuters")
        if Path(r"cacheTop10.txt").exists():
            with open("cacheTop10.txt", 'r') as f:
                top10 = [line.rstrip('\n') for line in f]
            return top10
        else:
            print("Ma non esiste, quindi procedo a trovarla")
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
    # CONTO QUANTE OCCORRENZE DI CIASCUNA CATEGORIA CI SONO#
    nomi = {}
    for i in range(len(categories)):
        nomi[i] = categories[i]
    conteggi = {}
    for j in range(len(categories)):
        conteggi[j] = 0
    for x in reuters_files:
        y = open(x, 'r')
        io = y.readlines()
        for line in io:
            for j in nomi.keys():
                if '<D>' + str(nomi.get(j)).rstrip("']").lstrip("['") + '</D>' in line:
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


def Reuters_load_files():
    cwf = os.getcwd()
    reut_files = []
    for i in range(0, 22):
        if i < 10:
            reut_files.append(cwf + "/reuters21578/reut2-00" + str(i) + ".sgm")
        else:
            reut_files.append(cwf + "/reuters21578/reut2-0" + str(i) + ".sgm")
    '''
    docfiles = [current_working_folder + '/reuters21578/reut2-000.sgm',
                current_working_folder + '/reuters21578/reut2-001.sgm',
                current_working_folder + '/reuters21578/reut2-002.sgm',
                current_working_folder + '/reuters21578/reut2-003.sgm',
                current_working_folder + '/reuters21578/reut2-004.sgm',
                current_working_folder + '/reuters21578/reut2-005.sgm',
                current_working_folder + '/reuters21578/reut2-006.sgm',
                current_working_folder + '/reuters21578/reut2-007.sgm',
                current_working_folder + '/reuters21578/reut2-008.sgm',
                current_working_folder + '/reuters21578/reut2-009.sgm',
                current_working_folder + '/reuters21578/reut2-010.sgm',
                current_working_folder + '/reuters21578/reut2-011.sgm',
                current_working_folder + '/reuters21578/reut2-012.sgm',
                current_working_folder + '/reuters21578/reut2-013.sgm',
                current_working_folder + '/reuters21578/reut2-014.sgm',
                current_working_folder + '/reuters21578/reut2-015.sgm',
                current_working_folder + '/reuters21578/reut2-016.sgm',
                current_working_folder + '/reuters21578/reut2-017.sgm',
                current_working_folder + '/reuters21578/reut2-018.sgm',
                current_working_folder + '/reuters21578/reut2-019.sgm',
                current_working_folder + '/reuters21578/reut2-020.sgm',
                current_working_folder + '/reuters21578/reut2-021.sgm']
    '''
    listaNomiCategorieTop10 = top10categories(reut_files)

    """print('lista dei tag riferiti alle 10 categorie più frequenti:')
    for y in listaNomiCategorieTop10:
        print('<D>' + str(y).rstrip("']").lstrip("['") + '</D>')
    """
    print('Orgnizzo file in cartelle')

    for x in reut_files:
        io = open(x, 'r')
        strr = io.read()
        docs = strr.split("</REUTERS>")
        for sample in docs:
            if any('<D>' + str(y).rstrip("']").lstrip("['") + '</D>' in sample for y in listaNomiCategorieTop10):
                if not os.path.exists(cwf + "/data-set"):
                    os.makedirs(cwf + "/data-set")
                if 'LEWISSPLIT="TRAIN"' in sample and 'TOPICS="YES"' in sample:
                    if not os.path.exists(cwf + "/data-set/training-set"):
                        os.makedirs(cwf + "/data-set/training-set")
                    for y in listaNomiCategorieTop10:
                        if '<D>' + str(y).rstrip("']").lstrip("['") + '</D>' in sample:
                            if not os.path.exists(cwf + "/data-set/training-set/" + str(y).rstrip("']").lstrip("['")):
                                os.makedirs(cwf + "/data-set/training-set/" + str(y).rstrip("']").lstrip("['"))
                            os.chdir(cwf + "/data-set/training-set/" + str(y).rstrip("']").lstrip("['"))
                            indexTitle = sample.index("NEWID=")
                            endTitle = sample.index('">')
                            title = '0' + sample[indexTitle + 7:endTitle]
                            soup = BeautifulSoup(sample, features="html.parser")
                            reut_body = soup.findAll("body")
                            if len(reut_body) != 0:
                                body = str(reut_body[0].string)
                                output = codecs.open(title + '.txt', 'w', "utf-8")
                                output.write(body)

                elif 'LEWISSPLIT="TEST"' in sample and 'TOPICS="YES"' in sample:
                    if not os.path.exists(cwf + "/data-set/test-set"):
                        os.makedirs(cwf + "/data-set/test-set")
                    for y in listaNomiCategorieTop10:
                        if '<D>' + str(y).rstrip("']").lstrip("['") + '</D>' in sample:
                            if not os.path.exists(cwf + "/data-set/test-set/" + str(y).rstrip("']").lstrip("['")):
                                os.makedirs(cwf + "/data-set/test-set/" + str(y).rstrip("']").lstrip("['"))
                            os.chdir(cwf + "/data-set/test-set/" + str(y).rstrip("']").lstrip("['"))
                            indexTitle = sample.index("NEWID=")
                            endTitle = sample.index('">')
                            title = '0' + sample[indexTitle + 7:endTitle]
                            soup = BeautifulSoup(sample, features="html.parser")
                            reut_body = soup.findAll("body")
                            if len(reut_body) != 0:
                                body = str(reut_body[0].string)
                                output = codecs.open(title + '.txt', 'w', "utf-8")
                                output.write(body)
    print('Organizzazione in cartelle completata.')


def return_X_y_reut():
    cwf = os.getcwd()
    reut_files = []
    for i in range(0, 22):
        if i < 10:
            reut_files.append(cwf + "/reuters21578/reut2-00" + str(i) + ".sgm")
        else:
            reut_files.append(cwf + "/reuters21578/reut2-0" + str(i) + ".sgm")

    listaNomiCategorieTop10 = top10categories(reut_files)

    # print('Orgnizzo file in cartelle')
    X = []
    y_target = []
    for x in reut_files:
        io = open(x, 'r')
        strr = io.read()
        docs = strr.split("</REUTERS>")
        for sample in docs:
            if any('<D>' + str(y).rstrip("']").lstrip("['") + '</D>' in sample for y in listaNomiCategorieTop10):
                if 'LEWISSPLIT="TRAIN"' in sample and 'TOPICS="YES"' in sample:
                    for y in listaNomiCategorieTop10:
                        if '<D>' + str(y).rstrip("']").lstrip("['") + '</D>' in sample:
                            soup = BeautifulSoup(sample, features="html.parser")
                            reut_body = soup.findAll("body")
                            if len(reut_body) != 0:
                                body = str(reut_body[0].string)
                                X.append(body)
                                y_target.append(y)

                elif 'LEWISSPLIT="TEST"' in sample and 'TOPICS="YES"' in sample:
                    for y in listaNomiCategorieTop10:
                        if '<D>' + str(y).rstrip("']").lstrip("['") + '</D>' in sample:
                            soup = BeautifulSoup(sample, features="html.parser")
                            reut_body = soup.findAll("body")
                            if len(reut_body) != 0:
                                body = str(reut_body[0].string)
                                X.append(body)
                                y_target.append(y)
    print('Organizzazione in cartelle completata.')
    return X, y_target


def warmup():
    if Path(r"data-set/").exists():
        print("Dataset reuter presente")
    else:
        print("Dataset reuter NON presente")
        Reuters_load_files()


if __name__ == '__main__':
    warmup()
