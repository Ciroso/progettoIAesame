import fileinput

from nltk.corpus import reuters
def forse():
    from bs4 import BeautifulSoup, SoupStrainer
    f = open('reuters21578/reut2-000.sgm', 'r')
    data= f.read()
    soup = BeautifulSoup(data,features="html.parser")
    contents = soup.findAll('body')
    for content in contents:
        print(content.text)

    contentss = soup.findAll('topics')
    for content in contents:
        print(content.text)

    content = soup.findAll("reuters")
    print()

reuter_docs = []
for i in range(0,22):
    if i < 10:
        reuter_docs.append("/reuters21578/reut2-00" + str(i))
    else:
        reuter_docs.append("/reuters21578/reut2-0" + str(i))


#def unga():

def categorie_freq():
    print('Creazione file categorie...\n')
    filenames = ['/reuters21578/all-topics-strings.lc.txt']

    with open('/reuters21578/categories', 'w') as fout, fileinput.input(filenames) as fin:
        for line in fin:
            fout.write(line)
    print('Inserimento categorie in lista..\n')
    file = open('/categories', 'r')
    categories = []
    parts = []
    for line in file:
        parts = line.rstrip()
        parts = line.split(' ')
        parts = [p.rstrip() for p in parts]
        categories.append(parts)
    print('Lista categorie:' + str(categories))
    print('Numero delle categorie letto da lista: ' + str(len(categories)))
    file.close()
    print('Inserimento completato')
    ###FILES####
    print('---------Inizio analisi delle occorrenze delle 10 categorie più frequenti-------')
    # CONTO QUANTE OCCORRENZE DI CIASCUNA CATEGORIA CI SONO#
    nomi = {}
    for i in range(len(categories)):
        nomi[i] = categories[i]
    print('dizionario delle categorie:\n' + str(nomi))
    conteggi = {}
    for j in range(len(categories)):
        conteggi[j] = 0
    for x in reuters_files:
        y = open(x, 'r')
        io = y.readlines()
        t = time.time()
        print('File SGML ' + str(y) + ' letto. Ricerca linea per linea in corso..')
        for line in io:
            for j in nomi.keys():
                if '<D>' + str(nomi.get(j)).rstrip("']").lstrip("['") + '</D>' in line:
                    conteggi[j] = conteggi[j] + 1
        y.close()
        tf = time.time() - t
        print('Ricerca completata: exec time : %0.3fs' % tf)
    print(str(conteggi))
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
    print('Dizionario delle dieci categorie più frequenti: ' + str(sorted_top10_categories))
    listaNomiCategorieTop10 = []
    for key, value in nomi.items():
        if key in sorted_top10_categories.keys():
            listaNomiCategorieTop10.append(value)
    return listaNomiCategorieTop10

#TODO
categorie_freq()

for x in reuter_docs: #"reuters21578/reut2-000.sgm"#for x in docfiles:
    io = open(x, 'r')
    strr = io.read()
    docs = strr.split("</REUTERS>")
    for sample in docs:
        if any('<D>' + str(y).rstrip("']").lstrip("['") + '</D>' in sample for y in listaNomiCategorieTop10):
            if not os.path.exists(current_working_folder + "/data-set"):
                os.makedirs(current_working_folder + "/data-set")
            if 'LEWISSPLIT="TRAIN"' in sample and 'TOPICS="YES"' in sample:
                if not os.path.exists(current_working_folder + "/data-set/training-set"):
                    os.makedirs(current_working_folder + "/data-set/training-set")
                for y in listaNomiCategorieTop10:
                    if '<D>' + str(y).rstrip("']").lstrip("['") + '</D>' in sample:
                        if not os.path.exists(current_working_folder + "/data-set/training-set/" + str(y).rstrip(
                                "']").lstrip("['")):
                            os.makedirs(
                                current_working_folder + "/data-set/training-set/" + str(y).rstrip("']").lstrip(
                                    "['"))
                        os.chdir(
                            current_working_folder + "/data-set/training-set/" + str(y).rstrip("']").lstrip("['"))
                        indexTitle = sample.index("NEWID=")
                        endTitle = sample.index('">')
                        title = '0' + sample[indexTitle + 7:endTitle]
                        body = Preprocessing.body_extractor(sample)
                        output = open(title + '.txt', 'w')
                        output.write(sample)
            elif 'LEWISSPLIT="TEST"' in sample and 'TOPICS="YES"' in sample:
                if not os.path.exists(current_working_folder + "/data-set/test-set"):
                    os.makedirs(current_working_folder + "/data-set/test-set")
                for y in listaNomiCategorieTop10:
                    if '<D>' + str(y).rstrip("']").lstrip("['") + '</D>' in sample:
                        if not os.path.exists(current_working_folder + "/data-set/test-set/" + str(y).rstrip(
                                "']").lstrip("['")):
                            os.makedirs(
                                current_working_folder + "/data-set/test-set/" + str(y).rstrip("']").lstrip("['"))
                        os.chdir(current_working_folder + "/data-set/test-set/" + str(y).rstrip("']").lstrip("['"))
                        indexTitle = sample.index("NEWID=")
                        endTitle = sample.index('">')
                        title = '0' + sample[indexTitle + 7:endTitle]
                        body = Preprocessing.body_extractor(sample)
                        output = open(title + '.txt', 'w')
                        output.write(body)
    print('Organizzazione in cartelle completata.')