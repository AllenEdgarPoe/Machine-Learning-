import re, bs4, urllib.request, urllib.parse

def loadDataSet(document1, document2):
    document1 = open(document1, encoding = "UTF-8")
    document1 = document1.read().lower()
    document1 = document1.split()
    baseball = []
    for i in document1:
        baseball.append(re.sub('[s]|[es]|[.]|[,]|["]|[!]|[?]$', '', i))
    document2 = open(document2, encoding = "UTF-8")
    document2 = document2.read().lower()
    document2 = document2.split()
    celeb = []
    for i in document1:
        celeb.append(re.sub('[s]|[es]|[.]|[,]|["]|[!]|[?]$', '', i))
    traindata1 = dict()
    for i in document1:
        if i in traindata1.keys():
            traindata1[i] += 1
        else:
            traindata1[i] = 1
    traindata2 = dict()
    for i in document2:
        if i in traindata2.keys():
            traindata2[i] += 1
        else:
            traindata2[i] = 1
    return traindata1, traindata2


def naive_bayes_train(first, second):
    length_first = sum(first.values())
    length_second = sum(second.values())
    trained = []
    words = dict()
    words.update(first)
    words.update(second)
    words = set(words)
    for i in words:
        if (i in first.keys()) and (i in second.keys()):
            trained.append([i, int(first[i])/length_first, int(second[i])/length_second])
        elif i not in second.keys():
            trained.append([i, int(first[i])/length_first, int(0)])
        else:
            trained.append([i, int(0), int(second[i])/length_second])
    return trained



def decide(trained, address):
    target = open(address, encoding = "UTF-8") 
    target = target.read().lower()
    target = target.split()
    cal = []
    for i in target:
        cal.append(re.sub('[s]|[es]|[.]|[,]|["]|[!]|[?]$', '', i))
    target = cal
    decider = {"야구 뉴스":0, "연애 뉴스":0}
    for i in trained:
        for j in target:
            if i[0] == j:
                if i[1]>i[2]:
                    decider["연애 뉴스"] += (i[1] - i[2])
                else:
                    decider["야구 뉴스"] += (i[2] - i[1])
            else:
                continue
    if decider["연애 뉴스"] > decider["야구 뉴스"]:
        print("연애 뉴스")
    elif decider["연애 뉴스"] < decider["야구 뉴스"]:
        print("야구 뉴스")
    else:
        print("판독 불가")


