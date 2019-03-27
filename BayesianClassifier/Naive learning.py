import re

#야구뉴스 와 연예뉴스 구분하기 

def file_name (file_name):
    a=open(file_name, encoding='utf-8')
    A=a.read().lower()
    b=A.split()
    List=[]
    for i in b:
        data=re.compile('\W')
        dated=data.sub ('', i)
        List.append(dated)

    dictionary={}
    for j in List:
        dictionary[j]=100*List.count(j)/len(List)

    return dictionary


MLB=file_name('MLB.txt')
STARS=file_name('stars.txt')

Group=set()
for i in MLB.keys():
    Group.add(i)
for j in STARS.keys():
    Group.add(j)

Word=input('숫자를 입력하시오: ')
list_Group=list(Group)
chosen_word=list_Group[int(Word)]
print(chosen_word)

if chosen_word not in MLB:
    print('연예 뉴스')
elif chosen_word not in STARS:
    print('야구 뉴스')
else:
    MLB_word=MLB[chosen_word]
    STARS_word=STARS[chosen_word]

    if MLB_word> STARS_word:
        print('야구 뉴스')

    if MLB_word== STARS_word:
        print ('모름')

    else:
        print('연예  뉴스')
        

#기사가 야구뉴스인지 연예뉴스인지 구별하기

def open_file(file_name):
    a=open(file_name, encoding='utf-8')
    A=a.read()
    b=A.split()
    List=[]
    Qlist={'MLB':0, 'STARS':0}
    for i in b:
        data=re.compile('\W')
        dated=data.sub ('', i)
        List.append(dated.lower())

    for j in List:
        if j not in Group:
            continue
        if j in Group:
            if j not in MLB:
                Qlist['STARS']+=STARS[j]
            elif j not in STARS:
                Qlist['MLB']+=MLB[j]
            else:
                if MLB[j]>STARS[j]:
                    p=MLB[j]-STARS[j]
                    Qlist['MLB']+=p
                elif MLB[j]<STARS[j]:
                    q=STARS[j]-MLB[j]
                    Qlist['STARS']+=q
                else:
                    continue
    if Qlist['MLB']>Qlist['STARS']:
        print('야구 뉴스')
    else:
        print('연예 뉴스')
                        
