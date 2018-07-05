import os
import re
# from util import *

DATA_DIR = os.path.join(os.path.abspath('..'), 'data')

def clean_text(text):
    text = text.replace('\xa0', '').replace('\u3000', '')#.replace('\n', '')
    return text
# with open(os.path.join(DATA_DIR, 'infoExample2100-2200.txt')) as file:
#     data = file.read()
# personList = data.split('*********&&&&&&&&')

# def extract(text):
#     pattern = re.compile(r'(<[a-zA-Z]+?>[^<>]+?</[\\a-zA-Z]+?>)')
#     return pattern.findall(text)
'''
PER
ADR
AFF
TIT
JOB
'''
# text = clean_text(personList[0])
# extractList = extract(text)
# print(extractList)

# for item in extractList:
#     res = re.search(r'<([a-zA-Z]+?)>([^<>]+?)</[\\a-zA-Z]+?>', item)
#     print(res.span(2))
tag = {'name' : 'PER', 'address' : 'ADR', 'aff' : 'AFF', 'title' : 'TIT', 'job' : 'JOB'}


def tagging(text):
    mark = ['O'] * len(text)

    while(1):
        pattern = re.compile(r'(<([a-zA-Z]+?)>([^<>]+?)</[\\a-zA-Z]+?>)')
        res = re.search(pattern, text)

        if res is None:
            break

        length = len(res.group(3))
        begin = res.span(1)[0]
        tag_name = res.group(2)
        # print(res.group(1), res.group(3), res.span(3), begin, length)
        text = re.sub(pattern, res.group(3), text, count=1)
        # print(text[begin:begin+length])
        if tag_name in tag:
            mark[begin] = 'B-' + tag[tag_name]
            for i in range(begin+1, begin+length):
                mark[i] = 'I-' + tag[tag_name]
        # print(mark[140:160])

    for i, ch in enumerate(text):
        if ch == '\n':
            print(ch) 
        elif ch not in [' ', '\r', '\t', '\u2003', '\u2009']:
            print(ch, '\t', mark[i])


train_data = ['infoExample2000-2100.txt', 'infoExample2100-2200.txt', 'infoExample2200-2300.txt', 'infoExample2400-2500.txt', 'infoExample2600-2700.txt']
test_data = ['infoExample2700-2800.txt', 'infoExample2800-2900.txt']
for file_name in test_data:
    # print(file_name)
    with open(os.path.join(DATA_DIR, file_name), 'r') as file:
        data = file.read()
    personList = data.split('*********&&&&&&&&')
    for text in personList:
        tagging(clean_text(text))


    
 
