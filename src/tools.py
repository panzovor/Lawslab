
__author__ = 'czb'
import jieba
import jieba.posseg as pog
import sys
encoding ="utf-8"

def get_regex_str_index(regex,content):
    iter = re.finditer(regex,content)
    res =[]
    for it in iter:
        res.append(it.start())
    return res

def read_txt(filepath,encoding= "utf-8"):
    with open(filepath,mode="r",encoding=encoding) as file:
        content = file.read()
    return content

def save_txt(filepath,content, encoding= encoding,split =","):
    with open(filepath,mode="w",encoding=encoding) as file:
        if isinstance(content,str):
            file.write(content)
        elif isinstance(content[0],str):
            file.write('\n'.join(content))
        elif isinstance(content[0],list):
            for line in content:
                if isinstance(line,list):
                    if isinstance(line[0],str):
                        file.write(split.join(line)+"\n")
                    else:
                        tmp =""
                        for var in line:
                            tmp+=str(var)+split
                        file.write(tmp[:-len(split)]+"\n")
                else:
                    file.write(line)


jieba.load_userdict("../res/parameter/segmentation/user_dict.txt")
stopwords = set()
with open("../res/parameter/segmentation/stop_words.txt",encoding="utf-8")as file:
    lines = file.readlines()
    for var in lines:
        stopwords.add(var)
def cut_sentence(sentence):
    words = list(jieba.cut(sentence))
    res = []
    for w in words:
        if w not in stopwords:
            res.append(w)
    return res


def read_into_map(file):
    content = read_txt(file).strip()
    res = {}
    if len(content) > 0:
        content =content.split("\n")
        for line in content:
            line = line.split(",", maxsplit=3)
            if line[0] not in res.keys():
                res[line[0]] = {}
            if line[1] not in res[line[0]].keys():
                res[line[0]][line[1]] = float(line[-1])
    return res

def read_into_map2(file):
    content = read_txt(file).strip()
    res = {}
    if len(content) > 0:
        content = content.split("\n")
        for line in content:
            line = line.split(",", maxsplit=3)
            if line[0] not in res.keys():
                res[line[0]] = float(line[-1])
    return res

def read_lines(filepath,encoding="utf-8"):
    with open(filepath,encoding=encoding) as file :
        return file.readlines()

def cut_tag_sentence(sentence):
    words,tag =[],[]
    for w in pog.cut(sentence):
        if w.word not in stopwords:
            words.append(w.word)
            tag.append(w.flag)
    return words,tag


def show_process(i,all_l,num= 10):
    if i % int(all_l / num) == 0:
        sys.stdout.write("*")
        sys.stdout.flush()

def show_file(filepath):
    with open(filepath,encoding="utf-8") as file:
        content = file.readlines()

    i =0
    while i<2000:
        print(content[i])
        print(len(content[i].split(",")))
        tmp = input()
        if tmp == "q":
            break
        if tmp != "":
            i=int(tmp)
        i+=1

from sklearn import metrics
import  re
def analyse_predict_result(real_y,predict_y):
    report = metrics.classification_report(real_y,predict_y)
    confuse =metrics.confusion_matrix(real_y,predict_y)
    report = [re.split(" {2,20}",var.strip()) for var in report.split("\n")[2:] if var.strip()!=""]
    return report,confuse

if __name__ == "__main__":
    # show_file("../res/seperated_data/test.arff")
    # show_file("../res/seperated_data/train.arff")

    predict_type = ["啊","非官方","企鹅"]
    real_type = ["啊","非官方","企鹅"]
    keys = list(set(predict_type))
    real_type_map, predict_type_map = [], []
    for kk in real_type:
        real_type_map.append(keys.index(kk))
    for kk in predict_type:
        predict_type_map.append(keys.index(kk))

    real_type_map = real_type
    predict_type_map = predict_type

    report, confuse = analyse_predict_result(real_type_map,predict_type_map)
    for line in report:
        print(line[0]+"\t"+'\t'.join(list(map(str, line[1:]))))

