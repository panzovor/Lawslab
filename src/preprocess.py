
__author__ = 'czb'

import src.tools as tools
import re
import random
jingdu= 4
sentence_min_len =3
sentences_regex = "，|。|：|；"

max_samples = 50000

def content_filter(content):
    if "。【" in content:
        content = content.replace("。【","。】【")
    if "，【" in content:
        content = content.replace("，【","，】【")
    if "；【" in content:
        content = content.replace("；【","；】【")
    content = get_sucheng(content)
    fake_label_regex = "\[.{2,7}\]"
    fayuan_regex = ".{2,5}省.{1,20}法院\n"
    zihao_regex = ".{3,10}字第.{1,6}号\n"
    content = content.strip()
    content = re.sub(fake_label_regex, "", content)
    content = re.sub(fayuan_regex,"",content)
    content = re.sub(zihao_regex,"",content)
    content = content.replace("\n","")
    return content

def get_assey(content,filter = True):
    instrument_regex = "【--\d{1,6}--】"
    asseys = [var.strip() for var in re.split(instrument_regex,content) if len(var.strip())>0]
    asseys_no = re.findall(instrument_regex,content)
    result ={}
    for i in range(len(asseys_no)):
        no = asseys_no[i][len("【--"):-len("--】")]
        # print(no,asseys_no[i])
        if no not in result.keys():
            if filter:
                result[no] = content_filter(asseys[i])
            else:
                result[no] = asseys[i]
    return result

def label_content_data(content):
    result = {}
    label_regex = "【.{2,18}?：.*?】"
    content = content.replace(":", "：")
    content = content.replace(",", "，")
    labeled_data = re.findall(label_regex, content)
    unlabeled_data = re.sub(label_regex, "", content)
    illegal ="：|，|。"
    for ldata in labeled_data:
        ldata = ldata.strip()
        label = ldata[1:ldata.index("：")]
        label = label.replace("【", "")
        label = label.replace("】", "")
        label = label.replace("-", "")
        if len(re.findall(illegal,label))>0:
            label = "None"
        data = ldata[ldata.index("：") + 1:-1]
        if label not in result.keys():
            result[label] = []
        if "【" not in data and "】" not in data:
            result[label].append(data.strip())
    if len(unlabeled_data) > 0:
        result["None"] = []
        for uldata in re.split(sentences_regex, unlabeled_data):
            if len(uldata.strip()) > sentence_min_len and ("【" not in uldata and "】" not in uldata):
                result["None"].append(uldata.strip())
    return result

def content_preprocess(content):
    no_content = get_assey(content)
    nos = list(sorted(no_content.keys()))
    nll,lnl,ll ={},{},{}
    for no in nos:
        content = no_content[no]
        nll[no] = label_content_data(content)
        for label in nll[no].keys():
            if label not in lnl.keys():
                lnl[label] = {}
            lnl[label][no] = nll[no][label]
            if label not in ll.keys():
                ll[label] = []
            for line in nll[no][label]:
                ll[label].append(line.replace(",","，"))
    return lnl,nll,ll

def label_content(data_filepth ="../res/data/已完成标注（6818）.txt"):
    data = tools.read_txt(data_filepth)
    return content_preprocess(data)

def seperate_data_by_label(data_file,train_file,test_file,rate = 0.75,shuffle = False):
    content = tools.read_txt(data_file).strip().split("\n")
    content.pop(0)
    data = {}
    for line in content:
        line = line.split(",",maxsplit=3)
        label = line[0]
        if label not in data.keys():
            data[label] = []
        data[label].append(line[1:])
    train_data,test_data =[],[]
    for label in data.keys():
        tmp_data = data[label]
        if shuffle:
            random.shuffle(tmp_data)
        train_size = int(len(tmp_data)*rate)
        test_size = int(len(tmp_data)-train_size)
        for var in tmp_data[:train_size]:
            train_data.append([label]+var)
        for var in tmp_data[-test_size:]:
            test_data.append([label]+var)
    if train_file!=None:
        tools.save_txt(train_file,train_data)
    if test_file!=None:
        tools.save_txt(test_file,test_data)
    return train_data,test_data


def csv2arff_file(featured_data,arff_file,arff_title = None):
    labels = []
    data = []
    for line in featured_data:
        print(line)
        print(line[-1])
        input()
        if line[-1][-1] not in labels:
            labels.append(line[-1][-1])
        tmp = line[-1][:-1]+[labels.index(line[-1][-1])]
        data.append(','.join(tmp))

    if arff_title == None:
        arff_title = "@relation  tmp\n"
        for i in range(len(fe.all_feature_len)):
            arff_title+="@attribute att"+str(i)+" numeric\n"
        arff_title+= "@attribute class {"
        for i in range(len(labels)):
            arff_title+=str(i)+","
        arff_title = arff_title[:-1]+"}\n@data\n"
    arff_string = arff_title+'\n'.join(data)
    tools.save_txt(arff_file,arff_string)
    return labels,arff_string

def csv_file2arff_file(csv_file,save_path,arff_title = None):
    content = tools.read_lines(csv_file)
    data = ''.join(content[1:])
    arff_string =""
    if arff_title == None:
        arff_title = "@relation  tmp\n"
        tmp = content[0].strip().split(",")
        for att in tmp[:-1]:
            arff_title+="@attribute "+att+" numeric\n"
        arff_title+= "@attribute class {"

        labels = set()
        for i in range(1,len(content)):
            tools.show_process(i,len(content),10)
            labels.add(content[i].split(",")[-1].strip())

        labels = sorted(list(labels))
        for l in labels:
            arff_title+= l+","
        arff_title = arff_title[:-1]
        arff_title += "}\n @data\n"

    arff_string = arff_title+data
    tools.save_txt(save_path,arff_string)
    print("csv 2 arff done")
    return arff_string

def labeled_data(filepath = "../res/data/已完成标注（6818）.txt",save_file = None):
    label_no_line, no_label_line, label_data = label_content(filepath)
    if save_file !=None:
        save_label_data(label_data,save_file)
    return label_data

def get_sucheng(content):
    sucheng_regex = "\[.{0,2}原告诉称.{0,2}\]"
    common_regex = "\[.{2,8}\]"
    sucheng_index = tools.get_regex_str_index(sucheng_regex,content)
    start = 0
    if  len(sucheng_index)>0:
        start = sucheng_index[0]+content[sucheng_index[0]:].index("]")+1
    end_index = tools.get_regex_str_index(common_regex,content[start:])
    if len(end_index) >0:
        end = start+end_index[0]
    else:
        end = len(content)
    content = content[start:end]
    return content

if __name__ =="__main__":
    lnl,nll,ll= label_content()