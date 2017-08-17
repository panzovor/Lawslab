import src.tools as tools
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

min_num = 30
def read2map(file):
    content = tools.read_txt(file).strip()
    res ={}
    for line in content.split("\n"):
        tmp = line.split(",")
        if tmp[0] not in res.keys():
            res[tmp[0]] =[]
        res[tmp[0]].append(tmp[1])
    return res

def filter(key,data):
    if len(data[key]) < min_num:
        return False
    return True

###
def keywords_features(data):
    save_file = "../res/parameter/characteristic/keywords.txt"
    labels = sorted(data.keys())
    content =[]
    keys=[]
    res ={}
    for label in labels:
        if filter(label,data):
            label = label.replace(" ","")
            keys.append(label)
            words = tools.cut_sentence(label)
            content.append([label,' '.join(words)])
            res[label] = words
    tools.save_txt(save_file,content)
    return keys,res

def calculate_tfidf( corpus, keys):
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    res = {}
    for i in range(len(keys)):
        k = keys[i]
        res[k] = {}
        for j in range(len(word)):
            if weight[i][j]>0:
                res[k][word[j]] = weight[i][j]
    return res

def tfidf_features(data,save_file = "../res/parameter/tfidf/tfidf.txt"):

    keys = list(sorted(data.keys()))
    corpus =[]
    keys_ = []
    for key in keys:
        if filter(key,data):
            keys_.append(key)
            words =[]
            for line in data[key]:
                ws = tools.cut_sentence(line)
                words.extend(ws)
            corpus.append(' '.join(words))
    keys = keys_
    tfidf = calculate_tfidf(corpus,keys)
    content = []
    for key in tfidf.keys():
        for word in tfidf[key].keys():
            content.append([key, word, str(tfidf[key][word])])
    tools.save_txt(save_file, content)
    return tfidf

def probability_features(data):
    save_file = "../res/parameter/probability/probability.txt"
    res ={}
    keys = list(sorted(data.keys()))
    keys_  = []
    for key in keys:
        if filter(key,data):
            keys_.append(key)
            for line in data[key]:
                words = tools.cut_sentence(line)
                for w in words:
                    if w not in res.keys():
                        res[w]={}
                    if key not in res[w].keys():
                        res[w][key] = 0
                    res[w][key]+=1
    keys = keys_
    pos = {}
    content = []
    for w in  res.keys():
        count = 0
        for var in res[w].keys():
            count+= res[w][var]
        for key in res[w].keys():
            res[w][key]/=count
            content.append([key,w,str(res[w][key])])
            if key not in pos.keys():
                pos[key] ={}
            if w not in pos[key].keys():
                pos[key][w] =res[w][key]
    tools.save_txt(save_file,content)
    return pos

class features_extractor():
    def __init__(self):
        self.tfidf_parameter = {}
        self.keywords_parameter = {}
        self.probability_parameter = {}
        self.keys = []
        self.arff_title = ""
        self.tfidf_len_each = 4
        self.probabilit_len_each = 4
        self.all_feature_len = 4
        self.jingdu = 4
        self.train= False



    def reset(self):
        self.tfidf_parameter = {}
        self.keywords_parameter = {}
        self.probability_parameter = {}
        self.keys = []
        self.tfidf_len_each = 4
        self.probabilit_len_each = 4
        self.all_feature_len = 4
        self.jingdu = 4
        self.train = False
        self.arff_title = ""


### 每个类别的关键词比例（例如：借款人 丧失 偿还 能力，若句中出现其中3个词，则对应值为0.75）
    def get_keywords_features(self,line):
        result =[]
        count =0
        if len(self.keywords_parameter)>0:
            for key in self.keys:
                for w in self.keywords_parameter[key]:
                    if w in line:
                        count+=1
                result.append(round(count/len(self.keywords_parameter[key]),self.jingdu))
        return result

    ### 名词数，动词数，数词数，总词数
    def get_character_features(self,line):
        words, tag = tools.cut_tag_sentence(line)
        result = [0, 0, 0, 0]
        for t in tag:
            if "n" in t:
                result[0] += 1
            if "v" in t:
                result[1] += 1
            if "m" in t:
                result[2] += 1
        result[3] = len(words)
        return result

    ### 每个类别中tfidf值前  tfidf_len_each 个tfidf 值
    def get_tfidf_features(self,line):
        result = []
        if len(self.tfidf_parameter)>0:
            words = tools.cut_sentence(line)
            keys = list(sorted(self.tfidf_parameter.keys()))
            for key in keys:
                tmp = []
                for w in words:
                    if w in self.tfidf_parameter[key].keys():
                        tmp.append(round(self.tfidf_parameter[key][w],self.jingdu))
                while len(tmp)< self.tfidf_len_each:
                    tmp.append(0.0)
                tmp.sort(reverse=True)
                result.extend(tmp[:self.tfidf_len_each])
        return result

    def train_from_data(self,data):
        self.keys,self.keywords_parameter = keywords_features(data)
        # print("keywords train done")
        self.tfidf_parameter = tfidf_features(data)
        # print("tfidf train done")
        self.probability_parameter = probability_features(data)
        # print("probability train done")

        print(len(self.keys),len(self.keywords_parameter),len(self.tfidf_parameter),len(self.probability_parameter))

        self.all_feature_len += len(self.keys)
        self.all_feature_len += len(self.tfidf_parameter.keys()) * self.tfidf_len_each
        self.all_feature_len += self.probabilit_len_each * len(self.probability_parameter.keys())

        self.arff_title = "@relation tmp\n"
        for i in range(self.all_feature_len):
            self.arff_title += "@attribute att" + str(i) + " numeric\n"
        self.arff_title += "@attribute class {"
        for i in range(len(self.keys) - 1):
            self.arff_title += str(i) + ","
        self.arff_title += str(len(self.keys) - 1) + "}\n@data\n"
        tools.save_txt("../res/parameter/arff/arff_title.txt",self.arff_title)
        self.train = True


    def train(self,train_file):
        data = read2map(train_file)
        self.train_from_data(data)

    def load_parameter(self):
        self.reset()
        keywords_file = "../res/parameter/characteristic/keywords.txt"
        content = tools.read_txt(keywords_file).strip()
        for line in content.split("\n"):
            tmp = line.split(",",maxsplit=2)
            if tmp[0].strip() not in self.keywords_parameter.keys():
                try:
                    if " " in tmp[1].strip():
                        self.keywords_parameter[tmp[0].strip()] = [var.strip() for var in tmp[1].strip().split(" ")]
                    else:
                        self.keywords_parameter[tmp[0].strip()] = [tmp[1].strip()]
                except:
                    print(line,tmp)
                    input()
        self.keys.extend(list(sorted(self.keywords_parameter.keys())))
        self.all_feature_len += len(self.keys)

        tfidf_file = "../res/parameter/tfidf/tfidf.txt"
        content = tools.read_txt(tfidf_file).strip()
        for line in content.split("\n"):
            tmp =[var.strip() for var in  line.split(",",maxsplit=3)]
            if tmp[0] not in self.tfidf_parameter.keys():
                self.tfidf_parameter[tmp[0]] = {}
            if tmp[1] not in self.tfidf_parameter[tmp[0]].keys():
                self.tfidf_parameter[tmp[0]][tmp[1]] = float(tmp[2])
        self.all_feature_len+= len(self.tfidf_parameter.keys())*self.tfidf_len_each

        probability_file = "../res/parameter/probability/probability.txt"
        content = tools.read_txt(probability_file).strip().split("\n")
        for line in content:
            line = [var.strip() for var in line.strip().split(",",maxsplit=3)]
            if line[0] not in self.probability_parameter.keys():
                self.probability_parameter[line[0]] = {}
            if line[1] not in self.probability_parameter[line[0]].keys():
                self.probability_parameter[line[0]][line[1]] = float(line[2])
        self.all_feature_len+= self.probabilit_len_each* len(self.probability_parameter.keys())
        content = tools.read_txt("../res/parameter/arff/arff_title.txt")
        self.arff_title = content
        self.train = True

        print("feature extractor parameter load done")
    ### 每个类别中,句中词语概率值前 probabilit_len_each 个概率值
    ### 例如（a类：[0.1,0.2,0.3,0.4,0.5], b类：[0.11,0.22,0.33,0.44]）-> [0.5,0.4,0.3,0.2,0.44,0.33,0.22,0.11]
    def get_probability_features(self,line):
        result =[]
        if len(self.probability_parameter)>0:
            words = tools.cut_sentence(line)
            keys = list(sorted(self.probability_parameter.keys()))

            for key in keys:
                tmp =[]
                for w in words:
                    if w in self.probability_parameter[key].keys():
                        tmp.append(round(self.probability_parameter[key][w],self.jingdu))
                while len(tmp)< self.probabilit_len_each:
                    tmp.append(0.0)
                result.extend(tmp[:self.probabilit_len_each])
        return result
    def get_features(self,line):
        key_fea = self.get_keywords_features(line)
        tfidf_fea = self.get_tfidf_features(line)
        pro_fea = self.get_probability_features(line)
        cha_fea = self.get_character_features(line)
        res = key_fea+tfidf_fea+pro_fea+cha_fea
        if len(res)!=self.all_feature_len:
            return None
        return res
    def get_label(self,label):
        if label in self.keys:
            return self.keys.index(label)
        else:
            return -1
    def get_label_name(self,index):
        if index>=0  and index< len(self.keys):
            return self.keys[index]
        else:
            return "not in the labels"
    def featureslize(self,sen,label):
        return self.get_features(sen),self.get_label(label)

    def feature_labeled_data(self,labeled_data,save_root = None):
        if not self.train:
            self.train_from_data(labeled_data)
        content = []
        for label in self.keys:
            if label in labeled_data.keys():
                for line in labeled_data[label]:
                    tmp  = self.get_features(line)
                    content.append([line]+list(map(str,tmp))+[str(self.keys.index(label))])
        if save_root!=None:
            tools.save_txt(save_root,content)
        return content

    def arff_featured_data(self,featured_data,arff_file):
        arff_string = ""
        data =[]
        for line in featured_data:
            data.append(','.join(line[1:]))
        arff_string+=self.arff_title
        arff_string+= '\n'.join(data)
        tools.save_txt(arff_file,arff_string)


if __name__ == "__main__":
    train_file = "../res/seperated_data/train_label.csv"
    # train(train_file)
    test_string = "被告黄浩、被告金小伟也均未按承诺履行连带清偿责任义务"
    fea = features_extractor()
    # fea.load_parameter()
    import time
    start = time.time()
    # for i in range(10000):
    features = fea.get_features(test_string)
    end = time.time()
    print(len(features))
    print(end-start)

