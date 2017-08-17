import src.tools as tools
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx

jingdu =  5
### 统计特征（名词数，动词数，数词数，所有词数，关键词数）
class characteristic_features():
    def __init__(self):
        self.parameter_file ="../res/parameter/characteristic/keywords.txt"
        self.keywords ={}

    def load_parameter(self):
        self.keywords= set(tools.read_txt(self.parameter_file).split("\n"))


    def get_features(self,sentence):
        words,tag = tools.cut_tag_sentence(sentence)
        result = [0,0,0,0,0]
        for t in tag:
            if "n" in t :
                result[0]+=1
            if "v" in t:
                result[1]+=1
            if "m" in t:
                result[2]+=1
        result[3] = len(words)
        for w in words:
            if w in self.keywords:
                result[4]+=1
        return result

### tfidf特征
class tfidf_features():
    def __init__(self):
        self.parameter_file ="../res/parameter/tfidf/tfidf.txt"
        self.tfidf={}

        self.labels = []
        self.feature_size = 4
        self.load_parameter()

    ## trainfile: 存储训练文件的地址
    ## 格式如下：
    ## label,...,sentence
    def train(self,train_file):
        content = tools.read_txt(train_file).strip().split("\n")
        data = {}

        for line in content:
            line = line.split(",",maxsplit=3)
            if line[0] not in data.keys():
                data[line[0]] = []
            words = tools.cut_sentence(line[-1])
            data[line[0]].extend(words)
        for key in data.keys():
            data[key] = ' '.join(data[key])
        keys = sorted(data.keys())
        corpus =[ ]
        for k in keys:
            corpus.append(data[k])
        self.tfidf = self.calculate_tfidf(corpus,keys)
        content =[]
        for key in self.tfidf.keys():
            for word in self.tfidf[key].keys():
                content.append([key,word,str(self.tfidf[key][word])])
        tools.save_txt(self.parameter_file,content)
        self.labels  = sorted(self.tfidf.keys())


    def calculate_tfidf(self,corpus,keys):
        vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
        tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
        weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        res= {}
        for i in range(len(keys)):
            k = keys[i]
            res[k] = {}
            for j in range(len(word)):
                res[k][word[j]] = weight[i][j]
        return res

    def load_parameter(self):
        self.tfidf = tools.read_into_map(self.parameter_file)
        # print([key+"-"+str(value) for key,value in self.tfidf["原告被告是夫妻关系"].items() if float(value)>0])
        self.labels = sorted(self.tfidf.keys())

    def get_features(self,sentence):
        words = tools.cut_sentence(sentence)
        res = []
        for l in self.labels:
            tmp =[]
            for w in words:
                if w in self.tfidf[l].keys():
                    tmp.append(round(self.tfidf[l][w],jingdu))
            while len(tmp)< self.feature_size:
                tmp.append(0.0)
            tmp.sort()
            res.extend(tmp[:self.feature_size])
        return res


class textrank_features():
    def __init__(self):
        self.feature_size = 4
        self.parameter_file = "../res/parameter/textrank/rank_score.txt"
        self.rank_scores ={}
        self.labels = []
        self.load_parameter()
    ## input : corpus:【
    ## [word11,...,words1n]
    ##   ,,,
    ## [wordn1,...,wordsnn]
    ## 】
    ## output: word_map,word_map_reverse,matrix
    ## word_map{word_1:index_1, ...,  word_n:index_n]
    ## word_map_reverse{index_1:word1, ... , index_n: word_n}
    ## matrix:[(start_node_1,end_node_1,weight_1),...,(start_node_n,end_node_n,weight_n)]
    ## ]
    def transfer2matrix(self,corpus):
        word_map={}
        word_map_reverse ={}
        matrix = {}
        for line in corpus:
            for i in range(len(line)):
                w = line[i]
                if w not in word_map.keys():
                    word_map[w] = len(word_map)
                    word_map_reverse[len(word_map_reverse)] = w
            for i in range(len(line)):
                for j in range(i+1,len(line)):
                    if word_map[line[i]] not in matrix.keys():
                        matrix[word_map[line[i]]]={}
                    if word_map[line[j]] not in matrix[word_map[line[i]]].keys():
                        matrix[word_map[line[i]]][word_map[line[j]]] = 0.0
                    if word_map[line[j]] not in matrix.keys():
                        matrix[word_map[line[j]]] = {}
                    if word_map[line[i]] not in matrix[word_map[line[j]]].keys():
                        matrix[word_map[line[j]]][word_map[line[i]]] = 0.0
                    matrix[word_map[line[i]]][word_map[line[j]]] +=1/abs(j-i)
                    matrix[word_map[line[j]]][word_map[line[i]]] += 1 / abs(j - i)
        return word_map,word_map_reverse,matrix

    def textrank(self,matrix):

        edges=[]
        for start in matrix.keys():
            for end in matrix[start].keys():
                edges.append((start,end,matrix[start][end]))

        g = nx.DiGraph()
        g.add_weighted_edges_from(edges)
        pr = nx.pagerank_numpy(g)
        return pr

    ## trainfile: 存储训练文件的地址
    ## 格式如下：
    ## label,...,sentence
    def train(self,train_file):
        content = tools.read_txt(train_file).strip().split("\n")
        data = {}
        for line in content:
            line = line.split(",")
            if line[0] not in data.keys():
                data[line[0]] = []
            words = tools.cut_sentence(line[-1])
            data[line[0]].append(words)
        self.labels = sorted(data.keys())
        content = []
        for l in self.labels:
            word_map,word_map_reverse,matrix = self.transfer2matrix(data[l])
            pr= self.textrank(matrix)
            if l not in self.rank_scores.keys():
                self.rank_scores[l]={}
            for key,value in pr.items():
                self.rank_scores[l][word_map_reverse[key]]= value
                content.append([l,word_map_reverse[key],str(value)])
        tools.save_txt(self.parameter_file, content)

    def load_parameter(self):
        self.rank_scores = tools.read_into_map(self.parameter_file)
        self.labels = sorted(self.rank_scores.keys())

    def get_features(self,sentence):
        words= tools.cut_sentence(sentence)
        res =[]
        for l in self.labels:
            tmp =[]
            for w in words:
                if w in self.rank_scores[l].keys():
                    tmp.append(round(self.rank_scores[l][w],jingdu))
            while len(tmp)< self.feature_size:
                tmp.append(0.0)
            res.extend(tmp[:self.feature_size])
        return res

class feature_extractor():
    def __init__(self):
        self.cha = characteristic_features()
        self.tfi = tfidf_features()
        self.tr = textrank_features()
        self.feature_size= 5+self.tfi.feature_size*len(self.tfi.labels)+self.tr.feature_size*len(self.tr.labels)

    ## trainfile: 存储训练文件的地址
    ## 格式如下：
    ## label,...,sentence
    def train(self,train_file):
        self.tfi.train(train_file)
        self.tr.train(train_file)
        self.feature_size = 5+self.tfi.feature_size*len(self.tfi.labels)+self.tr.feature_size*len(self.tr.labels)

    def get_features(self,sentence):
        cha_features =self.cha.get_features(sentence)
        tfi_features =self.tfi.get_features(sentence)
        tr_features =self.tr.get_features(sentence)
        # print(cha_features)
        # print(sum(tfi_features),tfi_features)
        # print(sum(tr_features),tr_features)
        return cha_features+tfi_features+tr_features

if __name__=="__main__":
    train_file = "../res/seperated_data/train_label.csv"
    fe = feature_extractor()
    fe.train(train_file)
    smaple ="2008年4月24日，原告与被告包平签订借款合同一份"
    features = fe.get_features(smaple)
    print(features)