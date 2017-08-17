import src.tools as tools
import os

class classifier():

    def __init__(self):
        self.pp = 0
        self.pn = 0
        self.nn =0
        self.np = 0
        self.precision_p =0
        self.recall_p =0
        self.fscore_p = 0

        self.precision_n = 0
        self.recall_n = 0
        self.fscore_n = 0

    def fill(self,content):
        lines = content.split("\n")
        if "," in lines[0]:
            tmp = lines[0].split(",")
            if len(tmp) == 4:
                self.pp = int(tmp[3])
                self.np = int(tmp[2])
                self.pn = int(tmp[1])
                self.nn = int(tmp[0])
                self.calcluate()
            else:
                print("wrong format")

    def calcluate(self):
        if self.pp+self.np!= 0 :
            self.recall_p = self.pp/(self.pp+self.np)
        if self.pp+self.pn!= 0:
            self.precision_p = self.pp/(self.pp+self.pn)
        if  self.precision_p+self.recall_p!=0:
            self.fscore_p = 2*self.precision_p*self.recall_p/(self.precision_p+self.recall_p)

        if self.nn+self.pn!= 0:
            self.recall_n = self.nn/(self.nn+self.pn)
        if self.nn+self.np!= 0:
            self.recall_n = self.nn/(self.nn+self.np)
        if self.precision_n+self.recall_n!=0:
            self.fscore_n = 2*self.precision_n*self.recall_n/(self.precision_n+self.recall_n)

    def get_list(self):
        return [self.pp,self.np,self.nn,self.pn,self.precision_p,self.recall_p,self.fscore_p,self.precision_n,self.recall_n,self.fscore_n]


def load_all_result():
    result_root = "../res/model/"
    filelist = os.listdir(result_root)
    result ={}
    for name in filelist:
        if name not in result.keys():
            result[name] = {}
        root = result_root+name+"/"
        filenames =[fname for fname in os.listdir(root) if fname[-4:] == ".txt"]
        for fname in filenames:
            if fname not in result[name].keys():
                content = tools.read(root+fname).strip().split("\n")
                result[name][fname] = content[2]+","+content[1]
    return result

def select_best(data,pos_fscore=True):
    result = []
    for name in data.keys():
        tmp =None
        max_v = 0
        for classname in data[name].keys():
            if float(data[name][classname].split(",")[3])>max_v:
                max_v = float(data[name][classname].split(",")[3])
                tmp =[classname,data[name][classname]]

        result.append([name]+tmp)

    content = [["关键词","模型","类别","准确率","召回率","f-值","样本数","类别","准确率","召回率","f-值","样本数"]]
    for line in result:
        tmp = ','.join(line)
        tmp = tmp.replace(".txt","")
        content.append(tmp.split(","))

    filepath = "../res/result.csv"
    tools.save_csv(content,filepath)

if __name__ == "__main__":
    data = load_all_result()
    res = select_best(data)