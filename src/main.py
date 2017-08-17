# coding=gbk
import src.preprocess as pre
from src.features import features_extractor
import src.train_weka as trainweka
import os
import pickle
import re

project_dir = os.path.abspath("../").replace("\\","/")

class classify():
    def __init__(self,model_name,model_dir):
        self.model_name = model_name
        self.model_path = model_dir+self.model_name if model_dir[-1] == "/" else model_dir+"/"+self.model_name+".model"
        self.feat_path = model_dir+"feature.pkl" if model_dir[-1] == "/" else model_dir+"/feature.pkl"
        self.tmp_arff = model_dir+"tmp.arff" if model_dir[-1] == "/" else model_dir+"/tmp.arff"
        self.fea = features_extractor()

    def train(self, train_file):
        labeld_data = pre.labeled_data(train_file)
        print("labeled done")
        feature_data = self.fea.feature_labeled_data(labeld_data)
        print("feature done")
        self.fea.arff_featured_data(feature_data, self.tmp_arff)
        print("arff done")
        result = trainweka.train(self.model_name, self.model_path, self.tmp_arff)
        res = trainweka.analyze_train_result(result)
        for label in res:
            print(label, self.fea.get_label_name(label), res[label])
        print("train done")

    def save_model(self):
        feature_data = {
            "tfidf":self.fea.tfidf_parameter,
            "keywords":self.fea.keywords_parameter ,
            "probability":self.fea.probability_parameter,
            "keys":self.fea.keys,
            "arff_title":self.fea.arff_title ,
            "tfidf_len":self.fea.tfidf_len_each ,
            "prob_len":self.fea.probabilit_len_each ,
            "all_featuer_len":self.fea.all_feature_len,
            "jingdu":self.fea.jingdu ,
            "train":self.fea.train
        }
        pickle.dump(feature_data,open(self.feat_path,"wb"))

    def load_model(self):
        feature_data = pickle.load(open(self.feat_path,"rb"))
        self.fea.tfidf_parameter = feature_data["tfidf"]
        self.fea.keywords_parameter = feature_data["keywords"]
        self.fea.probability_parameter = feature_data["probability"]
        self.fea.keys = feature_data["keys"]
        self.fea.arff_title = feature_data["arff_title"]
        self.fea.tfidf_len_each = feature_data["tfidf_len"]
        self.fea.probabilit_len_each = feature_data["prob_len"]
        self.fea.all_feature_len = feature_data["all_featuer_len"]
        self.fea.jingdu = feature_data["jingdu"]
        self.fea.train = feature_data["train"]



    def analyze_predict_result (self,feature_data,result):
        result = result.strip()
        predict_start = "prediction ()"
        start = result.index(predict_start) + len(predict_start)
        content = result[start:].strip()
        res =[]
        content = content.split("\n")
        for i in range(len(content)):
            line = content[i].replace("+","")
            line = re.split(" +", line.strip())
            pre_label = int(line[2][line[2].index(":") + 1:].strip())
            pre = self.fea.get_label_name(pre_label)
            res.append([feature_data[i][0],pre,line[-1]])
        return res

    def predict(self,text):
        labeled_data = pre.label_content_data(text)
        feature_data = self.fea.feature_labeled_data(labeled_data)
        self.fea.arff_featured_data(feature_data,self.tmp_arff)
        result = trainweka.predict(self.model_name,self.model_path,self.tmp_arff)
        return self.analyze_predict_result(feature_data,result)

if __name__=="__main__":
    print(project_dir)
    model_name ="RandomForest"
    train_file = project_dir+"/res/data/已完成标注（6818）.txt"
    model_dir = project_dir+"/res/model/"
    cls = classify(model_name,model_dir)
    cls.train(train_file)
    cls.save_model()

    cls.load_model()
    text = u'杭州市萧山区人民法院。（2009）萧民二初字第33号。原告倪××。委托代理人王××。被告汪××。原告倪××为与被告汪××民间借贷纠纷一案，于2008年12月11日向本院起诉，本院于同日受理后，依法由审判员张可乐独任审判，于2009年1月4日公开开庭进行了审理，并当庭宣告判决。原告委托代理人王××到庭参加了诉讼。被告汪××经本院合法传唤，无正当理由未到庭。本案现已审理终结。[原告诉称]。原告诉称：2008年10月1日，被告因经营需要向原告借款20万元，并出具借条1份，约定借款期限从2008年10月1日至2008年10月30日。被告借得款后，至今未还分文。现起诉要求被告立即归还借款20万元，并支付该款从2008年11月1日至实际支付日止的利息损失（按日万分之二点一标准计算）。。[被告俗称]。被告未作答辩。原告为证明其主张，提供了被告出具的借条1份，欲证明被告向原告借款20万元的事实。被告未到庭质证，视为放弃抗辩权。本院认为，该证据真实、合法、与本案具有关联性，故予以认定。经审理，本院查明的案件事实与原告的诉称相一致。本院认为，原、被告之间的借贷关系依法成立并有效。被告未按期还款，应承担相应的民事责任。被告经本院合法传唤，无正当理由未到庭，视为对原告所诉事实及诉讼请求抗辩权的放弃。据此，《中华人民共和国民法通则》第一百三十条、《中华人民共和国合同法》第二百零六条、第二百零七条的规定，判决如下：。被告汪××在本判决生效后10日内返还原告倪××借款20万元，并支付该款从2008年11月1日至实际支付日止的利息损失（按日万分之二点一标准计算）。如果未按本判决指定的期间履行给付金钱义务，应当依照《中华人民共和国民事诉讼法》第二百二十九条之规定，加倍支付迟延履行期间的债务利息。案件受理费4238元，减半收取2169元，由被告汪××负担。如不服本判决，可在判决书送达之日起十五日内，向本院递交上诉状，并按对方当事人的人数提出副本，上诉于浙江省杭州市中级人民法院，并向杭州市中级人民法院预交上诉案件受理费4238元（开户银行：工商银行湖滨分理处，帐号：1202024409008802968，户名：浙江省杭州市中级人民法院）。对财产案件提起上诉的，案件受理费按照不服一审判决部分的上诉请求预交。在上诉期满的次日起七日内仍未交纳的，按自动撤回上诉处理。审判员张可乐。二〇〇九年一月四日。书记员陈燕。案件类型:2。裁判日期:2009-01-04。案件名称:倪××与汪××民间借贷纠纷一审民事判决书。文书ID:90593351-e3ef-495e-bcb7-7b6ec2622c4b。审判程序:一审。案号:（2009）萧民二初字第33号。法院名称:杭州市萧山区人民法院。'
    res = cls.predict(text)
    for line in res:
        print(line)
