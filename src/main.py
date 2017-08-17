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
    train_file = project_dir+"/res/data/����ɱ�ע��6818��.txt"
    model_dir = project_dir+"/res/model/"
    cls = classify(model_name,model_dir)
    cls.train(train_file)
    cls.save_model()

    cls.load_model()
    text = u'��������ɽ������Ժ����2009����������ֵ�33�š�ԭ���ߡ�����ί�д�������������������������ԭ���ߡ���Ϊ�뱻�����������������һ������2008��12��11����Ժ���ߣ���Ժ��ͬ�����������������Ա�ſ��ֶ������У���2009��1��4�չ�����ͥ��������������ͥ�����о���ԭ��ί�д�������������ͥ�μ������ϡ���������������Ժ�Ϸ�����������������δ��ͥ���������������սᡣ[ԭ���߳�]��ԭ���߳ƣ�2008��10��1�գ�������Ӫ��Ҫ��ԭ����20��Ԫ�������߽���1�ݣ�Լ��������޴�2008��10��1����2008��10��30�ա������ÿ������δ�����ġ�������Ҫ�󱻸������黹���20��Ԫ����֧���ÿ��2008��11��1����ʵ��֧����ֹ����Ϣ��ʧ���������֮����һ��׼���㣩����[�����׳�]������δ����硣ԭ��Ϊ֤�������ţ��ṩ�˱�����ߵĽ���1�ݣ���֤��������ԭ����20��Ԫ����ʵ������δ��ͥ��֤����Ϊ��������Ȩ����Ժ��Ϊ����֤����ʵ���Ϸ����뱾�����й����ԣ��������϶�����������Ժ�����İ�����ʵ��ԭ����߳���һ�¡���Ժ��Ϊ��ԭ������֮��Ľ����ϵ������������Ч������δ���ڻ��Ӧ�е���Ӧ���������Ρ����澭��Ժ�Ϸ�����������������δ��ͥ����Ϊ��ԭ��������ʵ���������󿹱�Ȩ�ķ������ݴˣ����л����񹲺͹���ͨ�򡷵�һ����ʮ�������л����񹲺͹���ͬ�����ڶ������������ڶ����������Ĺ涨���о����£��������������ڱ��о���Ч��10���ڷ���ԭ���ߡ������20��Ԫ����֧���ÿ��2008��11��1����ʵ��֧����ֹ����Ϣ��ʧ���������֮����һ��׼���㣩�����δ�����о�ָ�����ڼ����и�����Ǯ����Ӧ�����ա��л����񹲺͹��������Ϸ����ڶ��ٶ�ʮ����֮�涨���ӱ�֧�����������ڼ��ծ����Ϣ�����������4238Ԫ��������ȡ2169Ԫ���ɱ����������������粻�����о��������о����ʹ�֮����ʮ�����ڣ���Ժ�ݽ�����״�������Է������˵���������������������㽭ʡ�������м�����Ժ�����������м�����ԺԤ�����߰��������4238Ԫ���������У��������к����������ʺţ�1202024409008802968���������㽭ʡ�������м�����Ժ�����ԲƲ������������ߵģ���������Ѱ��ղ���һ���о����ֵ���������Ԥ���������������Ĵ�������������δ���ɵģ����Զ��������ߴ�������Ա�ſ��֡�����������һ�����ա����Ա���ࡣ��������:2����������:2009-01-04����������:�ߡ��������������������һ�������о��顣����ID:90593351-e3ef-495e-bcb7-7b6ec2622c4b�����г���:һ�󡣰���:��2009����������ֵ�33�š���Ժ����:��������ɽ������Ժ��'
    res = cls.predict(text)
    for line in res:
        print(line)
