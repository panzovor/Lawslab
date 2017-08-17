__author__ = 'E440'
import src.preprocess as pre
import os
import src.tools as tools
import src.train_weka as wekatrain

project_dir = os.path.abspath("../").replace("\\","/")


### 重新训练特征提取器
### filepath: 标注的原文件
### train_file: 转换后的用于训练特征提取器的文件保存路径
def train_feature_extracter(filepath):
    train_file = "../res/parameter/label_data.csv"
    pre.label_process(filepath,train_file)
    pre.fe.train(train_file=train_file)
    pre.fe.load_parameter()

### 训练模型，保存模型，并返回训练效果
### modelname：模型简称及其对应的模型参数
    '''
    train_parameter ={
    "RandomForest":"weka.classifiers.trees.RandomForest -I 10 -K 0 -S 1",
    "NaiveBayes":"weka.classifiers.bayes.NaiveBayes",
    "RBF":"weka.classifiers.functions.RBFNetwork -B 2 -S 1 -R 1.0E-8 -M -1 -W 0.1",
    "SMO":"weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\"",
    "MultilayerPerceptron":"weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a",
    "BayesNet":"weka.classifiers.bayes.BayesNet -D -Q weka.classifiers.bayes.net.search.local.K2 -- -P 1 -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5"
    }
    '''
### model_path: 模型保存路径
### arff_file: 训练文件
### return :
### 类别：[准确率，召回率，f值，样本数]
def train_model_(modelname,model_path =project_dir+"/res/weka/model/RandomForest_less.model",arff_file=project_dir+"/res/weka/train_less.arff"):
    result = wekatrain.train(modelname,model_path ,arff_file)
    tools.save_txt("../res/weka/model/result.txt", result)
    report = wekatrain.analyze_train_result(result)

    return report

def train(modelname,model_path =project_dir+"/res/weka/model/RandomForest_less.model",filepath="../res/data/已完成标注（6818）.txt"):
    ### train feature_extractor
    print("training feature extractor")
    train_feature_extracter(filepath)
    print("feauter extractor train done")

    ### prepare model train file
    print("prepareing trian data")
    labeled_data = pre.labeled_data(filepath)
    featured_data = pre.featured_data(labeled_data)
    arff_file = "../res/weka/train.arff"
    labels,arff_string = pre.csv2arff_file(featured_data, arff_file)

    labels_file = "../res/parameter/labels/label.txt"
    tools.save_txt(labels_file,labels)
    arff_title_path = "../res/parameter/arff/arff_title.txt"
    tools.save_txt(arff_title_path,content=arff_string[arff_string.index("@data\n")+len("@data\n")])

    ### train model
    print("training model")
    res = train_model_(modelname, model_path, arff_file)
    print("train done")
    return res

### modelname:使用的模型类别（可用值：RandomForest，NaiveBayes，RBF，SMO，MultilayerPerceptron，BayesNet）
### model_path: modelname 所对应的模型文件保存路径
### filepath : 待分类文件
def predict(modelname,model_path,filepath):
    arff_file = project_dir+"/res/tmp/predict.arff"

    label_file = "../res/tmp/labeled_data.csv"
    pre.label_process(filepath, label_file)

    feature_file = "../res/tmp/feature_data.csv"
    sentences = pre.transfer_train(label_file, feature_file)
    arff_title_path = "../res/parameter/arff/arff_title.txt"
    arff_title = tools.read_txt(arff_title_path)
    pre.csv2arff(feature_file, arff_file,arff_title=arff_title)

    result = wekatrain.predict(modelname,model_path,arff_file)
    predict_result,poo = wekatrain.analyze_predict_result(result)
    res = []
    print(len(sentences),len(predict_result),len(poo))

    ll = [var[0] for var in sentences]
    print(len(set(ll)))
    print(set(ll))
    for i in range(len(sentences)):
        res.append([sentences[i],predict_result[i],poo[i]])
    return  res

if __name__ == "__main__":
    filepath = "../res/data/已完成标注（6818）.txt"
    model_path = project_dir + "/res/weka/model/RandomForest_less.model"

    ### train
    report = train("RandomForest",model_path,filepath)
    for key in report.keys():
        print(key, report[key])

    ### predict
    # result = predict("RandomForest",model_path,filepath)
    # for line in result:
    #     print(line)


