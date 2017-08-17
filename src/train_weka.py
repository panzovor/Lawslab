__author__ = 'E440'
import os
import src.tools as tools
import subprocess
import re
from nltk.internals import java,config_java
config_java()
project_dir = os.path.abspath("../").replace("\\","/")
javahome = os.environ.get('JAVA_HOME')
if ";" in javahome:
    javahome = javahome.split(";")[0]
java_path=javahome+"\\bin\\java.exe"
weka_path=project_dir+"/res/parameter/weka3-6-6.jar"
_cmd =  [java_path,"-cp", weka_path]


def execCmd(cmd):
   sub=subprocess.Popen(cmd,cwd=project_dir,shell=True,stdout=subprocess.PIPE)
   stdout,stderror = sub.communicate()
   return stdout

train_parameter ={
    "RandomForest":"weka.classifiers.trees.RandomForest -I 10 -K 0 -S 1",
    "NaiveBayes":"weka.classifiers.bayes.NaiveBayes",
    "RBF":"weka.classifiers.functions.RBFNetwork -B 2 -S 1 -R 1.0E-8 -M -1 -W 0.1",
    "SMO":"weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\"",
    "MultilayerPerceptron":"weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a",
    "BayesNet":"weka.classifiers.bayes.BayesNet -D -Q weka.classifiers.bayes.net.search.local.K2 -- -P 1 -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5"
}

predict_parameter ={
    "RandomForest": "weka.classifiers.trees.RandomForest",
    "NaiveBayes": "weka.classifiers.bayes.NaiveBayes",
    "RBF": "weka.classifiers.functions.RBFNetwork",
    "SMO": "weka.classifiers.functions.SMO",
    "MultilayerPerceptron": "weka.classifiers.functions.MultilayerPerceptron",
    "BayesNet": "weka.classifiers.bayes.BayesNet"

}

def analyze(weka_result):
    matrix_start = 0
    matrix_end = len(weka_result)
    matrix_regex = "<-- classified as"
    if matrix_regex in weka_result:
        matrix_start = weka_result.rindex(matrix_regex) + len(matrix_regex)
    matrix_string = weka_result[matrix_start:matrix_end].strip().replace("|", "")
    matrix_strings = matrix_string.split("\n")
    matrix = []
    # print(matrix_string)
    for line in matrix_strings:
        line = re.split(" +", line.strip())
        matrix.append([map_label(line[-1])] + list(map(int, line[:-3])))

    print(weka_result)

    report_start = "TP Rate   FP Rate   Precision   Recall  F-Measure   ROC Area  Class"
    report_end = "Weighted Avg"
    head = "                 "
    split_regex =" +"
    start = 0
    end =len(weka_result)
    if report_start in weka_result:
        start = weka_result.rindex(report_start)+len(report_start)
    if report_end in weka_result:
        end = weka_result.rindex(report_end)
    report = weka_result[start:end].replace(head,"").strip()
    print(report)
    reports = report.split("\n")
    report = []
    for i in range(len(reports)):
        line = reports[i]
        line = re.split(split_regex,line)
        report.append([map_label(line[-1]),line[2],line[3],line[4],str(sum(matrix[i][1:]))])
    return report,matrix


def predict(model_name,model_file, test_file):
    if model_name in predict_parameter.keys():
        if not os.path.exists(model_file):
            print("model file doesnot exist")
            return None
        if not os.path.exists(test_file):
            print("test file doesnot exist")
            return None
        try:
            content = tools.read_lines(test_file)
            class_index =0
            for line in content:
                if "@attribute" in line:
                    class_index+=1
                if "@data" in line:
                    break
            cmd = _cmd+[predict_parameter[model_name] ,"-p",str(class_index),"-l",model_file,"-T",test_file]
            result = execCmd(cmd)
            return str(result,"utf-8")
        except Exception as e :
            print("check your model file or test file")
    else:
        print("wrong model name")
        return  None

def analyze_predict_result(result):
    result = result.strip()
    predict_start = "prediction ()"
    start  = result.index(predict_start)+len(predict_start)
    content = result[start:].strip()
    predict,possib = [],[]
    for line in content.split("\n"):
        line = re.split(" +",line.strip())
        predict.append(map_label(line[2][line[2].index(":")+1:]))
        possib.append(line[-1])
    return predict,possib

def analyze_train_result(result):
    result_start = "classified as"
    labels_,matrix =[],[]
    if result_start in result:
        start = result.rindex(result_start)+len(result_start)
        content = result[start:].strip()
        content = content.replace("|","").split("\n")
        for line in content:
            line = re.split(" +",line.strip())
            labels_.append(line[-1])
            matrix.append([int(var) for var in line[:-3]])

    # for rddd in matrix:
    #     print(rddd)
    precision,recall,f_score,sample_num =[],[],[],[]
    for i in range(len(matrix)):
        pre_ = sum([matrix[j][i] for j in range(len(matrix))])
        precision.append(matrix[i][i]/pre_ if pre_>0 else 0)
        rec_ = sum([matrix[i][j] for j in range(len(matrix))])
        sample_num.append(rec_)
        recall.append(matrix[i][i]/rec_ if rec_ >0 else 0)
        f_score.append(precision[-1]*2*recall[-1]/(precision[-1]+recall[-1]) if precision[-1]+recall[-1]>0 else 0)
    res ={}
    for i in range(len(labels_)):
        if int(labels_[i]) not in res.keys():
            res[int(labels_[i])]=[precision[i],recall[i],f_score[i],sample_num[i]]
    return res

def train(model_name,save_file,train_file):
    # print(save_file)
    if ".." in train_file:
        train_file=train_file.replace("..",project_dir)
    if ".." in save_file:
        save_file = save_file.replace("..",project_dir)

    if model_name in train_parameter:
        if not os.path.exists(train_file):
            print("train file doesnot exist")
            return None
        # try:
        tmp = train_parameter[model_name].split(" ")
        cmd = _cmd+tmp+["-t",train_file,"-d",save_file]
        # print(cmd)
        result = execCmd(cmd)
        return str(result,"utf-8")
        # except:
        #     print("wrong trainfile")
    else:
        print("wrong model name")

def train_model(train_file,save_root):
    models_names = ['MultilayerPerceptron', 'RBF', 'NaiveBayes', 'RandomForest', 'SMO', 'BayesNet']
    for name in models_names:
        if name in train_parameter.keys():
            train(name,save_root+name+".model",train_file)
    return save_root

def test_model(models_root,test_file,save_root):

    models_names = ['MultilayerPerceptron', 'RBF', 'NaiveBayes', 'RandomForest', 'SMO', 'BayesNet']
    # test_file = "../res/train_featuredata/"+test_name+".arff_data"
    for name in models_names:
        save_file = save_root+name+".txt"
        model_file = models_root+name+".model"
        result =predict(name,model_file,test_file)
        tools.save_txt(result,save_file)
    # return save_root

if __name__ == "__main__":
    filepath = "../res/model/model_result/BayesNet"
    content = tools.read_txt(filepath)
    # print(content)
    report,matrix = analyze(content)

    for line in report:
        print('\t'.join(line))
    for line in matrix:
        print(line)