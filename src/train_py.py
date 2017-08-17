from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import metrics
from src.features import features_extractor
import re
import src.tools as tools
fea = features_extractor()


def load_data(file,featurelize = True,skip_first = False):
    lines = tools.read_lines(file)

    x,y =[],[]
    for i in range(len(lines)):
        if skip_first and i ==0 :
            continue
        line= lines[i]
        tools.show_process(i,len(lines))
        tmp = line.split(",",maxsplit=2)
        if featurelize:
            tt= fea.get_features(tmp[1])
            x.append(tt)
        else:
            x.append(tmp[1])
        # print(tmp[1],len(tt))
        y.append(fea.get_label(tmp[0]))
    print()
    return  x,y


def svm_classify(X_train, y_train, X_test=None, y_test=None):
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    if X_train!=None and y_test!=None:
        pre_y_test = clf.predict(X_test)
        report,confuse=analyse_predict_result(y_test,pre_y_test)
    else:
        report,confuse = analyse_predict_result(y_train,pre_y_train)
    return clf,report,confuse

def rf_classify(X_train, y_train, X_test=None, y_test=None):
    clf = RandomForestClassifier(random_state=0, n_estimators=100)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    if X_train!=None and y_test!=None:
        pre_y_test = clf.predict(X_test)
        report,confuse=analyse_predict_result(y_test,pre_y_test)
    else:
        report,confuse = analyse_predict_result(y_train,pre_y_train)
    return clf,report,confuse

def knn_classify(X_train, y_train,  X_test=None, y_test=None):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    if X_train!=None and y_test!=None:
        pre_y_test = clf.predict(X_test)
        report,confuse=analyse_predict_result(y_test,pre_y_test)
    else:
        report,confuse = analyse_predict_result(y_train,pre_y_train)
    return clf,report,confuse

def bagging_knn_classify(X_train, y_train,  X_test=None, y_test=None):


    clf = BaggingClassifier(KNeighborsClassifier(),
                            max_samples=0.5, max_features=0.5)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    if X_train!=None and y_test!=None:
        pre_y_test = clf.predict(X_test)
        report,confuse=analyse_predict_result(y_test,pre_y_test)
    else:
        report,confuse = analyse_predict_result(y_train,pre_y_train)
    return clf,report,confuse

def lr_classify(X_train, y_train,  X_test=None, y_test=None):

    clf = LogisticRegression(C=1e5)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    if X_train!=None and y_test!=None:
        pre_y_test = clf.predict(X_test)
        report,confuse=analyse_predict_result(y_test,pre_y_test)
    else:
        report,confuse = analyse_predict_result(y_train,pre_y_train)
    return clf,report,confuse

def nb_classify(X_train, y_train,  X_test=None, y_test=None):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    if X_train != None and y_test != None:
        pre_y_test = clf.predict(X_test)
        report, confuse = analyse_predict_result(y_test, pre_y_test)
    else:
        report, confuse = analyse_predict_result(y_train, pre_y_train)
    return clf,report,confuse

def da_classify(X_train, y_train,  X_test=None, y_test=None):

    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    pre_y_test = clf.predict(X_test)
    report,confuse=analyse_predict_result(y_test,pre_y_test)
    return clf,report,confuse

def decisionTree_classify(X_train, y_train,  X_test=None, y_test=None):
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    if X_train != None and y_test != None:
        pre_y_test = clf.predict(X_test)
        report, confuse = analyse_predict_result(y_test, pre_y_test)
    else:
        report, confuse = analyse_predict_result(y_train, pre_y_train)
    return clf,report,confuse

def GBDT_classify(X_train, y_train,  X_test=None, y_test=None):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    if X_train != None and y_test != None:
        pre_y_test = clf.predict(X_test)
        report, confuse = analyse_predict_result(y_test, pre_y_test)
    else:
        report, confuse = analyse_predict_result(y_train, pre_y_train)
    return clf,report,confuse

def save_model(cls,save_path):
    joblib.dump(cls,save_path,compress=3)

def load_model(model_path):
    return joblib.load(model_path)


models = {
    "randomforest":rf_classify,
    "svm":svm_classify,
    "naivebayes":nb_classify,
    # "nn":nn_classify
}


def train_model(train_file,test_file,model_name,model_save_path=None,res_save_path =None):
    # print("transfter train file: ",train_file)
    train_x,train_y = load_data(train_file)

    if test_file != None:
        # print(test_file)
        test_x,test_y = load_data(test_file)
    else:
        test_x = None
        test_y = None
    # print(train_file+" transfter done\ntraining model")
    return train_model_(train_x,train_y,test_x,test_y,model_name,model_save_path,res_save_path)


def train_model_(train_x,train_y,test_x,test_y,model_name,model_save_path=None,res_save_path =None):
    if model_name in models.keys():
        _model = models[model_name]
    elif model_name in models.values():
        _model = model_name
    else:
        return None
    clf,report,confuse = _model(train_x,train_y,test_x,test_y)
    if model_save_path!=None:
        save_model(clf,model_save_path)
    string = str(confuse[0][0])+","+str(confuse[0][1])+","+str(confuse[1][0])+","+str(confuse[1][1])+ "\n"
    string2 = ""
    for var in report:
        string2+= ','.join(list(map(str,var)))+'\n'
    if res_save_path!=None:
        tools.save_txt(string+string2,res_save_path)
    return string+string2

def predict(model_path,x_predict):
    _model = load_model(model_path)
    result= _model.predict(x_predict)
    return result


def idefine_class(train_file,test_file):
    def load(file,key =0,data =1,seperate=True):
        res ={}
        content = tools.read_txt(file).strip().split("\n")
        for line in content:
            tmp = line.split(",",maxsplit=2)
            if len(tmp[key]) >0:
                if tmp[key] not in res.keys():
                    res[tmp[key]] = []
                if seperate:
                    res[tmp[key]].append(tools.cut_sentence(tmp[data].strip()))
                else:
                    res[tmp[key]].append(tmp[data].strip())
        return res

    def similarity(corpus,line):
        res = 0
        for tmpline in corpus:
            count =0
            for w in tmpline:
                if w in line:
                    count+=1
            # print(tmpline,line,count)
            res+= count/(max(len(tmpline),len(line)))

        return res/len(corpus)

    def classify(train_data,line):
        line = tools.cut_sentence(line)
        max_label, max_value = "", 0
        for label in train_data.keys():
            mtmp = similarity(train_data[label],line)
            if mtmp> max_value:
                max_label = label
                max_value = mtmp
        return max_label

    train_data = load(train_file)
    test_data = load(test_file,key=1,data=0,seperate=False)
    real_type,predict_type =[],[]
    i =0
    for line in test_data.keys():
        i+=1
        tools.show_process(i,len(test_data),num=50)
        real_type.append(test_data[line][0])
        plabel = classify(train_data,line)
        predict_type.append(plabel)
        # print(i,real_type[-1],predict_type[-1])
    # keys= list(set(predict_type))

    tools.save_txt('../res/real_type.txt',content=real_type)
    tools.save_txt('../res/predict_type.txt',content=predict_type)

    # real_type_map,predict_type_map =[],[]
    # for kk in real_type:
    #     real_type_map.append(keys.index(kk))
    # for kk in predict_type:
    #     predict_type_map.append(keys.index(kk))

    report, confuse = analyse_predict_result(real_type,predict_type)
    for line in report:
        print('\t'.join(list(map(str, line))))




def analyse_predict_result(real_y,predict_y):
    report = metrics.classification_report(real_y,predict_y)
    confuse =metrics.confusion_matrix(real_y,predict_y)
    report = [re.split(" {2,20}",var.strip()) for var in report.split("\n")[2:] if var.strip()!=""]
    return report,confuse




if __name__ == "__main__":
    train_files = "../res/seperated_data/train_label.csv"
    test_files = "../res/seperated_data/test_label.csv"
    model_path = "../res/model/naivebayes.m"
    # result = train_model(train_file=train_files,test_file=test_files,model_name="naivebayes",model_save_path=model_path)
    # # print(result)
    # x_test,y_test = load_data(test_files)
    # pre_y_test = predict(model_path,x_test)
    # report,confuse = analyse_predict_result(y_test,pre_y_test)
    # print(confuse)
    # for line in report:
    #     print(fea.get_label_name(int(line[0]))+"\t"+'\t'.join(list(map(str,line[1:]))))
    idefine_class(train_files,test_files)
