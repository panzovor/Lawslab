<<<<<<< HEAD
__author__ = 'E440'
import src.preprocess as preprocess
import src.tools as tools
=======
__author__ = 'czb'
import src.preprocess as preprocess
import src.tools as tools
import re
>>>>>>> 5cc3198d18f23a94b8c0627b7c063d8a24028613

def distribution(data_path,label_path,save_path):
    content = tools.read_txt(data_path)
    labels = preprocess.load_labels(label_path)
    label_no_line,no_label_line,label_line =  preprocess.__label_content(content,labels)
    save_content =[["label","num"]]
    for label in label_line.keys():
        save_content.append([label[1:-4],str(len(label_line[label]))])

    tools.save_txt(save_path,save_content)

<<<<<<< HEAD

=======
def cound_labels(filepath):
    labels =set()
    with open(filepath,encoding="utf-8") as file:
        for line in  file.readlines():
            labels.add(line.split(",")[0])
    labels = list(labels)
    labels.sort()
    # print("\n".join(labels))
    print(len(labels))

def calculate_pre_recall(matrix):
    result = []
    for i in range(len(matrix)):
        tmp =[]
        line = matrix[i]
        tmp.append(line[i])
        tmp.append(sum(line))
        judge_count=0
        for j in range(len(matrix)):
            judge_count+=matrix[j][i]
        tmp.append(judge_count)
        tmp.append(tmp[0]/tmp[2] if tmp[2] >0 else 0)
        tmp.append(tmp[0]/tmp[1] if tmp[1] >0 else 0)
        tmp.append(2 * tmp[-1] * tmp[-2] / (tmp[-1] + tmp[-2]) if (tmp[-1]+tmp[-2]) >0 else 0)
        result.append(tmp)
    return result




def analyze_weka_result(filepath,label_path="../res/parameter/labels.txt"):
    content = tools.read_txt(filepath).strip()
    matrix_label = "=== Confusion Matrix ==="
    labels_con = tools.read_txt(label_path)
    labels = []
    for line in labels_con.strip().split("\n"):
        labels.append(line)
    start_index = content.rindex(matrix_label)+len(matrix_label)
    matrix_content = content[start_index:].strip()
    result = []
    result_labels = []
    for line in matrix_content.split("\n"):
        line = line.strip()
        if "|" in line:
            index = int(line[line.index("|"):].split(" ")[-1])
            mat = line[:line.index("|")].strip()
            tmp = [int(var) for var in re.split(" {1,10}",mat) if var.strip()!=""]
            result_labels.append(labels[index])
            result.append(tmp)

    pre_rescall = calculate_pre_recall(result)
    res ={}
    for i in range(len(result_labels)):
        if result_labels[i] not in res.keys():
            res[result_labels[i]] = result[i]+pre_rescall[i]

    tmp =[]
    for l in res.keys():
        tmp.append([l]+[str(var) for var in res[l][-6:]])
    tmp = sorted(tmp,key=lambda  d:float(d[2]) ,reverse=True)

    for t in tmp:
        print('\t'.join(t))
    return res

def count_words_words_size():
    path = "../res/labeled_data/label_data.csv"
    content = tools.read_txt(path)
    lines = []
    words = []
    tmp = content.split("\n")
    tmp.pop(0)
    for line in tmp:
        sen = line.split(",")[-1]
        lines.append(sen)
        words.extend(tools.cut_sentence(sen))

    print(len(words),len(set(words)))
>>>>>>> 5cc3198d18f23a94b8c0627b7c063d8a24028613

if __name__ == "__main__":
    data_path = "../res/data/已完成标注（6818）.txt"
    label_path = "../res/data/关键因子.txt"
    save_path = "../res/analysis_result/label_sample_num.csv"

<<<<<<< HEAD
    distribution(data_path,label_path,save_path)
=======
    # distribution(data_path,label_path,save_path)

    # cound_labels("../res/labeled_data/lnline.csv")
    # cound_labels("../res/seperated_data/test_label.csv")
    # cound_labels("../res/seperated_data/train_label.csv")
    # analyze_weka_result(filepath="../res/model/model_result/BayesNet")
    count_words_words_size()
>>>>>>> 5cc3198d18f23a94b8c0627b7c063d8a24028613
