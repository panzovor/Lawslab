__author__ = 'E440'



def gender_features(name):
      features = {}
      for letter in 'abcdefghijklmnopqrstuvwxyz':
        features['count(%s)' % letter] = name.lower().count(letter)
        features['has(%s)' % letter] = letter in name.lower()
        features['startswith(%s)' % letter] = (letter==name[0].lower())
        features['endswith(%s)' % letter] = (letter==name[-1].lower())
      return features

if __name__ == "__main__":
    import re
    string = "    "
    print(re.split(" +",string))

    path = "../res/weka/labeled_data.csv"
    import src.tools as tools

    content = tools.read_lines(path)
    res= [var.split(",",maxsplit=2)[0] for var in content]
    print(len(set(res)))
    for lin in set(res):
        print(lin)