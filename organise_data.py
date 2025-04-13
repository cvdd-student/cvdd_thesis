import os
import random
from math import ceil
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def get_data(folder_name, label):
    list_data = []
    for data in os.walk(folder_name):
        if data[0] != folder_name: # index 0 is the base folder, we only want subfolders
            for filename in data[2]:
                filepath = data[0] + "/" + filename
                with open(filepath, "r") as file:
                    item_data = [file.read(), label]
                    list_data.append(item_data)
    
    return list_data


def separate_items_labels(list_data):
    list_items = []
    list_labels = []
    for item, label in list_data:
        list_items.append(item)
        list_labels.append(label)
    
    return list_items, list_labels


def convert_to_feats(train_data, test_data):
    vec_count = CountVectorizer()
    vec_count.fit(train_data)
    train_count_vec = vec_count.transform(train_data)
    test_count_vec = vec_count.transform(test_data)
    
    vec_tfidf = TfidfTransformer()
    vec_tfidf.fit(train_count_vec)
    train_feats = vec_tfidf.transform(train_count_vec)
    test_feats = vec_tfidf.transform(test_count_vec)
    
    return train_feats, test_feats


def process_data(list_data):
    random.shuffle(list_data)

    training_amt = ceil(len(list_data) * 0.8)
    training_data = list_data[:training_amt]
    test_data = list_data[training_amt:]
    
    tr_items, tr_labels = separate_items_labels(training_data)
    te_items, te_labels = separate_items_labels(test_data)
    
    tr_feats, te_feats = convert_to_feats(tr_items, te_items)
    
    return tr_feats, tr_labels, te_feats, te_labels


def train_cls(train_feats, train_labels):
    '''Initiate and fit a SVM classifier, then return
    the classifier.'''
    # Init the classifier, decide what type is the best.
    cls = svm.LinearSVC()  # Accuracy around 0.835 on dev dataset
    #cls = svm.SVC(decision_function_shape='ovo')    # Accuracy around 0.827 on dev dataset

    print("Training model...")
    cls.fit(train_feats, train_labels)
    print("OK")

    return cls


def main():
    gemini_data = get_data("Gemini_Data", "gemini")
    human_data = get_data("Human_Data", "human")
    full_data = gemini_data + human_data
    tr_items, tr_labels, te_items, te_labels = process_data(full_data)
    
    cls = train_cls(tr_items, tr_labels)
    predict = cls.predict(te_items)
    
    eval_matrix = metrics.confusion_matrix(y_true=te_labels, y_pred=predict)
    print(eval_matrix)
    

if __name__ == "__main__":
    main()