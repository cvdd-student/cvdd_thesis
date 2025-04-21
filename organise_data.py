import os
import random
from math import ceil
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from transformers import AutoTokenizer, AutoModel
import torch

from sklearn.linear_model import LogisticRegression

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


def convert_to_feats_codeBERT(train_data, test_data):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    train_feats = tokenizer.tokenize(train_data[0])
    test_feats = tokenizer.tokenize(test_data[0])
    
    train_encoded = []
    for item in train_data:
        item_tokenized = " ".join(tokenizer.tokenize(item))
        train_encoded.append(item_tokenized)
        
    test_encoded = []
    for item in test_data:
        item_tokenized = " ".join(tokenizer.tokenize(item))
        test_encoded.append(item_tokenized)
    
    vec_count = CountVectorizer()
    vec_count.fit(train_encoded)
    train_count_vec = vec_count.transform(train_encoded)
    test_count_vec = vec_count.transform(test_encoded)
    
    vec_tfidf = TfidfTransformer()
    vec_tfidf.fit(train_count_vec)
    train_feats = vec_tfidf.transform(train_count_vec)
    test_feats = vec_tfidf.transform(test_count_vec)
    
    return train_feats, test_feats


def remove_docstrings(data):
    temp_print_item = False
    for item, label in data:
        item_new = ""
        token_index = 0
        flag_copy_token = True
        token_index = 0
        for token in item:
            if token == "'":
                if item[token_index:token_index+3] == "'''" and flag_copy_token is True:
                    flag_copy_token = False
                elif item[token_index-3:token_index] == "'''" and flag_copy_token is False:
                    flag_copy_token = True
                    temp_print_item = True
            if flag_copy_token is True:
                item_new += token
            token_index += 1
        if temp_print_item is True:
            print(item)
            print(item_new)
            exit()
    exit()


def clean_data(data):
    data_out = []
    
    for item, label in data:
        item_new = []
        for line in item.split("\n"):
            flag_skip_line = False
            line_clean = line.replace("    ", "")
            if line_clean[:3] == '"""':
                flag_skip_line = True
            if line_clean[:3] == "'''":
                flag_skip_line = True
            if line_clean[:1] == "#":
                flag_skip_line = True
            if "solve" in line:
                flag_skip_line = True
            
            if flag_skip_line is not True:
                item_new.append(line)
        item_new = "\n".join(item_new)
        data_out.append([item_new, label])
    
    return data_out


def process_data(list_data):
    random.shuffle(list_data)

    training_amt = ceil(len(list_data) * 0.8)
    training_data = list_data[:training_amt]
    test_data = list_data[training_amt:]
    
    tr_items, tr_labels = separate_items_labels(training_data)
    te_items, te_labels = separate_items_labels(test_data)
    
    return tr_items, tr_labels, te_items, te_labels


def train_clf(train_feats, train_labels):
    print("Training Logistic model...")
    clf = LogisticRegression(random_state=0).fit(train_feats, train_labels)
    print("OK")
    
    return clf


def train_cls(train_feats, train_labels):
    '''Initiate and fit a SVM classifier, then return
    the classifier.'''
    # Init the classifier, decide what type is the best.
    cls = svm.LinearSVC()  # Accuracy around 0.835 on dev dataset
    #cls = svm.SVC(decision_function_shape='ovo')    # Accuracy around 0.827 on dev dataset

    print("Training SVM model...")
    cls.fit(train_feats, train_labels)
    print("OK")

    return cls


def main():
    random.seed(31012001)
    gemini_data = get_data("Gemini_Data", "gemini")
    human_data = get_data("Human_Data", "human")
    full_data = gemini_data + human_data
    # full_data = remove_docstrings(full_data)
    full_data = clean_data(full_data)
    tr_items, tr_labels, te_items, te_labels = process_data(full_data)
    
    tr_feats, te_feats = convert_to_feats(tr_items, te_items)
    
    clf = train_clf(tr_feats, tr_labels)
    clf_predictions = clf.predict_proba(te_feats)
    
    list_gemini_prob = []
    pre_labels = []
    for prediction in clf_predictions:
        list_gemini_prob.append(prediction[0])
        if prediction[0] > 0.5:
            pre_labels.append("gemini")
        else:
            pre_labels.append("human")
    
    # accum = 0
    # for real, pred in zip(te_labels, pre_labels):
    #     if real != pred:
    #         print(te_items[accum])
    #         print(real)
    #     accum += 1
    
    eval_matrix = metrics.confusion_matrix(y_true=te_labels, y_pred=pre_labels)
    print(eval_matrix)
    
    print()
    cls = train_cls(tr_feats, tr_labels)
    cls_predictions = cls.predict(te_feats)
    
    eval_matrix = metrics.confusion_matrix(y_true=te_labels, y_pred=cls_predictions)
    print(eval_matrix)
    

if __name__ == "__main__":
    main()