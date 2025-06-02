import os
import random
from math import ceil
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from transformers import AutoTokenizer, AutoModel
# import torch


def get_data(folder_name, label):
    '''Reads all files in a provided folder, then
    stores them into a list of lists with the provided label:
    [[item, label]]'''
    list_data = []
    for data in os.walk(folder_name):
        if data[0] != folder_name:  # index 0 is the base folder, we only want subfolders
            for filename in data[2]:
                filepath = data[0] + "/" + filename
                with open(filepath, "r") as file:
                    item_data = [file.read(), label]
                    list_data.append(item_data)

    return list_data


def separate_items_labels(list_data):
    '''Separates a list of lists [[item, label]] into
    separate lists of items and labels.'''
    list_items = []
    list_labels = []
    for item, label in list_data:
        list_items.append(item)
        list_labels.append(label)

    return list_items, list_labels


def convert_to_feats(train_data, test_data):
    '''Converts training and testing data into
    features, using sklearn's CountVectorizer and
    TfidfTransformer.
    Returns the created features, as well as the vectorizers
    (vectorizers are stored together in a list).'''

    # Perform count vectorizing
    vec_count = CountVectorizer()
    vec_count.fit(train_data)
    train_count_vec = vec_count.transform(train_data)
    test_count_vec = vec_count.transform(test_data)

    # Perform Tfidf transformation
    vec_tfidf = TfidfTransformer()
    vec_tfidf.fit(train_count_vec)
    train_feats = vec_tfidf.transform(train_count_vec)
    test_feats = vec_tfidf.transform(test_count_vec)

    return train_feats, test_feats, [vec_count, vec_tfidf]


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
    '''Non-functional function to detect and remove docstrings.'''
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
    '''Removes various unwanted lines from the
    input texts. This includes docstrings, comments,
    and any line with the "solve" string.
    This is mostly for experimental purposes.'''
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
            #if "solve" in line:
            #    flag_skip_line = True

            if flag_skip_line is not True:
                item_new.append(line)
        item_new = "\n".join(item_new)
        data_out.append([item_new, label])

    return data_out


def process_data(list_data):
    '''Shuffles and divides a list of lists [item, label] into
    four separate lists: Test items, test labels,
    training items and training labels. These are
    then returned.
    80% of data gets used for the training data, the
    remaining 20% is used for the test data.'''

    # Shuffle the data to mix the different data labels
    random.shuffle(list_data)

    # Split into training and testing data
    training_amt = ceil(len(list_data) * 0.8)
    training_data = list_data[:training_amt]
    test_data = list_data[training_amt:]

    # Split the items from the labels into separate lists
    tr_items, tr_labels = separate_items_labels(training_data)
    te_items, te_labels = separate_items_labels(test_data)

    return tr_items, tr_labels, te_items, te_labels


def make_feats():
    # Set the seed to keep consistent results
    #random.seed(31012001)

    # Get the data
    gemini_data = get_data("Gemini_Data", "gemini")
    human_data = get_data("Human_Data", "human")

    # Merge the data and clean it up
    full_data = gemini_data + human_data
    full_data = clean_data(full_data)

    # Process the data and create the features and transformers
    tr_items, tr_labels, te_items, te_labels = process_data(full_data)
    tr_feats, te_feats, transformers = convert_to_feats(tr_items, te_items)

    return tr_feats, tr_labels, te_feats, te_labels, transformers


def make_2022_feats(transformers):
    # Get the data
    gemini_data = get_data("2022_Data/Gemini", "gemini")
    human_data = get_data("2022_Data/Human", "human")

    # Merge the data and clean it up
    full_data = gemini_data + human_data
    full_data = clean_data(full_data)

    random.shuffle(full_data)
    eval_2022_items, eval_2022_labels = separate_items_labels(full_data)
    eval_2022_count_vec = transformers[0].transform(eval_2022_items)
    eval_2022_feats = transformers[1].transform(eval_2022_count_vec)
    
    return eval_2022_feats, eval_2022_labels