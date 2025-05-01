from sklearn import svm
from sklearn import metrics
import organise_data
import random


def train_cls(train_feats, train_labels):
    '''Initiate and fit a SVM classifier, then return
    the classifier.'''
    cls = svm.LinearSVC()

    print("Training SVM model...")
    cls.fit(train_feats, train_labels)
    print("OK")

    return cls


def main():
    # Get necessary data
    tr_feats, tr_labels, te_feats, te_labels, transformers = organise_data.make_feats()

    random.seed()

    # Initiate and train the SVM classifier
    cls = train_cls(tr_feats, tr_labels)
    pre_labels = cls.predict(te_feats)

    # Create and show evaluation matrix
    eval_matrix = metrics.confusion_matrix(y_true=te_labels, y_pred=pre_labels)
    print(eval_matrix)

    list_words_ids = []
    for word in transformers[0].vocabulary_:
        list_words_ids.append([transformers[0].vocabulary_[word], word])
    list_words_ids = sorted(list_words_ids)
    
    list_words = []
    for thingie in list_words_ids:
        list_words.append(thingie[1])

    list_weights = []
    for coef in cls.coef_:
        for a in coef:
            list_weights.append(a)
    
    list_wordweights = []
    for word, weight in zip(list_words, list_weights):
        list_wordweights.append([weight, word])
    
    list_wordweights.sort()
    print(list_wordweights[:10])


if __name__ == "__main__":
    main()
