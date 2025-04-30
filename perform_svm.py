from sklearn import svm
from sklearn import metrics
import organise_data


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

    # Initiate and train the SVM classifier
    cls = train_cls(tr_feats, tr_labels)
    pre_labels = cls.predict(te_feats)

    # Create and show evaluation matrix
    eval_matrix = metrics.confusion_matrix(y_true=te_labels, y_pred=pre_labels)
    print(eval_matrix)

    list_words = []
    for word in transformers[0].vocabulary_:
        list_words.append(word)
    print(list_words)
    exit()

    list_weights = []
    for coef in cls.coef_:
        for a in coef:
            list_weights.append(a)
    
    list_wordweights = []
    for word, weight in zip(list_words, list_weights):
        list_wordweights.append([weight, word])
    
    list_wordweights.sort()
    print(list_wordweights[-10:])
    
    for item in list_wordweights:
        if item[1] == "heappop":
            print(item[0])


if __name__ == "__main__":
    main()
