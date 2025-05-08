from sklearn import svm
from sklearn import metrics
import organise_data
import random


def train_cls(train_feats, train_labels, quiet_mode=False):
    '''Initiate and fit a SVM classifier, then return
    the classifier.'''
    cls = svm.LinearSVC()

    if quiet_mode == False:
        print("Training SVM model...")
    cls.fit(train_feats, train_labels)
    if quiet_mode == False:
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


if __name__ == "__main__":
    main()
