from sklearn import svm
from sklearn import metrics
import organise_data

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
    tr_feats, tr_labels, te_feats, te_labels, transformers = organise_data.make_feats()
    cls = train_cls(tr_feats, tr_labels)
    cls_predictions = cls.predict(te_feats)
    
    eval_matrix = metrics.confusion_matrix(y_true=te_labels, y_pred=cls_predictions)
    print(eval_matrix)


if __name__ == "__main__":
    main()