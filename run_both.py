import organise_data
import perform_logreg
import perform_svm
from sklearn import metrics


def main():
    # Get necessary data
    tr_feats, tr_labels, te_feats, te_labels, transformers = organise_data.make_feats()

    # Initiate and train the SVM classifier
    cls = perform_svm.train_cls(tr_feats, tr_labels)
    svm_pre_labels = cls.predict(te_feats)

    # Initiate and train the logreg
    clf = perform_logreg.train_clf(tr_feats, tr_labels)
    clf_predictions = clf.predict_proba(te_feats)

    # Create LogReg labels from the probabilities to classify
    logreg_pre_labels = []
    for prediction in clf_predictions:
        if prediction[0] > 0.5:
            logreg_pre_labels.append("gemini")
        else:
            logreg_pre_labels.append("human")

    # Create and show evaluation matrices
    eval_matrix = metrics.confusion_matrix(y_true=te_labels, y_pred=svm_pre_labels)
    print("##### SVM CLASSIFIER #####")
    print(eval_matrix)

    eval_matrix = metrics.confusion_matrix(y_true=te_labels, y_pred=logreg_pre_labels)
    print("##### LOGREG CLASSIFIER #####")
    print(eval_matrix)


if __name__ == "__main__":
    main()
