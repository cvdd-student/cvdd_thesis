from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import organise_data


def train_clf(train_feats, train_labels):
    '''Initiate and fit a LogReg model, then return
    the classifier.'''
    print("Training Logistic model...")
    clf = LogisticRegression(random_state=0).fit(train_feats, train_labels)
    print("OK")

    return clf


def main():
    # Get necessary data
    tr_feats, tr_labels, te_feats, te_labels, transformers = organise_data.make_feats()

    # Initiate and train the logreg
    clf = train_clf(tr_feats, tr_labels)
    clf_predictions = clf.predict_proba(te_feats)

    # Create labels from the probabilities to classify
    pre_labels = []
    for prediction in clf_predictions:
        if prediction[0] > 0.5:
            pre_labels.append("gemini")
        else:
            pre_labels.append("human")
    
    # Evaluation metrics
    baseline_labels = []
    for i in range(len(te_labels)):
        baseline_labels.append("human")
    
    accuracy_baseline = metrics.accuracy_score(y_true=te_labels, y_pred=baseline_labels)
    precision_baseline = metrics.precision_score(y_true=te_labels, y_pred=baseline_labels, average='macro', zero_division=0.0)
    recall_baseline = metrics.recall_score(y_true=te_labels, y_pred=baseline_labels, average='macro')
    f1score_baseline = metrics.f1_score(y_true=te_labels, y_pred=baseline_labels, average='macro', zero_division=0.0)
    
    accuracy = metrics.accuracy_score(y_true=te_labels, y_pred=pre_labels)
    precision = metrics.precision_score(y_true=te_labels, y_pred=pre_labels, average='macro', zero_division=0.0)
    recall = metrics.recall_score(y_true=te_labels, y_pred=pre_labels, average='macro')
    f1score = metrics.f1_score(y_true=te_labels, y_pred=pre_labels, average='macro', zero_division=0.0)
    
    print("Accuracy: " + str(accuracy) + " (Baseline: " + str(accuracy_baseline) + ")")
    print("Precision: " + str(precision)  + " (Baseline: " + str(precision_baseline) + ")")
    print("Recall: " + str(recall)  + " (Baseline: " + str(recall_baseline) + ")")
    print("F1: " + str(f1score)  + " (Baseline: " + str(f1score_baseline) + ")")

    # Create and show evaluation matrix
    eval_matrix = metrics.confusion_matrix(y_true=te_labels, y_pred=pre_labels)
    print(eval_matrix)


if __name__ == "__main__":
    main()
