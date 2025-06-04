from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import organise_data
import random


def train_clf(train_feats, train_labels):
    '''Initiate and fit a LogReg model, then return
    the classifier.'''
    print("Training Logistic model...")
    clf = LogisticRegression(random_state=0).fit(train_feats, train_labels)
    print("OK")

    return clf


def main():
    random.seed("31012001")

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
    
    # OUT OF SCOPE Data
    print("\n##### 2022 DATA #####\n")
    eval_2022_feats, eval_2022_labels = organise_data.make_2022_feats(transformers)
    pre_2022_predictions = clf.predict_proba(eval_2022_feats)
    eval_2022_baseline_labels = []
    for i in range(len(eval_2022_labels)):
        eval_2022_baseline_labels.append("human")
    
    # Create OUT OF SCOPE labels from the probabilities to classify
    pre_2022_labels = []
    for prediction in pre_2022_predictions:
        if prediction[0] > 0.5:
            pre_2022_labels.append("gemini")
        else:
            pre_2022_labels.append("human")
    
    # OUT OF SCOPE metrics
    eval_2022_accuracy_baseline = metrics.accuracy_score(y_true=eval_2022_labels, y_pred=eval_2022_baseline_labels)
    eval_2022_precision_baseline = metrics.precision_score(y_true=eval_2022_labels, y_pred=eval_2022_baseline_labels, average='macro', zero_division=0.0)
    eval_2022_recall_baseline = metrics.recall_score(y_true=eval_2022_labels, y_pred=eval_2022_baseline_labels, average='macro')
    eval_2022_f1score_baseline = metrics.f1_score(y_true=eval_2022_labels, y_pred=eval_2022_baseline_labels, average='macro', zero_division=0.0)
    
    eval_2022_accuracy = metrics.accuracy_score(y_true=eval_2022_labels, y_pred=pre_2022_labels)
    eval_2022_precision = metrics.precision_score(y_true=eval_2022_labels, y_pred=pre_2022_labels, average='macro', zero_division=0.0)
    eval_2022_recall = metrics.recall_score(y_true=eval_2022_labels, y_pred=pre_2022_labels, average='macro')
    eval_2022_f1score = metrics.f1_score(y_true=eval_2022_labels, y_pred=pre_2022_labels, average='macro', zero_division=0.0)

    print("Accuracy: " + str(eval_2022_accuracy) + " (Baseline: " + str(eval_2022_accuracy_baseline) + ")")
    print("Precision: " + str(eval_2022_precision)  + " (Baseline: " + str(eval_2022_precision_baseline) + ")")
    print("Recall: " + str(eval_2022_recall)  + " (Baseline: " + str(eval_2022_recall_baseline) + ")")
    print("F1: " + str(eval_2022_f1score)  + " (Baseline: " + str(eval_2022_f1score_baseline) + ")")

    # Create and show OUT OF SCOPE evaluation matrix
    eval_matrix = metrics.confusion_matrix(y_true=eval_2022_labels, y_pred=pre_2022_labels)
    print(eval_matrix)


if __name__ == "__main__":
    main()
