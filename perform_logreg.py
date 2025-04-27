from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import organise_data

def train_clf(train_feats, train_labels):
    print("Training Logistic model...")
    clf = LogisticRegression(random_state=0).fit(train_feats, train_labels)
    print("OK")
    
    return clf


def main():
    tr_feats, tr_labels, te_feats, te_labels, transformers = organise_data.make_feats()
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
    
    eval_matrix = metrics.confusion_matrix(y_true=te_labels, y_pred=pre_labels)
    print(eval_matrix)


if __name__ == "__main__":
    main()