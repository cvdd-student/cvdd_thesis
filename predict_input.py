import os
import organise_data
import perform_logreg

def file_select():
    """
    Displays the files in the input folder,
    and returns the file chosen.
    """
    files = os.listdir("input")
    for i in range(len(files)):
        filepath = "input/" + files[i]
        if os.path.isfile(filepath):
            print("[" + str(i) + "]", end=" ")
            print(files[i])

    selected_file_index = input("Which file would you like to process? (Type only number)\n")
    try:
        selected_file_index = int(selected_file_index)
    except ValueError:
        print("Invalid selection")
        exit()

    selected_file = files[selected_file_index]
    return "input/" + selected_file


def get_data():
    """
    Gets a file from file_select() and returns it as read data.
    Input: None
    """
    selected_file = file_select()
    with open(selected_file, "r") as file:
        data = file.read()
    return data


def main():
    tr_feats, tr_labels, te_feats, te_labels, transformers = organise_data.make_feats()
    clf = perform_logreg.train_clf(tr_feats, tr_labels)
    
    input_file = get_data()
    input_count_vec = transformers[0].transform([input_file])
    input_tfidf = transformers[1].transform(input_count_vec)
    
    clf_predictions = clf.predict_proba(input_tfidf)
    print("Probability of being Gemini: " + str(clf_predictions[0][0]))


if __name__ == "__main__":
    main()