import organise_data
import perform_svm
import os
import time


def get_run_results():
    # Get necessary data
    tr_feats, tr_labels, te_feats, te_labels, transformers = organise_data.make_feats()

    # Initiate and train the SVM classifier
    cls = perform_svm.train_cls(tr_feats, tr_labels, quiet_mode=True)

    # Get the sorted dictionary of the classifier
    list_words_ids = []
    for word in transformers[0].vocabulary_:
        list_words_ids.append([transformers[0].vocabulary_[word], word])
    list_words_ids = sorted(list_words_ids)

    # Get the list of classifier coefficients (weights)
    list_weights = []
    for coef in cls.coef_:
        for a in coef:
            list_weights.append(a)

    # Merge the words and weights into one list
    list_wordweights = []
    for [word_id, word], weight in zip(list_words_ids, list_weights):
        list_wordweights.append([weight, word])

    return list_wordweights


def count_word_occurences(list_wordweights, data):
    # Prepare the counter list
    list_counters = []
    for i in range(len(list_wordweights)):
        list_counters.append(0)

    for item, label in data:
        for line in item.split("\n"):
            for i in range(len(list_wordweights)):
                if list_wordweights[i][0] in line:
                    list_counters[i] += 1

    return list_counters


def main():
    int_iterations = 100
    dict_wordweights = dict()
    print("Running SVM classifications...")
    for i in range(int_iterations):
        list_wordweights_new = get_run_results()
        for weight, word in list_wordweights_new:
            if word not in dict_wordweights:
                dict_wordweights[word] = weight
            else:
                dict_wordweights[word] += weight
    for word in dict_wordweights:
        dict_wordweights[word] = dict_wordweights[word] / int_iterations
    print("OK")

    list_wordweights = dict_wordweights.items()
    list_wordweights = sorted(list_wordweights, key=lambda x: x[1])

    # Collect the amount of times the words appear in our data
    print("Collecting word occurences...")
    data_gemini = organise_data.get_data("Gemini_Data", "a")
    data_human = organise_data.get_data("Human_Data", "a")
    counters_gemini = count_word_occurences(list_wordweights, data_gemini)
    counters_human = count_word_occurences(list_wordweights, data_human)
    print("OK")

    list_total_data = []
    for i in range(len(list_wordweights)):
        concat_item = [list_wordweights[i][0], list_wordweights[i][1], counters_gemini[i], counters_human[i]]
        list_total_data.append(concat_item)

    # Export the collected data
    filename = "svm_reports/"
    filename += str(time.time())
    filename += ".tsv"

    print("Exporting data to the svm_reports folder...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        file.write("word\tcoef\tocc_gemini\tocc_human\n")
        for item in list_total_data:
            file.write(str(item[0]) + "\t" + str(item[1]) + "\t" + str(item[2]) + "\t" + str(item[3]))
            file.write("\n")
    print("OK")


if __name__ == "__main__":
    main()
